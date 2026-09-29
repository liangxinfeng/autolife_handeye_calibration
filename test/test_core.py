import copy
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation as R
from autolife_handeye_calibration_307.core import (matrix,compute_report,object_corners,pose_error,
    Session,sample_guard,NECK,BODY,ARMS,detect)

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config/experiment.yaml').read_text())
CFG['arm']='right' # Explicit fixture; production default is the user's left hand.

def dataset(seed=1):
    rng=np.random.default_rng(seed)
    X=matrix(R.from_euler('xyz',[.1,-.2,.3]).as_matrix(),[.3,.1,.5])
    Y=matrix(R.from_euler('xyz',[.2,.1,-.3]).as_matrix(),[.05,.03,.09])
    H=matrix(R.from_euler('xyz',[.1,.15,.4]).as_matrix(),[.2,.05,.45])
    K=np.array([[620.,0,320],[0,620,240],[0,0,1]])
    samples={}
    for i in range(38):
        C=matrix(R.from_rotvec(rng.uniform(-.5,.5,3)).as_matrix(),rng.uniform([-.1,-.1,.4],[.1,.1,.7]))
        h=H if i<32 else H@matrix(R.from_rotvec(rng.uniform(-.2,.2,3)).as_matrix(),[0,0,0])
        x=X if i<32 else h@np.linalg.inv(H)@X
        G=x@C@np.linalg.inv(Y)
        corners=cv2.projectPoints(object_corners(CFG['marker_size_mm']/1000),cv2.Rodrigues(C[:3,:3])[0],C[:3,3],K,np.zeros(5))[0].reshape(-1,2)
        samples[str(i)]={'slot':i,'phase':'train' if i<24 else 'validation' if i<32 else 'head',
          'G':G.tolist(),'H':h.tolist(),'C':C.tolist(),'K':K.tolist(),'D':[0.]*5,'corners':corners.tolist()}
    return {'config':CFG,'samples':samples},X,Y

def test_eye_to_hand_direction_and_dynamic_neck():
    data,X,Y=dataset(); r=compute_report(data)
    assert pose_error(np.array(r['T_torso_camera']),X)[0]<1e-4
    assert pose_error(np.array(r['T_wrist_marker']),Y)[1]<1e-4
    assert r['fixed_head_pass'] and r['moving_head_pass']

def test_outlier_independent_validation_fails_and_identifies_slot():
    data,_,_=dataset(); data['samples']['25']['G'][0][3]+=.10
    r=compute_report(data)
    assert not r['fixed_head_pass']
    assert next(s for s in r['samples'] if s['slot']==25)['replace_suggested']

def test_head_failure_does_not_claim_dynamic_accuracy():
    data,_,_=dataset()
    for i in range(32,38): data['samples'][str(i)]['H'][1][3]+=.05
    r=compute_report(data); assert r['fixed_head_pass'] and not r['moving_head_pass']

def test_insufficient_and_degenerate_data():
    data,_,_=dataset(); data['samples']={k:v for k,v in data['samples'].items() if int(k)<8}
    with pytest.raises(ValueError,match='不足'):compute_report(data)
    data,_,_=dataset()
    for s in data['samples'].values():s['G']=np.eye(4).tolist()
    with pytest.raises(ValueError,match='旋转'):compute_report(data)

def test_replace_preserves_revision_and_invalidation(tmp_path):
    s=Session(tmp_path/'session',CFG,{},'hash')
    im=np.full((40,40,3),255,np.uint8)
    assert s.record(0,im,{'x':1})==1
    old=s.data['samples']['0']['image'];s.data['report']={'old':True};s.data['fixed_solution']={}
    assert s.record(0,im,{'x':2})==1
    assert s.data['report'] is None and 'fixed_solution' not in s.data
    assert (s.path/old).exists() and len(s.data['revisions'])==1
    assert Session.load(s.path).data['samples']['0']['x']==2

def test_fixed_jaw_neck_and_stability_guards():
    q={k:0. for k in BODY+NECK+ARMS['right']+['Joint_Right_Gripper']}
    hist=[(t,dict(q)) for t in np.linspace(1,2,100)]
    assert not sample_guard(q,q,'train',CFG,hist,2.)
    changed={**q,'Joint_Right_Gripper':2.}
    assert 'Gripper' in sample_guard(changed,q,'train',CFG,hist,2.)
    changed={**q,'Joint_Neck_Pitch':3.}
    assert 'Pitch' in sample_guard(changed,q,'train',CFG,hist,2.)
    hist2=[(t,dict(changed)) for t in np.linspace(1,2,100)]
    assert not sample_guard(changed,q,'head',CFG,hist2,2.)
    assert sample_guard(q,q,'train',CFG,[(1,dict(q))],2.)

def test_marker_id_size_detection_and_duplicate_guard():
    a=cv2.aruco; dictionary=a.getPredefinedDictionary(a.DICT_6X6_250)
    if hasattr(a,'generateImageMarker'):marker=a.generateImageMarker(dictionary,0,240)
    else:marker=a.drawMarker(dictionary,0,240)
    im=np.full((600,800,3),255,np.uint8);im[150:390,200:440]=marker[:,:,None]
    K=np.array([[600.,0,400],[0,600,300],[0,0,1]])
    result=detect(im,K,np.zeros(5),CFG)
    assert 'C' in result and result['edge_px']>230
    im[150:390,500:740]=marker[:,:,None]
    assert not detect(im,K,np.zeros(5),CFG)['ok']

def test_auto_id_and_ambiguous_multi_marker():
    ar=cv2.aruco;dictionary=ar.getPredefinedDictionary(ar.DICT_6X6_250)
    create=ar.generateImageMarker if hasattr(ar,'generateImageMarker') else ar.drawMarker
    im=np.full((600,800,3),255,np.uint8);im[150:390,100:340]=create(dictionary,42,240)[:,:,None]
    K=np.array([[600.,0,400],[0,600,300],[0,0,1]])
    cfg={**CFG,'marker_id':-1};r=detect(im,K,np.zeros(5),cfg)
    assert r['marker_id']==42 and r['detected_ids']==[42]
    im[150:390,450:690]=create(dictionary,17,240)[:,:,None]
    r=detect(im,K,np.zeros(5),cfg)
    assert not r['ok'] and sorted(r['detected_ids'])==[17,42]
    r=detect(im,K,np.zeros(5),{**cfg,'marker_id':42});assert r['marker_id']==42
