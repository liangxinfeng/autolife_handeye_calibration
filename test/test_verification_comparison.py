"""Camera-extrinsic A/B comparison. All robot/camera state is synthetic."""
import copy
import time
import numpy as np
import pytest
from test_ui import win
from test_verification import data_for
from autolife_handeye_calibration_306.core import TORSO,MOUNT,NECK
from autolife_handeye_calibration_306.verification_core import (
    camera_geometry,camera_transform,load_geometry,observed_target)
from autolife_handeye_calibration_306.depth_target import torso_point


def load_pair(w):
    d=data_for(w);d['report'].update(status='test',validation={})
    w.session.data=d;w.verification.load()
    return w.verification,d


@pytest.mark.parametrize('pitch,yaw',[(0,0),(-20,15),(15,-25)])
def test_urdf_uses_dynamic_head_fk_and_nominal_optical_axes(win,pitch,yaw):
    d=data_for(win);d['_verification_camera_source']='urdf'
    q=dict(win.node._current);q[NECK[1]]=pitch;q[NECK[2]]=yaw
    X=np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
    H=win.fk.fk(TORSO,MOUNT,q)
    assert np.allclose(camera_transform(d,win.fk,q,win.camera.frame),H@X)
    # Arbitrary fitted extrinsic cannot leak into URDF mode.
    d['report']['T_mount_camera'][0][3]=99.
    assert np.allclose(camera_transform(d,win.fk,q,win.camera.frame),H@X)


def test_switch_invalidates_plan_and_depth_without_modifying_result(win):
    v,d=load_pair(win);original=copy.deepcopy(d)
    before=load_geometry(v.data,win.fk,15)
    v.plan={'old':True};v.start_button.setEnabled(True)
    v.depth.result={'old':True};v.depth.pixel=(10,20)
    v.camera_source.setCurrentIndex(1)
    assert v.data['_verification_camera_source']=='urdf'
    assert v.plan is None and not v.start_button.isEnabled() and v.depth.result is None
    assert v.depth.pixel is None and 'URDF' in v.source_hint.text()
    assert np.array_equal(before,load_geometry(v.data,win.fk,15))
    assert d==original and not win.node.sent
    v.load();assert v.data['_verification_camera_source']=='urdf'
    v.camera_source.setCurrentIndex(0)
    assert v.data['_verification_camera_source']=='calibrated' and d==original


@pytest.mark.parametrize('state',['selecting','planning','moving'])
def test_cannot_switch_during_active_experiment(win,state):
    v,_=load_pair(win);v.state=state;v.controls(False)
    assert not v.camera_source.isEnabled()
    v.camera_source.setCurrentIndex(1)
    assert v.camera_source.currentData()=='calibrated'
    assert v.data['_verification_camera_source']=='calibrated' and not win.node.sent


def test_both_target_types_use_selected_extrinsic(win):
    v,_=load_pair(win);d=v.data;f=win.camera.frame
    q=dict(win.node._current);now=time.monotonic();f['received']=now
    f['detection']['C'][0][3]=.03;f['detection']['C'][2][3]=.8
    intr={'fx':600.,'fy':600.,'ppx':320.,'ppy':240.,'model_id':2,'coeffs':[0.]*5}
    f.update(depth=np.full((480,640),800,np.uint16),depth_id=f['frame_id'],intrinsics=intr,
             depth_intrinsics=intr.copy(),depth_scale_m=.001)
    targets=[]
    for source in ('calibrated','urdf'):
        d['_verification_camera_source']=source
        C=camera_transform(d,win.fk,q,f)
        T,goal=observed_target(d,win.fk,q,f,30)
        point,_,_=torso_point(d,win.fk,[(now,q)],f,(320,240),now)
        assert np.allclose(T,C@np.array(f['detection']['C']))
        assert np.allclose(point,(C@np.array([0,0,.8,1]))[:3])
        assert np.linalg.norm(goal-T[:3,3])==pytest.approx(.03)
        targets.append(T)
    assert not np.allclose(*targets)


def test_urdf_keeps_intrinsics_and_robot_checks(win):
    d=data_for(win);d['_verification_camera_source']='urdf'
    f=copy.deepcopy(win.camera.frame);f['K'][0,0]+=1
    with pytest.raises(ValueError,match='内参'):camera_transform(d,win.fk,win.node._current,f)
    d['urdf_sha256']='mismatch'
    with pytest.raises(ValueError,match='URDF'):camera_transform(d,win.fk,win.node._current,win.camera.frame)
    d['_verification_camera_source']='typo'
    with pytest.raises(ValueError,match='未知'):camera_geometry(d)


def test_stale_source_cannot_execute_old_plan(win):
    v,_=load_pair(win);v.comparison=camera_geometry(v.data)
    v.plan={'old':True};v.data['_verification_camera_source']='urdf'
    v.start()
    assert not win.node.sent and v.plan is None and '外参来源已改变' in v.status.text()


def test_metadata_contains_the_exact_extrinsic_and_fixed_tcp_scope(win):
    v,_=load_pair(win);v.camera_source.setCurrentIndex(1)
    meta=camera_geometry(v.data)
    assert meta['camera_source']=='urdf' and meta['camera_mount_frame']==MOUNT
    assert meta['urdf_sha256']==win.fk.sha and meta['comparison_scope']=='camera_extrinsic_only'
    assert np.array(meta['T_mount_camera_used']).shape==(4,4)
    assert 'same_calibrated' in meta['tcp_source']
