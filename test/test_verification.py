import copy
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_ui import win
from autolife_handeye_calibration_306.core import ARMS, NECK, matrix
from autolife_handeye_calibration_306.verification_core import (
    load_geometry, tcp_position, checked_observation, check_path_state)


def data_for(win):
    data=copy.deepcopy(win.session.data)
    data['config']['calibration_gripper_deg']=240.
    data['report']={key:np.eye(4).tolist() for key in ('T_torso_camera','T_mount_camera','T_wrist_marker')}
    f=win.camera.frame
    data['samples']={'0':{'image_shape':list(f['image'].shape),'K':f['K'].tolist(),'D':f['D'].tolist()}}
    return data


def test_tcp_15mm_is_rotated_marker_offset_not_camera_edit(win):
    data=data_for(win); Y=matrix(Rotation.from_euler('x',90,degrees=True).as_matrix(),[.22,-.02,-.05])
    data['report']['T_wrist_marker']=Y.tolist(); original=copy.deepcopy(data)
    tcp=load_geometry(data,win.fk,15)
    assert np.allclose(tcp,[.22,-.005,-.05])
    assert data==original
    q=dict(win.node._current); q[ARMS['right'][0]]=20
    delta=tcp_position(win.fk,data['config'],q,tcp)-tcp_position(win.fk,data['config'],q,Y[:3,3])
    assert np.linalg.norm(delta)==pytest.approx(.015)


def test_reject_wrong_robot_or_urdf(win):
    data=data_for(win);data['urdf_sha256']='old'
    with pytest.raises(ValueError,match='URDF'):load_geometry(data,win.fk)


def test_pairing_and_intrinsics_gate(win):
    data=data_for(win);f=win.camera.frame; f['detection']['C'][2][3]=.6
    now=time.monotonic(); f['received']=now;history=[(now,win.node._current)]
    target,goal=checked_observation(data,win.fk,history,f,now,30)
    assert np.linalg.norm(target[:3,3]-goal)==pytest.approx(.03)
    with pytest.raises(ValueError,match='过期'):checked_observation(data,win.fk,history,f,now+1,30)
    with pytest.raises(ValueError,match='配对'):checked_observation(data,win.fk,[(now-.3,win.node._current)],f,now,30)
    f['K'][0,0]+=20
    with pytest.raises(ValueError,match='内参'):checked_observation(data,win.fk,history,f,now,30)


def test_path_state_rejects_fixed_joint_motion():
    start={'arm':2.,'neck':0.,'Joint_Left_Gripper':240.};end={**start,'arm':4.}
    check_path_state({**start,'arm':3.},start,end,'arm')
    with pytest.raises(ValueError):check_path_state({**start,'neck':2.},start,end,'arm')
    with pytest.raises(ValueError):check_path_state({**start,'Joint_Left_Gripper':245},start,end,'arm')


def test_new_page_is_readonly_and_gates_gripper_neck_other_arm(win):
    v=win.verification
    assert win.pages.count()==4 and not v.active and not win.node.sent
    v.data=data_for(win);v.state='planning'
    assert win.motion_gate('right_arm')
    v.state='moving';v.edge={'joint':ARMS['right'][0]}
    assert not win.motion_gate('right_arm')
    assert win.motion_gate('grippers') and win.motion_gate('neck') and win.motion_gate('left_arm')
    old=win.pages.currentIndex();win.goto_page(0);assert win.pages.currentIndex()==old
    win.stop();assert not v.active and not v.plan


@pytest.mark.parametrize('opening',[0.,120.,240.,300.])
def test_any_stationary_gripper_opening_allowed(win,monkeypatch,opening):
    v=win.verification;v.data=data_for(win)
    q=win.node._current;q['Joint_Right_Gripper']=opening
    win.node.snapshot=lambda:dict(q)
    win.node._hardware_prepare=False
    now=time.monotonic();win.node._feedback_monotonic=now
    win.node.history=[(now,q)]
    win.camera.frame['received']=now;win.camera.frame['detection']['C'][2][3]=.6
    monkeypatch.setattr('autolife_handeye_calibration_306.verification.sample_guard',lambda *_:'')
    actual,target,goal=v.snapshot(win.camera.frame,True)
    assert actual['Joint_Right_Gripper']==opening and not win.node.sent
    assert np.isfinite(target).all() and np.isfinite(goal).all()


def test_preview_configuration_invalidates_motion(win):
    v=win.verification;v.plan={'anything':1};v.start_button.setEnabled(True)
    v.standoff.setValue(0)
    assert v.plan is None and not v.start_button.isEnabled() and not win.node.sent


def test_execution_ignores_rgb_loss_but_feedback_loss_stops(win,monkeypatch,tmp_path):
    v=win.verification;v.data=data_for(win);win.pages.setCurrentIndex(3)
    q=dict(win.node._current);name=ARMS['right'][0];end={**q,name:1.}
    win.node._current=q;win.node._hardware_ready=True;win.node.get_parameter=lambda _:NS(value=.3)
    v.plan={'waypoints':[q,end],'execution':'continuous_joint_path_v1','end_q':end,'tcp':[.2,0,0],'goal':[0,0,0]}
    v.state='moving';v.source=q;v.index=0;v.edge=None;v.marker_id=0
    v.started=time.monotonic();v.target=np.eye(4);v.run_path=tmp_path/'run'
    from autolife_handeye_calibration_306.verification_trajectory import JointPath,PathFollower
    v.follower=PathFollower(JointPath(ARMS['right'],[q,end]),time.monotonic()-.1);v.settled_since=None
    monkeypatch.setattr(v,'snapshot',lambda *_:(q,np.eye(4),np.zeros(3)))
    def apply(group,values,speed_scale):
        win.node.sent.append((group,values,speed_scale));win.node._active_goals[group]=True
    win.node.apply=apply
    v.tick(win.camera.frame)
    assert len(win.node.sent)==1 and win.node.sent[0][0]=='right_arm'
    assert set(win.node.sent[0][1])==set(ARMS['right']) and win.node.sent[0][2]==1.
    win.camera.frame['received']=time.monotonic()-1
    locked=[];win.node._lock=lambda reason:locked.append(reason)
    v.tick(win.camera.frame)
    assert not locked and v.active
    v.tick(None)
    assert not locked and v.active and '已锁定' in v.pose.text()
    # Changing/losing the detector after start cannot replace the target.
    original=v.target.copy();win.camera.frame['received']=time.monotonic()
    win.camera.frame['detection'].update(ok=False,marker_id=99,reason='未找到')
    v.tick(win.camera.frame)
    assert np.array_equal(v.target,original) and v.active
    win.node._feedback_monotonic=time.monotonic()-1
    v.tick(None)
    assert locked and not v.active and not v.plan


def test_completion_without_rgb_saves_result(win,monkeypatch,tmp_path):
    import json
    from autolife_handeye_calibration_306.verification_trajectory import JointPath,PathFollower
    v=win.verification;v.data=data_for(win);win.pages.setCurrentIndex(3)
    q=dict(win.node._current);names=ARMS['right'];v.source=q
    v.plan={'waypoints':[q,q],'execution':'continuous_joint_path_v1','end_q':q,'tcp':[.2,0,0],'goal':[0,0,0]}
    v.target=np.eye(4);v.marker_id=0;v.state='moving';v.edge={'continuous':True}
    v.started=time.monotonic();v.run_path=tmp_path/'finished';v.settled_since=v.started-.5
    v.follower=PathFollower(JointPath(names,[q,q]),v.started)
    v.follower.progress=v.follower.path.length;v.follower.end_since=v.started
    win.node._hardware_ready=True;win.node.get_parameter=lambda _:NS(value=.3)
    v.tick(None)
    result=json.loads((v.run_path/'result.json').read_text())
    assert result['status']=='completed' and not v.active
    assert '未保存' in result['end_image_status'] and not (v.run_path/'after.png').exists()
    assert not win.node.sent
