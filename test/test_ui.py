"""Qt workflow simulation with a fake camera and fake robot: no hardware calls."""
import copy
import time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import yaml
from PyQt5 import QtWidgets as Q
from autolife_handeye_calibration_306.core import Session,NECK,BODY,ARMS
from autolife_handeye_calibration_306.app import Window
import autolife_handeye_calibration_306.app as module

ROOT=Path(__file__).resolve().parents[1]
class FakeCamera:
    def __init__(self,cfg):
        self.config=cfg
        self.frame={'image':np.full((480,640,3),255,np.uint8),'K':np.array([[600.,0,320],[0,600,240],[0,0,1]]),
            'D':np.zeros(5),'frame_id':1,'received':time.monotonic(),'config':cfg,
            'detection':{'ok':True,'reason':'test','marker_id':0,'detected_ids':[0],'C':np.eye(4).tolist(),'corners':[[240,160],[400,160],[400,320],[240,320]],
            'edge_px':160.,'reprojection_px':.1},'time_source':'synthetic'}
    def get(self):return self.frame,''
    def configure(self,c):self.config=c;self.frame['config']=c
    def close(self):pass

@pytest.fixture
def win(tmp_path,monkeypatch):
    app=Q.QApplication.instance() or Q.QApplication([])
    cfg=yaml.safe_load((ROOT/'config/experiment.yaml').read_text())
    cfg['arm']='right' # Existing manual-workflow tests explicitly exercise right arm.
    q={n:0. for n in BODY+NECK+ARMS['left']+ARMS['right']+['Joint_Left_Gripper','Joint_Right_Gripper']}
    n=NS(_current=q,_last_event='',_hardware_ready=False,_hardware_enabled=False,_active_goals={},
        _feedback_monotonic=time.monotonic(),history=[],sampling=False)
    n.snapshot=lambda:dict(q);n._lock=lambda reason:None;n._is_feedback_fresh=lambda:True
    n._event=lambda *args:None
    n.sent=[];n.apply=lambda group,targets:n.sent.append((group,targets))
    monkeypatch.setattr(module.rclpy,'spin_once',lambda *a,**kw:None)
    w=Window(n,cfg,FakeCamera(cfg));w.ros_timer.stop();w.tick_timer.stop()
    w.set_session(Session(tmp_path/'round',cfg,q,w.fk.sha))
    yield w
    w.close()

def batch(w):
    now=time.monotonic();q=w.node._current
    w.node.history=[(t,dict(q)) for t in np.linspace(now-.9,now,60)]
    w.node._feedback_monotonic=now
    f=w.camera.frame;f['received']=now
    w.start_capture()
    assert w.collect
    for i in range(8):
        f=copy.deepcopy(f);f['frame_id']+=i+1;f['received']=time.monotonic()
        w.capture_tick(f)

def test_capture_advances_and_duplicate_does_not(win):
    batch(win)
    assert win.index==1 and len(win.session.data['samples'])==1
    batch(win)
    assert win.index==1 and len(win.session.data['samples'])==1
    assert '相似' in win.banner.text()

def test_replacement_keeps_old_photo_and_auto_next(win):
    batch(win);old=win.session.data['samples']['0']['image']
    win.select_sample(0);batch(win)
    assert win.index==1
    assert (win.session.path/old).exists()
    assert len(win.session.data['revisions'])==1

def test_control_phase_gates(win):
    assert win.motion_gate('neck')
    assert win.motion_gate('grippers')
    assert win.motion_gate('leg_waist')
    assert win.motion_gate('left_arm')
    assert not win.motion_gate('right_arm')
    win.index=32
    assert win.motion_gate('neck')
    win.session.data['fixed_solution']={'x':1}
    assert not win.motion_gate('neck')
    assert win.motion_gate('grippers')

def test_jaw_moved_sample_rejected(win):
    win.node._current['Joint_Right_Gripper']=3
    now=time.monotonic()
    win.node.history=[(t,dict(win.node._current)) for t in np.linspace(now-.9,now,60)]
    win.start_capture()
    assert win.collect is None and not win.session.data['samples']
    assert '固定姿态' in win.banner.text()

def test_stale_image_rejected(win):
    win.camera.frame['received']=time.monotonic()-5
    win.start_capture()
    assert not win.collect and '过期' in win.banner.text()

def test_candidate_invalidated_after_replacement(win):
    batch(win)
    p=win.session.path/'calibration_candidate.json';p.write_text('{"valid":true}')
    win.select_sample(0);batch(win)
    assert 'false' in p.read_text()

def test_live_drag_sends_latest_without_apply_click(win):
    win.node._hardware_ready=True;win.current=dict(win.node._current)
    win.realtime.setChecked(True)
    name=ARMS['right'][0]
    win.rows[name][2].setValue(2.);win.rows[name][2].setValue(4.)
    win.live_tick()
    assert len(win.node.sent)==1
    assert win.node.sent[0][1][name]==4.

def test_preview_and_sync_do_not_send(win):
    name=ARMS['right'][0];win.rows[name][2].setValue(2.);win.live_tick()
    assert not win.node.sent
    win.node._hardware_ready=True;win.current=dict(win.node._current)
    win.realtime.setChecked(True);win.sync_group('right_arm');win.live_tick()
    assert not win.node.sent

def test_no_live_before_ready_or_during_capture(win):
    win.realtime.setChecked(True);assert not win.realtime.isChecked()
    win.node._hardware_ready=True;win.collect={'frames':[]}
    win.realtime.setChecked(True);assert not win.realtime.isChecked()
    win.collect=None

def test_model_routes_to_correct_group_and_blocks_waist(win):
    name=ARMS['right'][3];win.model_selected(name)
    assert win.tabs.currentIndex()==win.group_indexes['right_arm']
    win.model_dragged(name,-4.);assert win.rows[name][2].value()==-4.
    win.model_dragged('Joint_Waist_Pitch',30.);assert not win.node.sent

def test_reset_is_blocked_in_active_round(win):
    win.node._hardware_ready=True;win.reset_pose()
    assert not win.node.sent and '结束本轮' in win.banner.text()

def test_model_gripper_schematic_tracks_feedback(win):
    a=win.model.display_joints({'Joint_Right_Gripper':10})
    b=win.model.display_joints({'Joint_Right_Gripper':360})
    assert b['Joint_Right_Gripper']<a['Joint_Right_Gripper']
    assert b['Joint_Right_Gripper_Right_Finger']!=a['Joint_Right_Gripper_Right_Finger']

def test_start_has_visible_prerequisite_error(win,monkeypatch):
    win.session=None;win.board_check.setChecked(False);messages=[]
    monkeypatch.setattr(Q.QMessageBox,'information',lambda *a:messages.append(a[2]))
    win.new_session()
    assert not win.session and messages and win.pages.currentIndex()==0
    assert '尚未开始' in win.start_status.text()

def test_start_auto_id_not_blocked_by_first_pose_quality(win,monkeypatch,tmp_path):
    win.session=None;win.cfg['marker_id']=-1;win.marker.setValue(-1)
    win.camera.frame['detection'].update(ok=False,reason='平面姿态有歧义')
    now=time.monotonic();win.camera.frame['received']=now
    win.node.history=[(t,dict(win.node._current)) for t in np.linspace(now-.9,now,60)]
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    errors=[];monkeypatch.setattr(Q.QMessageBox,'information',lambda *a:errors.append(a[2]))
    win.new_session()
    assert not errors and win.session and win.session.data['config']['marker_id']==0
    assert win.index==0 and '1 / 38' in win.guide.text()
    # Recording still rejects this poor-quality image.
    win.start_capture();assert win.collect is None

def test_gripper_full_range_buttons_targets(win):
    win.session=None;win.node._hardware_ready=True;win.current=dict(win.node._current)
    win.realtime.setChecked(True)
    name='Joint_Right_Gripper'
    win.rows[name][2].setValue(360);win.live_tick()
    assert win.node.sent[-1][1][name]==360
    win.rows[name][2].setValue(10);win.last_live_send=0;win.live_tick()
    assert win.node.sent[-1][1][name]==10

@pytest.mark.parametrize('side',['Left','Right'])
def test_gripper_model_full_closed_to_full_open_clearance(win,side):
    from control_center_joint_control.robot_view import _load_binary_stl
    model=win.model.model
    def clouds(q):
        tf=model.link_transforms(q);out=[]
        for finger in ('Left','Right'):
            link=f'Link_{side}_Gripper_{finger}_Finger'
            v=next(v for v in model.visuals if v.link==link)
            vertices,_=_load_binary_stl(v.mesh_path);t=tf[link]@v.origin
            out.append(vertices@t[:3,:3].T+t[:3,3])
        return out
    neutral=clouds({});axis=neutral[1].mean(axis=0)-neutral[0].mean(axis=0);axis/=np.linalg.norm(axis)
    def gap(motor):
        c=clouds(win.model.display_joints({f'Joint_{side}_Gripper':motor}))
        return np.min(c[1]@axis)-np.max(c[0]@axis)
    assert gap(360)==pytest.approx(0,abs=.0001)
    assert gap(10)==pytest.approx(.095,abs=.0001)
    assert gap(360)<gap(185)<gap(10)
