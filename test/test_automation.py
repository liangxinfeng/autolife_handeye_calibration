"""No hardware or ROS publishers: state-machine and numerical regressions."""
import copy
import json
import time
from pathlib import Path
from PyQt5 import QtWidgets as Q
from types import SimpleNamespace as NS
import pytest
from test_core import dataset
from test_ui import win,batch
from autolife_handeye_calibration_306.core import compute_report,ARMS,has_solution
from autolife_handeye_calibration_306.automation import edge_error


def test_twenty_samples_independent_validation():
    data,_,_=dataset();data['config']=copy.deepcopy(data['config'])
    data['config'].update(minimum_training_samples=12,minimum_validation_samples=4)
    keys=list(range(12))+list(range(24,28))+list(range(32,36))
    data['samples']={str(i):s for i,k in enumerate(keys) for s in [copy.deepcopy(data['samples'][str(k)])]}
    for i,s in enumerate(data['samples'].values()):s['slot']=i
    report=compute_report(data)
    assert report['fixed_head_pass'] and report['moving_head_pass']
    data['samples']['13']['G'][0][3]+=.08
    assert not compute_report(data)['fixed_head_pass']


def test_segment_guard_bounds_and_nonmoving_joints():
    e={'joint':'a','start_q':{'a':0,'b':2},'end_q':{'a':10,'b':2}}
    assert not edge_error({'a':5,'b':2},e,moving=True)
    assert edge_error({'a':11,'b':2},e,moving=True)
    assert edge_error({'a':5,'b':3},e,moving=True)
    assert edge_error({'a':5,'b':2},e)
    assert not edge_error({'a':9.8,'b':2.1},e)
    assert edge_error({'a':float('nan'),'b':2.},e,moving=True)


def setup_run(w):
    a=w.automation;q=dict(w.node._current);joint=ARMS['right'][0]
    end={**q,joint:3.}
    edge={'joint':joint,'target':3.,'start_q':q,'end_q':end}
    a.cfg=w.cfg;a.cfg['auto_visual_horizon_deg']=2.;a.source=q;a.plan={'targets':[{'q':end,'edges':[edge]}]}
    a.target_index=0;a.edge_index=0;a.state='next_edge'
    w.node._hardware_ready=True
    def lock(reason):w.node._active_goals.clear();w.node._hardware_ready=False
    w.node._lock=lock
    def apply(group,targets,speed_scale=None):
        w.node.sent.append((group,targets));w.node._active_goals[group]={'targets':targets}
    w.node.apply=apply
    return a,edge


def stable_endpoint(w):
    now=time.monotonic()
    w.node.history=[(now-d,dict(w.node._current)) for d in (.22,.17,.12,.07,.02)]


def test_auto_edge_only_one_joint_and_waits_for_measured_arrival(win):
    a,e=setup_run(win);a.tick(win.camera.frame)
    assert a.state=='moving' and len(win.node.sent)==1
    assert sum(abs(v-e['start_q'][n])>.01 for n,v in win.node.sent[0][1].items())==1
    win.node._current[e['joint']]=1.5;a.tick(win.camera.frame)
    assert a.state=='moving'
    win.node._current[e['joint']]=3.;win.node._active_goals.clear();stable_endpoint(win);a.tick(win.camera.frame)
    assert a.state=='next_edge' and a.edge_index==1
    a.tick(win.camera.frame);assert a.state=='reset_settling'


def test_camera_loss_stops_without_auto_resume(win):
    a,_=setup_run(win);a.tick(win.camera.frame)
    win.camera.frame['received']=time.monotonic()-1;a.tick(win.camera.frame)
    assert not a.active and not win.node._active_goals
    win.camera.frame['received']=time.monotonic();a.tick(win.camera.frame)
    assert len(win.node.sent)==1


def test_rejected_command_cancels(win):
    a,_=setup_run(win);win.node.apply=lambda *args,**kwargs:None
    a.tick(win.camera.frame)
    assert not a.active and '拒绝' in win.banner.text()


def test_stop_cannot_restart_from_delayed_plan_or_solver(win):
    a,_=setup_run(win);win.stop()
    a.planned(0,0)
    assert not a.active and not win.node.sent


def test_quality_retries_are_bounded(win):
    a,_=setup_run(win);a.state='capturing';a.capture_tries=2
    a.capture_failed('blur');assert a.state=='settling'
    a.capture_tries=3;a.capture_failed('blur')
    assert not a.active and not win.node.sent


def test_manual_changes_blocked_during_automation(win):
    a,_=setup_run(win)
    win.apply_group('right_arm');win.target_changed(ARMS['right'][0],5)
    assert not win.node.sent and not win.pending_groups


def test_invalid_solution_still_stops(win):
    a,_=setup_run(win);a.state='solving';win.busy=False
    a.solved({'fixed_head_pass':False},None)
    assert not a.active and win.pages.currentIndex()==2


@pytest.mark.parametrize('head_count',[4,8,12])
def test_actual_sixteen_quality_warning_continues_to_plan_end_and_exports(win,head_count):
    # Actual 306 round stopped at 16 due to 10.77px > 3px recommendation.
    # Head samples below are synthetic holdout data, never a real test result.
    data=json.loads((Path(__file__).parent/'actual_sixteen_session.json').read_text())
    data['steps']=data['steps'][:16]+[{'phase':'head','hint':'synthetic test holdout'} for _ in range(head_count)]
    data['config']['auto_head_validation_samples']=head_count
    total=len(data['steps'])
    report=compute_report(data)
    assert has_solution(report) and not report['fixed_head_pass']
    assert 10<report['validation']['pixel_p95']<12
    win.session.data=copy.deepcopy(data);win.cfg=data['config']
    a,_=setup_run(win);a.state='solving';a.target_index=16
    a.plan={'targets':copy.deepcopy(data['steps'])}
    win.solved(report,None)
    assert a.state=='next_edge' and win.session.data['fixed_solution']
    assert '不中断' in win.banner.text() and not win.node.sent
    # No premature calculation/stop at the former twentieth-image boundary.
    calls=[];win.solve=lambda:calls.append(a.target_index)
    for i in range(16,total):
        sample=copy.deepcopy(data['samples'][str((i-16)%12)])
        sample.update(slot=i,phase='head')
        win.session.data['samples'][str(i)]=sample
        a.captured()
        assert a.state==('solving' if i==total-1 else 'next_edge')
    assert calls==[total]
    final=compute_report(win.session.data);win.solved(final,None)
    assert not a.active and win.pages.currentIndex()==2 and not win.node.sent
    assert f'{total}张采集及计算已完成' in win.banner.text()
    saved=json.loads((win.session.path/'calibration_candidate.json').read_text())
    assert saved['report']['result_available']
    assert saved['report']['quality_grade']=='needs_validation'
    assert not saved['report']['fixed_head_pass'] and not saved['applied_to_robot']
    assert saved['report']['head']['count']==head_count


@pytest.mark.parametrize('corruption',['nan','reflection','shape'])
def test_nonrigid_or_nonfinite_solution_cannot_continue(win,corruption):
    data,_,_=dataset();report=compute_report(data)
    if corruption=='nan':report['T_mount_camera'][0][3]=float('nan')
    elif corruption=='reflection':report['T_mount_camera'][0]=[-v for v in report['T_mount_camera'][0]]
    else:report['T_mount_camera']=[]
    assert not has_solution(report)
    a,_=setup_run(win);a.state='solving';a.target_index=16
    a.solved(report,None)
    assert not a.active and not win.node.sent


@pytest.mark.parametrize('arm_count,head_count',[(16,8),(14,6),(16,0)])
def test_phase_and_completion_boundaries_come_from_plan(win,arm_count,head_count):
    data,_,_=dataset();report=compute_report(data)
    a,_=setup_run(win);a.target_index=0;a.state='capturing'
    a.plan={'targets':[{'phase':'train'} for _ in range(arm_count)]+
                       [{'phase':'head'} for _ in range(head_count)]}
    calls=[];win.solve=lambda:calls.append(a.target_index)
    total=arm_count+head_count
    for number in range(1,total+1):
        a.captured()
        if a.state=='solving':a.solved(report,None)
        if number<total:assert a.active and a.state=='next_edge'
    assert calls==([arm_count,total] if head_count else [total])
    assert not a.active and not win.node.sent


def test_stopped_automation_cannot_resume_from_late_solution(win):
    data,_,_=dataset();report=compute_report(data)
    a,_=setup_run(win);a.state='solving';a.target_index=16
    win.stop();a.solved(report,None)
    assert not a.active and not win.node.sent


def test_preflight_creates_twenty_step_session_without_enable(win,monkeypatch,tmp_path):
    w=win;a=w.automation;w.session=None;w.node._hardware_ready=True
    a.state='planning';a.source=dict(w.node._current);a.cfg=copy.deepcopy(w.cfg)
    a.request_mode='collect'
    a.observation={'q':a.source,'C':w.camera.frame['detection']['C'],
                   'K':w.camera.frame['K'].tolist(),'D':w.camera.frame['D'].tolist(),'size':[640,480]}
    a.cfg.update(minimum_training_samples=12,minimum_validation_samples=4)
    targets=[{'phase':'train' if i<12 else 'validation' if i<16 else 'head',
              'hint':'test','q':a.source,'edges':[]} for i in range(20)]
    result=json.dumps({'ok':True,'plan':{'targets':targets}}).encode()
    a.output=b''
    a.process=NS(readAllStandardOutput=lambda:result,readAllStandardError=lambda:b'',deleteLater=lambda:None)
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    a.planned(0,0)
    assert a.state=='next_edge' and len(w.session.data['steps'])==20
    assert w.session.data['config']['minimum_training_samples']==12
    assert (w.session.path/'auto_plan.json').exists()
    assert not w.node.sent


def test_source_drift_invalidates_preflight(win):
    a=win.automation;a.state='planning';a.source=dict(win.node._current);a.output=b''
    a.process=NS(readAllStandardOutput=lambda:b'{"ok":true}',readAllStandardError=lambda:b'',deleteLater=lambda:None)
    win.node._current[ARMS['right'][0]]=2.
    a.planned(0,0)
    assert not a.active and not win.node.sent and '姿态发生变化' in win.banner.text()


def test_collect_commands_have_visual_horizon_and_fresh_frame_gate(win):
    a,e=setup_run(win);a.request_mode='collect';a.cfg['marker_id']=0
    a.vision=NS(anchor=lambda *args:None,check=lambda *args:True)
    f=win.camera.frame;win.node.history=[(f['received'],dict(win.node._current))]
    a.tick(f)
    assert a.state=='moving' and win.node.sent[-1][1][e['joint']]==2.
    a.tick(f);assert len(win.node.sent)==1
    win.node._current[e['joint']]=1.5;f['received']=time.monotonic()
    win.node.history.append((f['received'],dict(win.node._current)));a.tick(f)
    assert win.node.sent[-1][1][e['joint']]==3.


def test_visual_boundary_blocks_before_motion(win):
    a,e=setup_run(win);a.request_mode='collect';a.cfg['marker_id']=0
    a.vision=NS(anchor=lambda *args:None,check=lambda *args:False)
    win.node.history=[(win.camera.frame['received'],dict(win.node._current))]
    a.tick(win.camera.frame)
    assert not a.active and not win.node.sent and '提前停止' in win.banner.text()


def test_reset_works_without_marker_but_does_not_capture_blindly(win):
    a,e=setup_run(win);a.request_mode='reset'
    win.camera.frame['detection'].update(ok=False,reason='no marker')
    a.tick(win.camera.frame)
    assert a.state=='moving' and win.node.sent
    win.node._current[e['joint']]=3.;win.node._active_goals.clear();stable_endpoint(win);a.tick(win.camera.frame);a.tick(win.camera.frame)
    assert a.state=='reset_settling'
    before=len(win.session.data['samples']);a.tick(win.camera.frame)
    assert a.state=='reset_settling' and len(win.session.data['samples'])==before


def test_auto_start_requests_checked_reset_from_nonhome_pose(win,monkeypatch):
    win.session=None;win.node._hardware_ready=True
    win.node._current['Joint_Neck_Pitch']=-3
    win.node._current['Joint_Right_Gripper']=180
    win.camera.frame['received']=time.monotonic()
    win.camera.frame['detection'].update(ok=False,reason='offscreen')
    requests=[]
    monkeypatch.setattr(Q.QMessageBox,'question',lambda *args,**kwargs:Q.QMessageBox.Yes)
    win.automation.request_plan=lambda q,mode:requests.append((q,mode))
    win.automation.start()
    assert requests and requests[0][1]=='reset' and not win.node.sent


def test_transient_marker_blur_does_not_send_blind_target(win):
    a,e=setup_run(win);a.request_mode='collect';a.cfg['marker_id']=0
    win.camera.frame['detection'].update(ok=False,reason='blur')
    a.tick(win.camera.frame)
    assert a.active and not win.node.sent
    a.vision_bad_since=time.monotonic()-1;a.tick(win.camera.frame)
    assert not a.active and not win.node.sent


@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('remaining',[.301,.332,.49])
def test_small_final_target_is_not_filtered(win,direction,remaining):
    a,e=setup_run(win);a.request_mode='collect';a.cfg['marker_id']=0
    e['target']=direction*7.98901382825419;e['end_q'][e['joint']]=e['target']
    q=win.node._current;q[e['joint']]=e['target']-direction*remaining
    a.state='moving';a.edge=e;a.issued=q[e['joint']]
    a.deadline=time.monotonic()+60
    a.vision=NS(anchor=lambda *args:None,check=lambda *args:True)
    f=win.camera.frame;f['received']=time.monotonic();win.node.history=[(f['received'],dict(q))]
    a.tick(f)
    assert len(win.node.sent)==1
    assert win.node.sent[-1][1][e['joint']]==e['target']
    # Final goal is submitted once, not repeated on every fresh frame.
    f['received']=time.monotonic();a.tick(f);assert len(win.node.sent)==1
    q[e['joint']]=e['target'];win.node._active_goals.clear();stable_endpoint(win);a.tick(f)
    assert a.state=='next_edge' and a.edge_index==1
    a.tick(f);assert a.state=='settling'
    batch(win)
    assert len(win.session.data['samples'])==1 and a.target_index==1


def test_small_final_target_still_requires_good_visibility(win):
    a,e=setup_run(win);a.request_mode='collect';a.cfg['marker_id']=0
    q=win.node._current;q[e['joint']]=2.668
    a.state='moving';a.edge=e;a.issued=2.668;a.deadline=time.monotonic()+60
    a.vision=NS(anchor=lambda *args:None,check=lambda *args:False)
    f=win.camera.frame;win.node.history=[(f['received'],dict(q))]
    a.tick(f)
    assert not a.active and not win.node.sent


def test_arrival_uses_same_tolerance_as_bridge(win):
    a,e=setup_run(win);a.state='moving';a.edge=e;a.deadline=time.monotonic()+60
    win.node.get_parameter=lambda name:NS(value=.4 if name=='goal_tolerance_deg' else 1.)
    win.node._current[e['joint']]=e['target']-.35
    win.node._active_goals.clear();stable_endpoint(win);a.tick(win.camera.frame)
    assert a.state=='next_edge'


def test_endpoint_crossing_and_stale_feedback_cannot_handoff(win):
    a,e=setup_run(win);a.tick(win.camera.frame)
    q=win.node._current;q[e['joint']]=3.;win.node._active_goals.clear()
    now=time.monotonic()
    win.node.history=[(now-.22,{**q,e['joint']:2.45}),(now-.1,{**q,e['joint']:2.8}),(now,dict(q))]
    a.tick(win.camera.frame)
    assert a.state=='moving' and a.edge_index==0
    stable_endpoint(win);win.node.history=[(t-1,p) for t,p in win.node.history]
    a.tick(win.camera.frame);assert a.state=='moving'
    # Even many repeats of one packet cannot fake a stable time span.
    win.node.history=[(now,dict(q))]*20
    a.tick(win.camera.frame);assert a.state=='moving'
    stable_endpoint(win);a.tick(win.camera.frame)
    assert a.state=='next_edge'


def test_nonmoving_joint_target_does_not_follow_feedback_drift(win):
    a,e=setup_run(win);a.edge=e
    held=ARMS['right'][4]
    for drift in (.2,.4,.6):
        win.node._current[held]=drift
        a.send_motion(dict(win.node._current),win.camera.frame,True)
        assert win.node.sent[-1][1][held]==e['start_q'][held]
    assert edge_error({**e['start_q'],held:.71},e,moving=True)


def test_faster_horizon_still_checks_intermediate_visibility(win):
    a,e=setup_run(win);a.request_mode='collect';a.cfg.update(marker_id=0,auto_visual_horizon_deg=3.)
    checked=[]
    def check(q,margin):
        checked.append(q[e['joint']]);return not .9<q[e['joint']]<1.1
    a.vision=NS(anchor=lambda *args:None,check=check)
    f=win.camera.frame;win.node.history=[(f['received'],dict(win.node._current))]
    a.tick(f)
    assert not a.active and not win.node.sent and 1. in checked


def test_stop_records_exact_edge_and_feedback(win):
    a,e=setup_run(win);a.tick(win.camera.frame)
    win.node._current[ARMS['right'][4]]=1.
    stable_endpoint(win);a.tick(win.camera.frame)
    saved=json.loads((win.session.path/'auto_stop.json').read_text())
    assert saved['sample']==1 and saved['edge']==e and saved['history']
    assert '偏差+1.000°' in saved['reason']


def pairing_run(w,monkeypatch):
    a,e=setup_run(w);a.request_mode='collect';a.cfg['marker_id']=0
    a.vision=NS(anchor=lambda *args:None,check=lambda *args:True)
    clock=[time.monotonic()]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    f=w.camera.frame;f['received']=clock[0]-.2
    w.node.history=[(clock[0],dict(w.node._current))]
    events=[];w.node._event=lambda *args:events.append(args)
    return a,e,f,clock,events


def test_pairing_transient_wait_requires_new_frame_then_continues(win,monkeypatch):
    a,e,f,clock,events=pairing_run(win,monkeypatch)
    a.tick(f)
    assert a.active and a.pair_bad_since is not None and not win.node.sent
    assert '等待新图像' in win.banner.text()
    assert len(events)==1 and '配对差=200.0ms' in events[0][0]
    # Even late feedback matching the rejected frame cannot revive that frame.
    win.node.history.append((f['received'],dict(win.node._current)))
    clock[0]+=.1;a.tick(f)
    assert a.pair_bad_since is not None and not win.node.sent and len(events)==1
    f['received']=clock[0];win.node.history.append((clock[0],dict(win.node._current)))
    a.tick(f)
    assert a.active and a.pair_bad_since is None and a.target_index==0
    assert win.node.sent[-1][1][e['joint']]==2.
    assert len(events)==2 and '配对恢复' in events[-1][0]
    a.tick(f);assert len(win.node.sent)==1


def test_pairing_continuous_fresh_but_unmatched_frames_time_out(win,monkeypatch):
    a,e,f,clock,events=pairing_run(win,monkeypatch);a.tick(f)
    for _ in range(5):
        clock[0]+=.21;f['received']=clock[0]-.2
        win.node.history=[(clock[0],dict(win.node._current))]
        a.tick(f)
    assert not a.active and not win.node.sent
    assert '持续1秒' in win.banner.text() and '配对差=200.0ms' in win.banner.text()
    assert len([x for x in events if '暂缓新目标' in x[0]])==1
    f['received']=clock[0];a.tick(f)
    assert not a.active and not win.node.sent


def test_pairing_wait_cannot_advance_arrival_or_survive_user_stop(win,monkeypatch):
    a,e,f,clock,events=pairing_run(win,monkeypatch);a.tick(f)
    win.node._current[e['joint']]=e['target'];a.tick(f)
    assert a.state=='moving' and a.edge_index==0 and not win.collect
    win.stop()
    f['received']=clock[0];win.node.history=[(clock[0],dict(win.node._current))];a.tick(f)
    assert not a.active and a.pair_bad_since is None and not win.node.sent


def test_pairing_recovery_rechecks_visibility(win,monkeypatch):
    a,e,f,clock,events=pairing_run(win,monkeypatch);a.tick(f)
    clock[0]+=.1;f['received']=clock[0];win.node.history=[(clock[0],dict(win.node._current))]
    a.vision.check=lambda *args:False;a.tick(f)
    assert not a.active and not win.node.sent and '提前停止' in win.banner.text()


def test_pairing_empty_history_waits_and_camera_loss_still_stops(win,monkeypatch):
    a,e,f,clock,events=pairing_run(win,monkeypatch);win.node.history=[];a.tick(f)
    assert a.active and not win.node.sent and '反馈历史为空' in events[0][0]
    clock[0]+=.4;a.tick(f)
    assert not a.active and 'RGB断流' in win.banner.text()


def test_pairing_pause_does_not_replace_already_issued_subgoal(win,monkeypatch):
    a,e,f,clock,events=pairing_run(win,monkeypatch)
    f['received']=clock[0];a.tick(f);assert len(win.node.sent)==1
    previous=copy.deepcopy(win.node._active_goals)
    clock[0]+=.4;f['received']=clock[0]-.2
    win.node.history=[(clock[0],dict(win.node._current))];a.tick(f)
    assert a.pair_bad_since is not None and len(win.node.sent)==1
    assert win.node._active_goals==previous
