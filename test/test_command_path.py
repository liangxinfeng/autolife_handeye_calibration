"""Exercise the inherited real command parser/progressor with fake publishers.
No ROS init, service calls, or robot network access.
"""
import json
import time
from types import SimpleNamespace as NS
import pytest
from autolife_handeye_calibration_306.bridge import CalibrationBridge
from control_center_joint_control.joint_definitions import GROUPS

@pytest.fixture
def bridge():
    b=CalibrationBridge.__new__(CalibrationBridge)
    b.gui_call=False;b.sampling=False;b.conflict='';b.progress_watch={};b.last_conflict_check=time.monotonic()
    b.manual_speed_scale=1.0
    b.gate=lambda group:'';b._action_state=None;b._hardware_ready=True;b._hardware_enabled=True
    b._is_feedback_fresh=lambda:True;b._communication_lost_joints=lambda names:[]
    b._event=lambda text,*args:setattr(b,'_last_event',text)
    b._current={n:0. for g in GROUPS.values() for n,*_ in g['joints']}
    b._current.update(Joint_Left_Gripper=10.,Joint_Right_Gripper=10.)
    b._controller_target=dict(b._current);b._preview=dict(b._current);b._active_goals={};b._goal_sent_groups=set()
    b._whole_body_pub=NS(get_subscription_count=lambda:1);b._gripper_pub=NS(get_subscription_count=lambda:1)
    b.sent=[];b._publish_vendor=lambda pub,payload:b.sent.append((pub,payload))
    params={'goal_tolerance_deg':.3,'gripper_goal_tolerance_deg':1.,'max_single_step_deg':5.,'max_gripper_command_lead_deg':30.}
    b.get_parameter=lambda name:NS(value=params[name])
    return b

@pytest.mark.parametrize('group',['right_arm','left_arm','neck','grippers'])
def test_ui_apply_reaches_vendor_publisher_bounded(bridge,group):
    targets={n:bridge._current[n] for n,*_ in GROUPS[group]['joints']}
    name=next(iter(targets));targets[name]+=40
    bridge.apply(group,targets);assert group in bridge._active_goals
    bridge._advance_motion();assert bridge.sent
    record=bridge._active_goals[group]
    assert record['commands'][name]-bridge._current[name]==pytest.approx(30. if group=='grippers' else 5.)

def test_reduced_speed_and_reset_override(bridge):
    bridge.manual_speed_scale=.3
    targets={n:30. for n,*_ in GROUPS['neck']['joints']}
    bridge.apply('neck',targets);assert bridge._active_goals['neck']['speed_scale']==.3
    bridge.apply('neck',targets,speed_scale=.5);assert bridge._active_goals['neck']['speed_scale']==.5

def test_external_apply_cannot_move(bridge):
    bridge._on_apply(NS(data=json.dumps({'group':'neck'})))
    assert not bridge.sent and not bridge._active_goals

def test_missing_subscriber_fails_closed(bridge):
    bridge._whole_body_pub.get_subscription_count=lambda:0
    bridge.apply('neck',{n:0 for n,*_ in GROUPS['neck']['joints']})
    assert not bridge._hardware_ready and not bridge.sent

def test_stalled_feedback_cancels_target(bridge):
    bridge.apply('neck',{n:4 for n,*_ in GROUPS['neck']['joints']})
    bridge.progress_watch['neck']=(time.monotonic()-6,dict(bridge._current))
    bridge._advance_motion()
    assert not bridge._active_goals and not bridge._hardware_ready and not bridge.sent
