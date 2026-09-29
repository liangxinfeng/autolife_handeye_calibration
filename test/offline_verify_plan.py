"""Plan a synthetic nearby goal from stored feedback; NEVER creates ROS nodes."""
import json
from pathlib import Path
import numpy as np
from autolife_handeye_calibration_307.core import Kinematics
from autolife_handeye_calibration_307.verification_core import load_geometry,tcp_position
from autolife_handeye_calibration_307.verify_plan_worker import plan

p=Path('/home/ubuntu/ros2_ws/src/autolife_handeye_calibration_307/sessions/20260914_102651_586775/session.json')
d=json.loads(p.read_text());cfg=d['config'];fk=Kinematics(cfg['urdf_path']);tcp=load_geometry(d,fk)
q=d['reference'];start=tcp_position(fk,cfg,q,tcp)
request={'config':cfg,'q':q,'tcp':tcp.tolist(),'goal':(start+np.array([.02,0,0])).tolist(),'urdf_sha256':fk.sha}
r=plan(request)
assert r['nominal_error_mm']<5
assert r['execution']=='continuous_joint_path_v1'
assert all(p['Joint_Left_Gripper']==q['Joint_Left_Gripper'] and p['Joint_Neck_Pitch']==q['Joint_Neck_Pitch'] for p in r['waypoints'])
print(json.dumps({k:r[k] for k in ('distance_mm','nominal_error_mm','checked_states','planning_seconds')},ensure_ascii=False))
print('tcp_in_wrist_m',tcp.tolist(),'waypoint_count',len(r['waypoints']),'path_max_speed',r['path_max_speed'])
request['goal']=(start+np.array([1.,0,0])).tolist()
try:plan(request)
except ValueError as exc:assert '30厘米' in str(exc)
else:raise AssertionError('unbounded goal accepted')

# Replan the user's stored failed run, still offline, no hardware publishers.
saved=json.loads((p.parent/'verification/20260914_200149_810042/plan.json').read_text())['plan']
request.update(q=saved['source_q'],tcp=saved['tcp'],goal=saved['goal'])
r=plan(request)
print('USER_TARGET',json.dumps({k:r[k] for k in ('distance_mm','nominal_error_mm','checked_states','planning_seconds','path_max_speed')}))
