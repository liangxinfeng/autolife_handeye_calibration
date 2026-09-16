"""Offline regression using the user's failed two-frame session, no ROS nodes."""
import json
from pathlib import Path
import sys
import time
import yaml
import pyrealsense2 as rs
import numpy as np
from autolife_handeye_calibration_306.auto_plan_worker import plan
from autolife_handeye_calibration_306.auto_plan_worker import CollisionGuard
import hppfcl as fcl
from autolife_handeye_calibration_306.visibility import Visibility
from autolife_handeye_calibration_306.core import compute_report

root=Path(__file__).resolve().parents[1]
s=json.loads(Path(sys.argv[1]).read_text())
cfg=yaml.safe_load((root/'config/experiment.yaml').read_text())
first,second=list(s['samples'].values())[:2]
context=rs.context();devices=list(context.query_devices());sensors=[sensor for d in devices for sensor in d.query_sensors()]
profiles=[p for sensor in sensors
          for p in sensor.get_stream_profiles() if p.stream_type()==rs.stream.color
          and p.as_video_stream_profile().width()==1280 and p.as_video_stream_profile().height()==720]
intr=profiles[0].as_video_stream_profile().get_intrinsics()
K=[[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]]
D=list(intr.coeffs)
obs={'q':first['q_deg'],'C':first['C'],'K':K,'D':D,'size':[1280,720]}
v=Visibility(cfg,obs)
pred=v.corners(second['q_deg'])
print(json.dumps({'native_intrinsics':K,'native_size':obs['size']},ensure_ascii=False),flush=True)
t=time.monotonic();p=plan(first['q_deg'],cfg,obs)
print(json.dumps({'states':p['checked_states'],'seconds':time.monotonic()-t,
                  'exact_distance_calls':p['exact_distance_calls'],
                  'scales':[x['view_scale'] for x in p['targets']],
                  'bounds':[[v.corners(x['q'],True).min(axis=0).tolist(),v.corners(x['q'],True).max(axis=0).tolist()] for x in p['targets']]},ensure_ascii=False),flush=True)
assert len(p['targets'])==24 and all(v.check(x['q']) for x in p['targets'])
print(json.dumps({'joint_spans_deg':{n:float(np.ptp([t['q'][n] for t in p['targets']]))
  for n in ('Joint_Left_Forearm','Joint_Left_Wrist_Lower','Joint_Neck_Pitch','Joint_Neck_Yaw')}},ensure_ascii=False),flush=True)
cfg.update(minimum_training_samples=12,minimum_validation_samples=4)
samples={}
for i,t in enumerate(p['targets']):
    q=t['q'];G=v.wrist(q);H=v.fk.fk('Link_Waist_Yaw_to_Shoulder_Inner','Link_Camera_Head_Forehead',q)
    C=np.linalg.inv(v.camera(q))@G@v.Y
    samples[str(i)]={'slot':i,'phase':t['phase'],'G':G.tolist(),'H':H.tolist(),'C':C.tolist(),
                     'corners':v.corners(q).tolist(),'K':K,'D':D}
report=compute_report({'config':cfg,'samples':samples})
assert report['fixed_head_pass'] and report['moving_head_pass']
print(json.dumps({'rotation_singular_values':report['rotation_singular_values'],
                  'ideal_data_validation_pass':report['fixed_head_pass']},ensure_ascii=False),flush=True)
reset_source=dict(second['q_deg']);reset_source['Joint_Left_UpperArm']=-18
reset_source['Joint_Neck_Pitch']=-12
t=time.monotonic();r=plan(reset_source,cfg,mode='reset')
print(json.dumps({'reset_seconds':time.monotonic()-t,'reset_states':r['checked_states'],
                  'reset_joints':[e['joint'] for e in r['targets'][0]['edges']]}),flush=True)
assert r['kind']=='reset' and r['targets'][0]['q']['Joint_Left_UpperArm']==0
# Verify AABB broad phase against the former exact distance test on real meshes.
guard=CollisionGuard(cfg);checked=0
for target in p['targets'][::3]:
    guard.check(target['q']);wrist=guard.kin.data.oMf[guard.frame]
    center=wrist.translation+wrist.rotation@guard.offset
    for i,obj in enumerate(guard.kin.geom_model.geometryObjects):
        if any(obj.name.startswith(x) for x in guard.skip):continue
        t=guard.kin.geom_data.oMg[i];low,high=guard.boxes[i]
        for c in (center,t.translation+t.rotation@((low+high)/2)):
            local=t.rotation.T@(c-t.translation)
            separated=np.linalg.norm(local-np.clip(local,low,high))>guard.radius+.005
            distance=fcl.distance(guard.sphere,fcl.Transform3f(np.eye(3),c),obj.geometry,
                                  fcl.Transform3f(t.rotation,t.translation),fcl.DistanceRequest(),fcl.DistanceResult())
            assert not separated or distance>=.005-1e-8
            checked+=1
print(json.dumps({'broadphase_equivalence_cases':checked}),flush=True)
(root/'test/view_plan_result.json').write_text(json.dumps(p,ensure_ascii=False,indent=2))
