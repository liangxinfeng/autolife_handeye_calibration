"""Read-only live SHM check; no ROS node, enable or movement."""
import json
import time
from pathlib import Path
import yaml
from autolife_handeye_calibration_307.camera import Camera
from autolife_handeye_calibration_307.depth_target import surface_point

cfg=yaml.safe_load(Path('/tmp/handeye_depth_stage/config/experiment.yaml').read_text())
cfg.update(read_depth=True,depth_scale_m=.001)
camera=Camera(cfg);seen=set();paired=0;errors={};example=None
try:
    end=time.monotonic()+8
    while time.monotonic()<end:
        frame,error=camera.get()
        if frame and frame['frame_id'] not in seen:
            seen.add(frame['frame_id'])
            if frame.get('depth') is not None:
                paired+=1
                try:
                    for pixel in [(320,240),(200,300),(400,300),(200,180),(400,180)]:
                        try:point,stats=surface_point(frame,pixel,time.monotonic());break
                        except ValueError:
                            if pixel==(400,180):raise
                    example={'pixel':pixel,'point_camera_m':point.tolist(),**stats}
                except ValueError as exc:errors[str(exc)]=errors.get(str(exc),0)+1
            else:
                issue=frame.get('depth_error',error);errors[issue]=errors.get(issue,0)+1
        time.sleep(.02)
finally:camera.close()
print(json.dumps({'rgb_frames':len(seen),'exact_paired_frames':paired,'center_sample':example,'issues':errors},ensure_ascii=False))
assert paired>=7,'Insufficient exact RGB/depth pairs'
