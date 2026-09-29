"""Read current driver modes, gripper feedback and marker detection; no commands."""
import os,time,json,yaml
from pathlib import Path
os.environ.update(ROS_DOMAIN_ID='0',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',
 CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>')
import rclpy
from rcl_interfaces.srv import GetParameters
from std_msgs.msg import String
from autolife_handeye_calibration_307.camera import Camera
rclpy.init();node=rclpy.create_node('handeye_readonly_diagnosis');result={};futures={}
def feedback(m):
    d=json.loads(m.data)
    result['feedback']={k:v for k,v in d.items() if 'gripper' in k.lower()}
node.create_subscription(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_306',feedback,10)
clients={side:node.create_client(GetParameters,f'/node_mod_motor_{side}_arm_control_0_306/get_parameters') for side in ('left','right')}
cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/experiment.yaml').read_text());cam=Camera(cfg)
start=time.monotonic()
while time.monotonic()-start<5:
    rclpy.spin_once(node,timeout_sec=.01)
    for side,c in clients.items():
        if side not in futures and c.service_is_ready():
            req=GetParameters.Request();req.names=['param_arm_gripper_joint_control_mode'];futures[side]=c.call_async(req)
for side,f in futures.items():result[side+'_mode']=str(f.result()) if f.done() else 'timeout'
frame,error=cam.get();result['camera_error']=error
if frame:result['marker']={k:v for k,v in frame['detection'].items() if k not in ('C','corners')}
print(json.dumps(result,ensure_ascii=False));cam.close();node.destroy_node();rclpy.shutdown()
