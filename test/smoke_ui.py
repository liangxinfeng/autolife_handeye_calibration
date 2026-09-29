"""Offscreen read-only UI check. Never calls unlock/apply or creates a session."""
import json
import os
import sys
import time
from pathlib import Path
import yaml
import rclpy
from PyQt5 import QtCore, QtWidgets
from autolife_handeye_calibration_307.app import Window, STYLE
from autolife_handeye_calibration_307.bridge import CalibrationBridge

root=Path(__file__).resolve().parents[1]
os.environ['ROS_DOMAIN_ID']='0'
os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
args=['--ros-args']
for name in ['joint_states','target_joint_states','status','preview_command','apply_command','action_command',
             'set_hardware_enabled','set_hardware_enabled_without_reset','set_leg_waist_enabled']:
    args+=['-r',f'/control_center_joint_control/{name}:=/handeye306/{name}']
rclpy.init(args=args)
app=QtWidgets.QApplication([]);app.setStyleSheet(STYLE)
node=CalibrationBridge()
config=yaml.safe_load((root/'config/experiment.yaml').read_text())
window=Window(node,config);window.resize(1500,940);window.show()
result={}
directory=Path('/home/ubuntu/ros2_ws/diagnostics');directory.mkdir(parents=True,exist_ok=True)
def show_work():
    window.grab().save(str(directory/'handeye306_prepare.png'))
    window.goto_page(1)
QtCore.QTimer.singleShot(8000,show_work)
def check():
    frame,error=window.camera.get()
    result.update(feedback_fresh=node._is_feedback_fresh(),joints=len(node._current),
        hardware_enabled=node._hardware_enabled,camera_error=error,
        camera_fresh=bool(frame and time.monotonic()-frame['received']<.5),
        image_shape=None if frame is None else list(frame['image'].shape),
        intrinsics=None if frame is None else str(frame['intrinsics']),
        status=window.control_status.text(),model_error=window.model.render_error,
        mesh_count=len(window.model.meshes),model_context=window.model.isValid())
    path=Path('/home/ubuntu/ros2_ws/diagnostics/handeye306_ui.png');path.parent.mkdir(parents=True,exist_ok=True)
    window.grab().save(str(path));window.close()
QtCore.QTimer.singleShot(12000,check)
app.exec_();node.destroy_node();rclpy.shutdown()
print(json.dumps(result,ensure_ascii=False))
assert result['hardware_enabled'] is False
assert result['feedback_fresh'],result
assert result['camera_fresh'],result
if os.environ.get('QT_QPA_PLATFORM')!='offscreen':
    assert result['model_context'] and result['mesh_count']>0 and not result['model_error'],result
