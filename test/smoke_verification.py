"""Installed UI + live read-only camera; isolated ROS domain, NEVER enables."""
import json
import os
from pathlib import Path
import time
os.environ['ROS_DOMAIN_ID']='88'
os.environ['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>200</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
import yaml
import rclpy
from PyQt5 import QtCore,QtWidgets as Q
from ament_index_python.packages import get_package_share_directory
from autolife_handeye_calibration_306.app import Window,STYLE
from autolife_handeye_calibration_306.bridge import CalibrationBridge
import autolife_handeye_calibration_306.verification as mod

rclpy.init();app=Q.QApplication([]);app.setStyleSheet(STYLE)
node=CalibrationBridge()
share=Path(get_package_share_directory('autolife_handeye_calibration_306'))
w=Window(node,yaml.safe_load((share/'config/experiment.yaml').read_text()))
w.goto_page(3);w.verification.mode.setCurrentIndex(1);w.show();result={}
def done():
    frame,error=w.camera.get();v=w.verification
    result.update(module=mod.__file__,camera_fresh=bool(frame and time.monotonic()-frame['received']<.5),
                  camera_error=error,hardware_enabled=node._hardware_enabled,
                  page=w.pages.currentIndex(),model_error=v.model.render_error,meshes=len(v.model.meshes),
                  experiment=str(v.path),depth_available=frame.get('depth') is not None if frame else False,
                  depth_error=frame.get('depth_error') if frame else None,mode=v.mode.currentText())
    w.grab().save('/tmp/handeye_verify306_live_ui.png');w.close()
QtCore.QTimer.singleShot(6500,done)
app.exec_();node.destroy_node();rclpy.shutdown()
print(json.dumps(result,ensure_ascii=False))
assert result['hardware_enabled'] is False and result['page']==3
assert result['meshes']>0 and not result['model_error']
assert result['camera_fresh'],result
