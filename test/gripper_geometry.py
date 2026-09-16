"""Offline URDF finger separation, no hardware."""
import xml.etree.ElementTree as ET
import numpy as np
from control_center_joint_control.robot_view import RobotModel,_load_binary_stl
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
path=Path(get_package_share_directory('control_center_joint_control'))/'urdf/robot_v2_2.urdf'
model=RobotModel(path);root=ET.parse(path).getroot()
axis=None
for angle in [0,-45,-40,-35,-30,-20,20,40]:
    q={'Joint_Left_Gripper':angle}
    for j in root.findall('joint'):
        m=j.find('mimic')
        if m is not None:q[j.get('name')]=q.get(m.get('joint'),0)*float(m.get('multiplier','1'))+np.degrees(float(m.get('offset','0')))
    tf=model.link_transforms(q);pts=[];clouds=[]
    for link in ['Link_Left_Gripper_Left_Finger','Link_Left_Gripper_Right_Finger']:
        v=next(v for v in model.visuals if v.link==link);vertices,_=_load_binary_stl(v.mesh_path)
        center=(vertices.min(axis=0)+vertices.max(axis=0))/2
        pts.append((tf[link]@v.origin@np.r_[center,1])[:3])
        matrix=tf[link]@v.origin;clouds.append(vertices@matrix[:3,:3].T+matrix[:3,3])
    if axis is None:axis=(pts[1]-pts[0])/np.linalg.norm(pts[1]-pts[0])
    gap=float(np.min(clouds[1]@axis)-np.max(clouds[0]@axis))
    print(angle,round(float(np.linalg.norm(pts[0]-pts[1]))*1000,2),'gap',round(gap*1000,2))
