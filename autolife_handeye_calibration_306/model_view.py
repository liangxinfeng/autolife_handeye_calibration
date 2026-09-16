"""Reuse the control centre renderer; gripper articulation is schematic only."""
import math
import xml.etree.ElementTree as ET
from control_center_joint_control.robot_view import RobotOpenGLWidget


class CalibrationModel(RobotOpenGLWidget):
    def __init__(self,path):
        super().__init__(path)
        self.mimics=[]
        for joint in ET.parse(path).getroot().findall('joint'):
            mimic=joint.find('mimic')
            if mimic is not None:
                self.mimics.append((joint.get('name'),mimic.get('joint'),
                    float(mimic.get('multiplier','1')),math.degrees(float(mimic.get('offset','0')))))

    def display_joints(self,values):
        result=dict(values)
        # SDK: 10 deg is open, 360 deg closed. Derive URDF endpoints from
        # opposing finger mesh surfaces: closed gap 0, nominal open gap 95 mm.
        # Positive URDF angles OPEN the fingers, unlike positive motor values.
        # Display only: motor/linkage nonlinearity still needs physical calibration.
        for side in ('Left','Right'):
            name=f'Joint_{side}_Gripper'
            fraction=min(1.,max(0.,(float(values.get(name,10))-10)/350))
            closed,opened=(-21.26611456,32.78154895) if side=='Left' else (-21.26597270,32.78242096)
            result[name]=opened+(closed-opened)*fraction
        for name,parent,mult,offset in self.mimics:
            result[name]=result.get(parent,0)*mult+offset
        return result

    def set_joint_states(self,current,target):
        self.current_degrees=self.display_joints(current)
        self.target_degrees=self.display_joints(target)
        self.update()
