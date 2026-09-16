"""Nominal-URDF visibility prediction only, never used as calibration output.

Anchor the finger marker using the current image. Re-anchor from fresh frames
during motion; bound each motor target so prediction errors cannot accumulate
through a whole joint excursion. The final solver uses only captured samples.
"""
import numpy as np
import cv2
from .core import Kinematics,TORSO,MOUNT,object_corners,pose_error


class Visibility:
    def __init__(self,cfg,observation,fk=None):
        self.cfg=cfg;self.fk=fk or Kinematics(cfg['urdf_path'])
        self.tip=f'Link_{cfg["arm"].title()}_Wrist_Lower_to_Gripper'
        # Same CAD optical convention as adaptive_object_grasping hardware.yaml.
        self.optical=np.array([[0.,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
        self.K=np.array(observation['K']);self.D=np.array(observation['D'])
        self.width,self.height=observation['size']
        self.anchor(observation['q'],observation['C'])

    def camera(self,q):return self.fk.fk(TORSO,MOUNT,q)@self.optical
    def wrist(self,q):return self.fk.fk(TORSO,self.tip,q)

    def anchor(self,q,C):
        self.Y=np.linalg.inv(self.wrist(q))@self.camera(q)@np.array(C)

    def corners(self,q,board=False):
        C=np.linalg.inv(self.camera(q))@self.wrist(q)@self.Y
        size=max(self.cfg.get('board_size_mm',[75,75])) if board else self.cfg['marker_size_mm']
        obj=object_corners(size/1000)
        if np.min((obj@C[:3,:3].T+C[:3,3])[:,2])<=.05:return None
        return cv2.projectPoints(obj,cv2.Rodrigues(C[:3,:3])[0],C[:3,3],self.K,self.D)[0].reshape(-1,2)

    def check(self,q,margin=None):
        # Preserve physical field-of-view reserve when the native size changes.
        margin=self.cfg.get('auto_view_margin_px',32.)*min(self.width/640.,self.height/480.) if margin is None else margin
        pts=self.corners(q,True);black=self.corners(q)
        if pts is None or black is None or not np.all(np.isfinite(pts)):return False
        return bool(np.min(pts[:,0])>=margin and np.max(pts[:,0])<=self.width-margin
                    and np.min(pts[:,1])>=margin and np.max(pts[:,1])<=self.height-margin
                    and np.min(np.linalg.norm(np.roll(black,-1,axis=0)-black,axis=1))>=self.cfg['minimum_marker_pixels']+1)

    def centering_cost(self,q):
        pts=self.corners(q,True)
        if pts is None:return float('inf')
        return float(np.linalg.norm((pts.mean(axis=0)-np.array([self.width,self.height])/2)/np.array([self.width,self.height])))

    def distinct(self,q,others,head=False):
        for other in others:
            pe,ae=pose_error(self.wrist(q),self.wrist(other))
            _,he=pose_error(self.camera(q),self.camera(other))
            if pe<17 and ae<5.5 and (not head or he<3.5):return False
        return True
