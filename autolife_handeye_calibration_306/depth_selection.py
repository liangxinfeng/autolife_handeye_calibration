"""Selection-only Qt helper; never sends a movement command."""
import time
import numpy as np
from PyQt5 import QtCore,QtWidgets as Q
from .core import sample_guard
from .depth_target import PointSamples,torso_point


def image_pixel(pos,rect,image_shape):
    x,y=pos;left,top,width,height=rect
    if width<=0 or height<=0 or not left<=x<left+width or not top<=y<top+height:return None
    h,w=image_shape[:2]
    return (min(w-1,int((x-left)*w/width)),min(h-1,int((y-top)*h/height)))


class ClickImage(Q.QLabel):
    pixel_clicked=QtCore.pyqtSignal(object)
    source_shape=None
    def mousePressEvent(self,event):
        pm=self.pixmap()
        if event.button()==QtCore.Qt.LeftButton and pm and self.source_shape is not None:
            area=self.contentsRect();width=pm.width()/pm.devicePixelRatio();height=pm.height()/pm.devicePixelRatio()
            rect=(area.x()+(area.width()-width)/2,area.y()+(area.height()-height)/2,width,height)
            p=image_pixel((event.pos().x(),event.pos().y()),rect,self.source_shape)
            if p is not None:self.pixel_clicked.emit(p)
        super().mousePressEvent(event)


class DepthSelection:
    def __init__(self,verification):
        self.v=verification;self.pixel=None;self.result=None;self.samples=PointSamples();self.last_issue=''

    def clear(self):
        self.pixel=None;self.result=None;self.samples.clear()

    def begin(self,pixel,frame):
        v=self.v;w=v.w
        if v.active or not v.is_depth or v.data is None:return
        try:
            now=time.monotonic();q=w.node.snapshot()
            if frame is None or now-frame['received']>.4:raise ValueError('等待实时RGB，不能在旧图上选点')
            if w.node._active_goals:raise ValueError('请等待关节停稳再选择目标')
            reason=sample_guard(q,q,'train',v.data['config'],w.node.history,now)
            if reason:raise ValueError(reason)
            self.clear();v.invalidate();self.pixel=tuple(pixel);self.source=q;self.deadline=now+4
            v.state='selecting';v.controls(False);v.say('正在采集7个独立深度帧，请保持机器人和物体不动…')
        except Exception as exc:v.say('未选择目标：'+str(exc))

    def tick(self,frame):
        v=self.v;w=v.w;now=time.monotonic()
        if v.state!='selecting':return
        try:
            if now>self.deadline:raise RuntimeError('深度选点未通过：'+self.last_issue+'；请点击物体内部平整区域重试')
            q=w.node.snapshot()
            for n,value in self.source.items():
                if abs(q[n]-value)>(1. if 'Gripper' in n else .35):raise RuntimeError('采样期间机器人发生移动，请停稳后重选')
            point,camera,stats=torso_point(v.data,w.fk,w.node.history,frame,self.pixel,now)
            result=self.samples.add(frame['depth_id'],point,camera,stats)
            if result:
                self.result={**result,'pixel':list(self.pixel),'locked_at':now,'source_q':self.source,
                             'depth_scale_m':frame['depth_scale_m']}
                v.state='idle';v.controls(True)
                v.say(f'选点已锁定：深度{result["depth_m"]:.3f}m，{result["inlier_frames"]}/7帧一致，波动{result["spread_mm"]:.1f}mm。点击规划不运动；精度仍需实物验证。')
            else:v.status.setText(f'选点采样 {len(self.samples.points)}/7 · 请保持不动')
        except ValueError as exc:
            self.last_issue=str(exc)
            # Severe temporal disagreement is not rescued by appending more data.
            if len(self.samples.points)>=7:
                self.clear();v.state='idle';v.controls(True);v.say(self.last_issue)
        except Exception as exc:
            self.clear();v.state='idle';v.controls(True);v.say(str(exc))

    def target(self,frame,stationary):
        v=self.v;w=v.w;r=self.result
        if r is None:raise ValueError('请先在RGB物体内部点击，完成多帧深度选点')
        if time.monotonic()-r['locked_at']>60:raise ValueError('选点已超过60秒，请重新点击')
        q=w.node.snapshot()
        for n,value in r['source_q'].items():
            if abs(q[n]-value)>(1. if 'Gripper' in n else .35):raise ValueError('选点后关节姿态发生变化，请重选')
        if stationary:
            check,_,_=torso_point(v.data,w.fk,w.node.history,frame,self.pixel,time.monotonic())
            if np.linalg.norm(check-np.array(r['point']))>.012:raise ValueError('点击处表面位置已变化，请重新选点')
        T=np.eye(4);T[:3,3]=r['point'] # orientation deliberately unspecified
        direction=np.array(r['camera'])-T[:3,3];direction/=np.linalg.norm(direction)
        return T,T[:3,3]+direction*v.standoff.value()/1000.
