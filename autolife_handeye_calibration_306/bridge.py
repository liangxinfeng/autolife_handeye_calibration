"""Reuse the control centre guard, with all hardware actions explicitly UI-gated."""
from collections import deque
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from control_center_joint_control.joint_bridge import JointBridge
from .core import ARMS, BODY, NECK

def competing_controller():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: text=path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,PermissionError):continue
        if 'python' in text and ('navigation_web_bridge' in text or
             'openarmx_teleop_vr_' in text and 'controller_node' in text):
            return '检测到VR遥操控制器正在运行，请先正常退出VR遥操程序'
    return ''

class CalibrationBridge(JointBridge):
    def __init__(self):
        self.history=deque(maxlen=400); self.gui_call=False; self.gate=lambda group:''
        self.sampling=False; self.conflict=''; self.last_conflict_check=0; self.progress_watch={}
        self.manual_speed_scale=1.0
        super().__init__()

    def _on_feedback(self, message):
        super()._on_feedback(message)
        if self._is_feedback_fresh(): self.history.append((time.monotonic(),dict(self._current)))

    def _on_apply(self, message):
        if not self.gui_call: return
        group=json.loads(message.data).get('group')
        reason=self.gate(group)
        if self.sampling or reason or self.conflict:
            self._event(reason or self.conflict or '采样中禁止运动','warn'); return
        publisher=self._gripper_pub if group=='grippers' else self._whole_body_pub
        if publisher.get_subscription_count()==0:
            self._lock('没有电机指令接收端，请检查机械臂服务；没有发送运动命令'); return
        super()._on_apply(message)
        if group in self._active_goals and group not in self.progress_watch:
            self.progress_watch[group]=(time.monotonic(),dict(self._current))

    def _on_action_command(self, message):
        return # no external action library execution in a calibration session

    def _set_hardware_enabled_mode(self, request, response, prepare_reset):
        if not self.gui_call or (request.data and prepare_reset):
            response.success=False; response.message='标定工具仅允许界面显式的保持式解锁'; return response
        self.conflict=competing_controller()
        if request.data and self.conflict:
            response.success=False; response.message=self.conflict; return response
        return super()._set_hardware_enabled_mode(request,response,False)

    def _set_leg_waist_enabled(self, request, response):
        response.success=False; response.message='标定期间不开放腰腿运动'; return response

    def _advance_motion(self):
        if time.monotonic()-self.last_conflict_check>1:
            self.conflict=competing_controller(); self.last_conflict_check=time.monotonic()
        if self.conflict and self._hardware_enabled:self._lock(self.conflict)
        for group in list(self._active_goals):
            if self.gate(group): self._lock('采样阶段变化，已停止调整')
        for group,(stamp,previous) in list(self.progress_watch.items()):
            if group not in self._active_goals:
                self.progress_watch.pop(group,None); continue
            goal=self._active_goals[group]['targets']
            moved=max(abs(self._current[n]-previous[n]) for n in goal)
            if moved>.15:
                self.progress_watch[group]=(time.monotonic(),dict(self._current))
            elif time.monotonic()-stamp>5:
                self._lock('目标已发送但实测关节5秒无跟随，已锁定。检查驱动使能/故障/机械阻挡，禁止继续追赶'); break
        super()._advance_motion()

    def _lock(self,reason):
        self.progress_watch.clear()
        super()._lock(reason)

    def unlock(self):
        self.gui_call=True
        try:return self._set_hardware_enabled_mode(SimpleNamespace(data=True),SimpleNamespace(),False)
        finally:self.gui_call=False

    def apply(self,group,targets,speed_scale=None):
        self.gui_call=True
        speed=self.manual_speed_scale if speed_scale is None else speed_scale
        try:self._on_apply(SimpleNamespace(data=json.dumps({'group':group,'targets_deg':targets,'speed_scale':speed})))
        finally:self.gui_call=False

    def snapshot(self):
        if not self._is_feedback_fresh(): raise ValueError('关节反馈过期，不能记录或调整')
        q=dict(self._current)
        names=BODY+NECK+ARMS['left']+ARMS['right']+['Joint_Left_Gripper','Joint_Right_Gripper']
        lost=self._communication_lost_joints(names)
        if lost:raise ValueError('关节通信离线：'+', '.join(lost[:3]))
        if not all(n in q for n in names):raise ValueError('全身关节反馈不完整')
        if not all(math.isfinite(q[n]) for n in names):raise ValueError('关节反馈含非有限数值')
        return q
