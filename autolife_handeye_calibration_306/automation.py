"""Explicit-start, fail-closed calibration sequencer. No motion on construction.

Collision preflight runs outside Qt/ROS. The UI owns a single state machine;
camera or feedback loss, manual stop and controller rejection never auto-resume.
"""
import copy
import datetime
import json
import math
from pathlib import Path
import time
from PyQt5 import QtCore, QtWidgets as Q
from .core import ARMS, BODY, NECK, Session, atomic_json, sample_guard, has_solution
from .visibility import Visibility


def edge_error(q, edge, moving=False, tolerance=.7):
    """Track the checked one-joint segment, including unchanged joints."""
    name=edge['joint']; start=edge['start_q']; end=edge['end_q']
    for joint,value in start.items():
        if joint not in q or not math.isfinite(q[joint]): return '关节反馈无效：'+joint
        tol=max(tolerance,1.) if 'Gripper' in joint else tolerance
        if joint==name and moving:
            low,high=sorted((value,end[joint]))
            if not low-tol<=q[joint]<=high+tol:return f'运动越过已检查区间：{joint}，实测{q[joint]:.3f}°，区间[{low:.3f}, {high:.3f}]°，容差{tol:.2f}°'
        elif abs(q[joint]-(end[joint] if joint==name else value))>tol:
            expected=end[joint] if joint==name else value
            return f'关节偏离已检查轨迹：{joint}，实测{q[joint]:.3f}°，应保持{expected:.3f}°，偏差{q[joint]-expected:+.3f}°，容差{tol:.2f}°'
    return ''


class Automation:
    def __init__(self,w):
        self.w=w; self.state='idle'; self.process=None; self.plan=None
        self.target_index=0; self.edge_index=0; self.edge=None; self.capture_tries=0
        self.deadline=0.; self.next_try=0.; self.last_issue=''; self.output=b''
        self.request_mode='reset';self.observation=None;self.vision=None
        self.issued=None;self.last_vision_stamp=None
        self.vision_bad_since=None
        self.pair_bad_since=None;self.pair_rejected_stamp=None

    @property
    def active(self):return self.state!='idle'

    def controls(self,enabled):
        w=self.w
        for widget in (w.tabs,w.realtime,w.unlock_button,w.reset_button,w.auto_button,
                       w.begin_button,w.samples,w.model,w.speed):widget.setEnabled(enabled)
        # Model still renders while disabled; no dragging during automation.

    def cancel(self):
        self.state='idle'
        self.pair_bad_since=None;self.pair_rejected_stamp=None
        if self.process is not None:
            proc=self.process;self.process=None
            proc.blockSignals(True);proc.kill();proc.deleteLater()
        self.controls(True)

    def fail(self,reason):
        # Persist the failing edge and recent feedback, not just a joint name.
        try:
            if self.w.session:
                atomic_json(self.w.session.path/'auto_stop.json',{
                    'reason':reason,'state':self.state,'sample':self.target_index+1,
                    'edge_index':self.edge_index,'edge':self.edge,
                    'history':list(self.w.node.history)[-40:]})
        except Exception as exc:self.w.node._event('停止诊断记录未保存：'+str(exc),'warn')
        self.cancel();self.w.stop()
        self.w.node._event('自动采集停止：'+reason,'warn')
        self.w.message('自动采集已停止：'+reason+'。已采数据保留；排查后可手动补拍。')

    def arrival_tolerance(self,joint):
        name='gripper_goal_tolerance_deg' if 'Gripper' in joint else 'goal_tolerance_deg'
        getter=getattr(self.w.node,'get_parameter',None)
        value=float(getter(name).value) if getter else (1. if 'Gripper' in joint else .3)
        if not math.isfinite(value):raise ValueError('到位容差参数无效')
        return max(0.,value)

    def edge_settled(self,q,now):
        """A single in-tolerance packet is not a stopped motor.

        Require fresh, distinct feedback covering 180ms at the endpoint before
        another joint may move. Never retarget a still-settling joint to its
        transient feedback. Capture retains its longer 800ms stability check.
        """
        history=[(t,p) for t,p in self.w.node.history if now-.25<=t<=now]
        if len(history)<3 or now-history[-1][0]>.10 or history[-1][0]-history[0][0]<.18:return False
        end=self.edge['end_q'];moving=self.edge['joint']
        for n in end:
            tolerance=self.arrival_tolerance(n) if n==moving else (1. if 'Gripper' in n else .7)
            values=[p.get(n,math.nan) for _,p in history]+[q[n]]
            if any(not math.isfinite(v) or abs(v-end[n])>tolerance for v in values):return False
            if max(values)-min(values)>(.5 if 'Gripper' in n else .15):return False
        return True

    def start(self):
        w=self.w
        if self.active or w.busy or w.collect:return
        try:
            if w.session:raise ValueError('请先结束当前实验，再开始自动采集')
            if w.realtime.isChecked() or w.node._active_goals:raise ValueError('请关闭实时调整，停稳并解锁就绪后再开始')
            if not w.node._hardware_ready:raise ValueError('请先点击保持式解锁，并等待就绪')
            if not w.board_check.isChecked() or not w.intr_check.isChecked():raise ValueError('先在准备页确认板固定和相机参数')
            w.update_config();cfg=copy.deepcopy(w.cfg);q=w.node.snapshot()
            # Loading an old experiment must not resurrect the former 15%
            # execution profile in the next newly started automatic round.
            for key,value in w.base_cfg.items():
                if key.startswith('auto_') or key in ('calibration_gripper_deg','board_envelope_radius_m'):
                    cfg[key]=copy.deepcopy(value)
            cfg.setdefault('board_size_mm',copy.deepcopy(w.base_cfg['board_size_mm']))
            frame,error=w.camera.get()
            if frame is None or time.monotonic()-frame['received']>.4:raise ValueError(error or 'RGB已过期')
            text=('先检查回程路径，自动将双臂和头部回到初始姿态，装板侧夹爪到240°。\n'
                  '再按当前视野安排12张手臂训练＋4张独立验证＋8张转头验证。手腕转动融合在手臂姿态中。\n'
                  '采集时夹爪、腰腿及另一只手臂不动。复位途中可能暂时看不到标定板。\n'
                  '仅检查模型自碰撞，未感知桌椅等外部障碍。\n'
                  '请确认：板只固定在单片手指、不会滑动；整块板在腕前保守包络内；'
                  '双臂及头颈周围空间清空，实体急停可随时触及。\n'
                  '软件停止只停止后续目标，不替代硬件急停。是否先进行离线路径检查？')
            if Q.QMessageBox.question(w,'自动采集前确认',text,defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
            cfg.update(minimum_training_samples=12,minimum_validation_samples=4,auto_head_validation_samples=8)
            self.cfg=cfg;self.controls(False);w.pages.setCurrentIndex(1)
            self.request_plan(q,'reset')
        except Exception as exc:w.message('尚未开始自动采集：'+str(exc))

    def request_plan(self,q,mode,observation=None):
        self.source=dict(q);self.request_mode=mode;self.observation=observation
        self.state='planning';self.deadline=time.monotonic()+120;self.output=b''
        proc=QtCore.QProcess(self.w);self.process=proc
        proc.readyReadStandardOutput.connect(self.read_output);proc.finished.connect(self.planned)
        proc.errorOccurred.connect(lambda *_:self.fail('离线轨迹检查进程无法运行') if self.state=='planning' else None)
        proc.started.connect(lambda:self.send_request(proc))
        proc.start('/home/ubuntu/miniconda3/envs/robot_env/bin/python',['-m','autolife_handeye_calibration_306.auto_plan_worker'])
        self.w.message('正在检查返回初始姿态的路径…' if mode=='reset' else '初始姿态已就绪，正在根据当前画面安排视野内的采集路径…')

    def send_request(self,proc):
        if self.process is proc:
            proc.write(json.dumps({'q':self.source,'config':self.cfg,'mode':self.request_mode,'observation':self.observation}).encode());proc.closeWriteChannel()

    def read_output(self):
        if self.process:self.output+=bytes(self.process.readAllStandardOutput())

    def planned(self,code,status):
        if self.state!='planning':return
        try:
            self.read_output();proc=self.process;self.process=None
            errors=bytes(proc.readAllStandardError()).decode(errors='replace');proc.deleteLater()
            if code:raise ValueError('碰撞预检未完成：'+errors[-300:])
            result=json.loads(self.output.decode().strip().splitlines()[-1])
            if not result['ok']:raise ValueError(result['reason'])
            q=self.w.node.snapshot()
            if max(abs(q[n]-v) for n,v in self.source.items())>.5:raise ValueError('检查期间机器人姿态发生变化，请重新检查')
            if not self.w.node._hardware_ready:raise ValueError('检查期间硬件已锁定，请重新解锁')
            self.plan=result['plan']
            self.target_index=0;self.edge_index=0;self.state='next_edge';self.edge=None
            if self.request_mode=='reset':
                self.w.message('回程检查通过，自动回到初始姿态；请勿触碰机器人。');return
            self.vision=Visibility(self.cfg,self.observation,self.w.fk)
            steps=[{'phase':t['phase'],'hint':t['hint']} for t in self.plan['targets']]
            directory=Path.home()/'ros2_ws/src/autolife_handeye_calibration_306/sessions'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            session=Session(directory,self.cfg,self.source,self.w.fk.sha,steps=steps)
            atomic_json(directory/'auto_plan.json',self.plan)
            self.w.set_session(session);self.controls(False)
            self.w.message('路径检查通过，开始自动采集。请守在实体急停旁。')
        except Exception as exc:self.fail(str(exc))

    def tick(self,frame):
        if not self.active:return
        w=self.w;now=time.monotonic()
        try:
            if self.state=='planning':
                if now>self.deadline:raise ValueError('路径检查超时，没有执行运动')
                return
            if self.state=='solving':return
            q=w.node.snapshot()
            if not w.node._hardware_ready:raise ValueError('控制器已锁定：'+w.node._last_event)
            if frame is None or now-frame['received']>.5:raise ValueError('RGB断流，停止后续运动')
            resetting=self.request_mode=='reset'
            fixed=BODY+[f'Joint_{"Right" if self.cfg["arm"]=="left" else "Left"}_Gripper']
            if not resetting:fixed+=ARMS['right' if self.cfg['arm']=='left' else 'left']+[f'Joint_{self.cfg["arm"].title()}_Gripper']
            for name in fixed:
                tolerance=1. if 'Gripper' in name else .7
                if abs(q[name]-self.source[name])>tolerance:raise ValueError('固定关节发生变化：'+name)
            if self.state=='reset_settling':
                if now>self.deadline:raise ValueError('已回到初始姿态，但没有清晰看到标定码：'+self.last_issue+'。请微调安装角度后重试')
                d=frame['detection'];self.last_issue=d['reason']
                reason=sample_guard(q,q,'train',self.cfg,w.node.history,now)
                if reason or not d['ok']:return
                if self.cfg['marker_id']<0:self.cfg['marker_id']=d['marker_id']
                observation={'q':q,'C':d['C'],'K':frame['K'].tolist(),'D':frame['D'].tolist(),
                             'size':[frame['image'].shape[1],frame['image'].shape[0]]}
                self.request_plan(q,'collect',observation);return
            target=self.plan['targets'][self.target_index]
            if self.state=='next_edge':
                if self.edge_index==len(target['edges']):
                    self.state='reset_settling' if resetting else 'settling'
                    self.deadline=now+15;self.capture_tries=0;self.next_try=now+.85
                    w.message('关节已到位，等待停稳并检查标定图像…')
                    return
                edge=target['edges'][self.edge_index]
                check=dict(edge);check['end_q']=check['start_q']
                reason=edge_error(q,check)
                if reason:raise ValueError(reason)
                self.edge=edge;self.issued=None;self.last_vision_stamp=None
                self.vision_bad_since=None
                self.pair_bad_since=None;self.pair_rejected_stamp=None
                self.state='moving';self.deadline=now+60
                self.send_motion(q,frame,resetting)
                if self.pair_bad_since is None:
                    w.message('自动复位中…' if resetting else f'自动采集 {self.target_index+1}/{len(self.plan["targets"])} · 视野内移动，停稳后记录')
            elif self.state=='moving':
                reason=edge_error(q,self.edge,moving=True)
                if reason:raise ValueError(reason)
                if now>self.deadline:raise ValueError('关节到位超时')
                # Do not advance a step or capture while time pairing is pending.
                if self.pair_bad_since is not None and self.paired_feedback(frame) is None:return
                final=abs(q[self.edge['joint']]-self.edge['target'])<=self.arrival_tolerance(self.edge['joint'])
                if not w.node._active_goals and final:
                    reason=edge_error(q,self.edge)
                    if reason:raise ValueError(reason)
                    if self.edge_settled(q,now):
                        self.edge_index+=1;self.state='next_edge'
                elif not resetting:self.send_motion(q,frame,False)
                elif not w.node._active_goals:raise ValueError('复位目标结束但实测未到位')
            elif self.state=='settling':
                if now>self.deadline:raise ValueError('当前姿态未得到合格图像：'+self.last_issue)
                if now<self.next_try:return
                try:w.validate_frame(frame)
                except Exception as exc:
                    self.last_issue=str(exc);w.message('等待当前图像合格：'+str(exc));return
                self.state='capturing';self.capture_tries+=1;w.start_capture()
                if w.collect is None:self.capture_failed('不能开始记录')
        except Exception as exc:self.fail(str(exc))

    def paired_feedback(self,frame):
        """Bounded wait; never widen 120ms or reuse a previously rejected frame."""
        now=time.monotonic();stamp=frame['received'];history=self.w.node.history
        nearest=min(history,key=lambda x:abs(x[0]-stamp)) if history else None
        gap=abs(nearest[0]-stamp) if nearest else math.inf
        latest=max((h[0] for h in history),default=None)
        detail=(f'配对差={gap*1000:.1f}ms，图像年龄={(now-stamp)*1000:.1f}ms，'
                +('反馈历史为空' if latest is None else f'最新反馈年龄={(now-latest)*1000:.1f}ms'))
        if self.pair_bad_since is not None and now-self.pair_bad_since>=1.:
            raise ValueError('图像与关节反馈持续1秒未完成重新配对：'+detail)
        if nearest is None or not math.isfinite(gap) or gap>.12:
            if self.pair_bad_since is None:
                self.pair_bad_since=now
                self.w.node._event('自动采集暂缓新目标：'+detail,'warn')
            self.pair_rejected_stamp=max(stamp,self.pair_rejected_stamp or stamp)
            self.w.message('暂缓新运动目标，等待新图像与关节反馈配对（最多1秒）…')
            return None
        if self.pair_bad_since is not None:
            if stamp<=self.pair_rejected_stamp:return None
            self.w.node._event(f'图像反馈配对恢复，等待{(now-self.pair_bad_since)*1000:.0f}ms；'+detail)
            self.pair_bad_since=None;self.pair_rejected_stamp=None
            self.w.message('图像反馈配对已恢复，继续当前步骤的视野及到位检查。')
        return nearest

    def send_motion(self,q,frame,resetting):
        w=self.w;name=self.edge['joint'];goal=self.edge['target']
        if 'Gripper' in name:group='grippers';names=['Joint_Left_Gripper','Joint_Right_Gripper']
        elif name in NECK:group='neck';names=NECK
        else:
            side='left' if name in ARMS['left'] else 'right';group=side+'_arm';names=ARMS[side]
        # Hold checked coordinates, never integrate instantaneous feedback into
        # the next target (especially a forearm still settling after rotation).
        values={n:self.edge['start_q'][n] for n in names}
        if not resetting:
            if self.issued is not None and abs(self.issued-goal)<.001:return
            if self.last_vision_stamp==frame['received']:return
            d=frame['detection']
            if not d['ok'] or d.get('marker_id')!=self.cfg['marker_id']:
                if self.vision_bad_since is None:self.vision_bad_since=time.monotonic()
                if time.monotonic()-self.vision_bad_since>.6:raise ValueError('视觉跟踪持续不可靠，已停止：'+d['reason'])
                w.message('暂缓发送新目标，等待标定板图像恢复清晰…');return
            self.vision_bad_since=None
            nearest=self.paired_feedback(frame)
            if nearest is None:return
            self.vision.anchor(nearest[1],d['C'])
            horizon=self.cfg.get('auto_visual_horizon_deg',2.)
            if group=='neck':horizon=min(horizon,2.)
            final_target=abs(goal-q[name])<=horizon
            value=goal if final_target else q[name]+math.copysign(horizon,goal-q[name])
            predicted={**q,**values,name:value}
            # Check the full short sweep, including small hold corrections;
            # a larger horizon must not check only its endpoint.
            steps=max(1,math.ceil(max(abs(predicted[n]-q[n]) for n in names)/.5))
            if not self.vision.check(q,20) or any(not self.vision.check(
                    {**q,**{n:q[n]+(predicted[n]-q[n])*i/steps for n in names}},24)
                    for i in range(1,steps+1)):
                raise ValueError('标定板接近画面边缘，已提前停止，请重新居中后采集')
            # Intermediate jitter suppression must never swallow the endpoint:
            # the bridge can finish a subgoal 0.3--0.5° short of the final pose.
            # Keep all fresh-frame/visibility checks above for this final send.
            if not final_target and self.issued is not None and abs(value-self.issued)<.5:return
            self.last_vision_stamp=frame['received']
        else:value=goal
        values[name]=value
        w.changing=True
        try:
            for n,v in values.items():w.rows[n][2].setValue(v)
        finally:w.changing=False
        speed=self.cfg.get('auto_reset_speed_scale',.45) if resetting else self.cfg.get('auto_speed_scale',.45)
        if group=='neck':speed=min(speed,.45)
        if group=='grippers':speed=min(speed,.25)
        w.node.apply(group,values,speed_scale=speed);self.issued=value
        if group not in w.node._active_goals and abs(q[name]-value)>self.arrival_tolerance(name):raise ValueError('运动指令被拒绝：'+w.node._last_event)

    def capture_failed(self,reason):
        if not self.active:return
        self.last_issue=reason
        if self.capture_tries>=3:self.fail('同一姿态连续3次采样不合格：'+reason)
        else:self.state='settling';self.next_try=time.monotonic()+1

    def captured(self):
        self.target_index+=1;self.edge_index=0;self.edge=None
        targets=self.plan['targets'];total=len(targets)
        # Solve once before independent head validation, then only when the
        # entire actual plan is captured. Never hard-code a 20-image boundary.
        entering_head=(0<self.target_index<total and
                       targets[self.target_index]['phase']=='head' and
                       targets[self.target_index-1]['phase']!='head')
        if self.target_index==total or entering_head:
            self.state='solving';self.w.solve();return
        self.state='next_edge'

    def solved(self,report,error):
        if not self.active:return
        # Accuracy is advisory: it must not stop the remaining head holdouts.
        # Those poses still use the original collision and live visibility guards.
        if error or not has_solution(report):
            self.fail(error or '未获得可计算的外参，请在评估页查看数据');self.w.goto_page(2);return
        total=len(self.plan['targets'])
        if self.target_index==total:
            self.cancel();self.w.stop();self.w.message(f'{total}张采集及计算已完成。'+report['status']+'；转头验证：'+('通过' if report['moving_head_pass'] else '精度提示，详见报告'))
            self.w.goto_page(2)
        else:
            self.state='next_edge';self.w.busy=False
            quality='验证通过' if report['fixed_head_pass'] else '精度未达推荐值，不中断流程'
            self.w.message(f'{self.target_index}张已计算外参（'+quality+f'），自动继续{total-self.target_index}张转头验证。')
