"""Explicitly gated final workflow step: observe -> plan -> execute -> inspect."""
import copy
import datetime
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets as Q
from ament_index_python.packages import get_package_share_directory
from .core import ARMS, NECK, BODY, Session, atomic_json, sample_guard
from .model_view import CalibrationModel
from .verification_core import (load_geometry, checked_observation, tcp_position,
                                check_path_state, camera_geometry, CAMERA_SOURCES)
from .verification_trajectory import JointPath,PathFollower
from .depth_selection import ClickImage,DepthSelection


class Verification:
    def __init__(self, w):
        self.w=w; self.data=None; self.path=None; self.plan=None; self.process=None
        self.state='idle'; self.edge=None; self.seen=None; self.output=b''
        self.index=0; self.run_path=None
        self.page=Q.QWidget(); layout=Q.QVBoxLayout(self.page)
        title=Q.QLabel('④ 实物精度验证 · 夹爪中心对准靶标 / 深度选点（不开合）')
        title.setStyleSheet('font-size:21px;font-weight:bold'); layout.addWidget(title)
        self.info=Q.QLabel('读取已有标定，不修改原始结果'); self.info.setWordWrap(True); layout.addWidget(self.info)
        comparison=Q.QHBoxLayout();layout.addLayout(comparison)
        comparison.addWidget(Q.QLabel('对比相机外参'))
        self.camera_source=Q.QComboBox()
        for key,label in CAMERA_SOURCES.items():self.camera_source.addItem(label,key)
        comparison.addWidget(self.camera_source)
        self.source_hint=Q.QLabel();self.source_hint.setWordWrap(True);comparison.addWidget(self.source_hint,1)
        modes=Q.QHBoxLayout();layout.addLayout(modes)
        modes.addWidget(Q.QLabel('目标来源'))
        self.mode=Q.QComboBox();self.mode.addItems(['ArUco靶标','图像物体选点 · 深度']);self.mode.setMinimumWidth(220);modes.addWidget(self.mode)
        self.clear_point=Q.QPushButton('清除选点');modes.addWidget(self.clear_point)
        self.point_hint=Q.QLabel('选深度模式后，点击RGB中的物体内部；请避开边缘、透明和反光表面。');self.point_hint.setWordWrap(True);modes.addWidget(self.point_hint,1)
        split=Q.QSplitter(); layout.addWidget(split,1)
        self.video=ClickImage('等待实时RGB'); self.video.setMinimumSize(400,300)
        self.video.setAlignment(QtCore.Qt.AlignCenter); self.video.setStyleSheet('background:#050b10')
        self.video.setSizePolicy(Q.QSizePolicy.Ignored,Q.QSizePolicy.Expanding); split.addWidget(self.video)
        urdf=Path(get_package_share_directory('control_center_joint_control'))/'urdf/robot_v2_2.urdf'
        self.model=CalibrationModel(str(urdf)); self.model.setMinimumSize(330,300)
        self.model.set_interaction_mode('camera'); split.addWidget(self.model); split.setSizes([800,450])
        self.pose=Q.QLabel('靶标位姿等待中'); self.pose.setWordWrap(True); layout.addWidget(self.pose)
        self.status=Q.QLabel('先加载结果，再选择靶标或深度点；不限制起始夹爪开度，验证中不开合夹爪。')
        self.status.setWordWrap(True); self.status.setObjectName('guide'); layout.addWidget(self.status)
        line=Q.QHBoxLayout(); layout.addLayout(line)
        def button(text, slot):
            b=Q.QPushButton(text); b.clicked.connect(slot); line.addWidget(b); return b
        self.load_button=button('读取最新/本轮标定',self.load)
        self.unlock_button=button('保持式解锁',self.unlock)
        line.addWidget(Q.QLabel('原靶标 → 夹取中心'))
        self.offset=Q.QDoubleSpinBox(); self.offset.setRange(0,50); self.offset.setValue(15); self.offset.setSuffix(' mm（板内侧）'); line.addWidget(self.offset)
        line.addWidget(Q.QLabel('目标前预留'))
        self.standoff=Q.QDoubleSpinBox(); self.standoff.setRange(0,100); self.standoff.setValue(30); self.standoff.setSuffix(' mm'); line.addWidget(self.standoff)
        self.offset.valueChanged.connect(self.invalidate); self.standoff.valueChanged.connect(self.invalidate)
        self.confirm=Q.QCheckBox('确认原板贴在夹爪上表面，中心在板内侧15mm且无前后偏移；开度不同于标定时，真实夹取中心可能变化')
        self.confirm.toggled.connect(self.invalidate); layout.addWidget(self.confirm)
        line=Q.QHBoxLayout(); layout.addLayout(line)
        self.plan_button=button('① 锁定目标并规划（不运动）',self.prepare)
        self.start_button=button('② 开始运动…',self.start); self.start_button.setEnabled(False)
        stop=button('停止 / 锁定',w.stop); stop.setObjectName('stop')
        button('返回调姿',lambda:w.goto_page(1))
        layout.addWidget(Q.QLabel('实体模型＝反馈，蓝色虚影＝规划终点。只检查模型自碰撞，不含桌子/盒子/人；请清空路径。软件停止不代替实体急停。'))
        w.pages.addWidget(self.page)
        self.depth=DepthSelection(self);self.display_frame=None
        self.camera_source.currentIndexChanged.connect(self.source_changed)
        self.source_changed()
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.clear_point.clicked.connect(self.clear_selection)
        self.video.pixel_clicked.connect(lambda p:self.depth.begin(p,self.display_frame))

    @property
    def active(self):return self.state in ('selecting','planning','moving')

    @property
    def is_depth(self):return self.mode.currentIndex()==1

    def clear_selection(self):
        if self.active:return
        self.depth.clear();self.invalidate();self.seen=None
        self.say('请在RGB物体内部重新点击；点击不会运动。')

    def mode_changed(self,*_):
        if self.active:return
        self.clear_selection();self.configure_camera()
        self.say('深度选点：点击物体内部，等待7帧位置确认，再规划。' if self.is_depth else 'ArUco模式：将57mm靶标完整放入画面。')

    def source_changed(self,*_):
        if self.active:
            # Also protect programmatic changes, not only disabled UI clicks.
            source=(self.data or {}).get('_verification_camera_source','calibrated')
            self.camera_source.blockSignals(True)
            self.camera_source.setCurrentIndex(self.camera_source.findData(source))
            self.camera_source.blockSignals(False)
            return
        source=self.camera_source.currentData()
        if self.data is not None:self.data['_verification_camera_source']=source
        self.invalidate();self.depth.clear();self.seen=None
        self.source_hint.setText('当前：'+CAMERA_SOURCES[source]+'。仅切换相机外参；两组使用相同夹爪中心和15mm补偿，不是整机纯URDF对比。')
        self.say('已切换为'+CAMERA_SOURCES[source]+'。旧目标和规划已清除，请重新选点/规划；不会自动运动。')

    def configure_camera(self):
        if self.data is None:return
        self.detect_cfg={**self.data['config'],'marker_id':-1,'minimum_marker_pixels':32.,
                         'read_depth':self.is_depth,'depth_scale_m':self.w.base_cfg.get('depth_scale_m',.001)}
        self.w.camera.configure(self.detect_cfg)

    def say(self,text):
        self.status.setText(text); self.w.message(text)

    def controls(self,enabled):
        self.w.automation.controls(enabled)
        for widget in (self.load_button,self.unlock_button,self.offset,self.standoff,self.confirm,self.plan_button,self.mode,self.clear_point,self.camera_source):
            widget.setEnabled(enabled)
        self.start_button.setEnabled(enabled and self.plan is not None)

    def invalidate(self,*_):
        self.plan=None; self.start_button.setEnabled(False)

    def enter(self):
        if self.data is None:self.load()
        else:self.w.camera.configure(self.detect_cfg)

    def leave(self):
        self.invalidate(); self.depth.clear(); self.w.camera.configure(self.w.cfg)

    def load(self):
        if self.active:return
        try:
            root=Path.home()/'ros2_ws/src/autolife_handeye_calibration_306/sessions'
            if self.w.session and self.w.session.data.get('report'):
                s=self.w.session
            else:
                paths=sorted(root.glob('*/calibration_candidate.json'),reverse=True)
                if not paths:raise ValueError('尚无标定结果，请先完成采集计算')
                s=Session.load(paths[0].parent)
            load_geometry(s.data,self.w.fk,self.offset.value())
            self.data=copy.deepcopy(s.data); self.path=s.path; self.source_changed()
            # Physical verification can tolerate a smaller marker than calibration
            # capture. Keep the saved calibration config and other quality gates.
            self.depth.clear();self.configure_camera()
            report=self.data['report']; error=report.get('validation',{}).get('position_p95_mm')
            error_text='未提供' if error is None else f'{error:.1f}mm'
            self.info.setText(f'实验：{s.path.name} · 黑框{self.detect_cfg["marker_size_mm"]:g}mm · 自动识别唯一ID · 验证最短边≥32px\n'
                +report['status']+f'；独立验证P95位置残差 {error_text}（不是实物精度）。原相机外参保留不变。')
            self.say('已加载。先把靶标完整放入画面，确认夹爪中心偏移；点击规划不会运动。')
        except Exception as exc:
            self.data=None; self.invalidate(); self.say('未加载：'+str(exc))

    def unlock(self):
        if self.active:return
        self.invalidate()
        if Q.QMessageBox.question(self.w,'保持式解锁','退出其他遥操/关节程序，并确认机器人周围安全。\n仅保持当前位置准备控制，不复位、不改变夹爪开度。',defaultButton=Q.QMessageBox.No)==Q.QMessageBox.Yes:
            self.say(self.w.node.unlock().message)

    def snapshot(self,frame,stationary=False):
        if self.data is None:raise ValueError('请先加载标定结果')
        w=self.w; q=w.node.snapshot(); cfg=self.data['config']; now=time.monotonic()
        # Accept the current opening; do not move the gripper to calibration
        # opening. Runtime still holds the planned geometry unchanged.
        if now-w.node._feedback_monotonic>.3:raise ValueError('关节反馈过期')
        if stationary:
            reason=sample_guard(q,q,'train',cfg,w.node.history,now)
            if reason:raise ValueError(reason)
            if w.node._active_goals or w.node._hardware_prepare:raise ValueError('仍在调整或使能中')
        if self.is_depth:target,goal=self.depth.target(frame,stationary)
        else:target,goal=checked_observation(self.data,w.fk,w.node.history,frame,now,self.standoff.value())
        return q,target,goal

    def prepare(self):
        if self.active:return
        try:
            self.invalidate()
            if not self.confirm.isChecked():raise ValueError('先确认15毫米偏移及夹取中心误差提示')
            frame,_=self.w.camera.get(); q,target,goal=self.snapshot(frame,True)
            tcp=load_geometry(self.data,self.w.fk,self.offset.value())
            self.request={'config':self.data['config'],'q':q,'tcp':tcp.tolist(), 'goal':goal.tolist(),
                          'urdf_sha256':self.w.fk.sha}
            self.comparison=camera_geometry(self.data)
            self.request['comparison']=copy.deepcopy(self.comparison)
            self.target=target; self.marker_id=None if self.is_depth else frame['detection']['marker_id']; self.source=q
            self.state='planning'; self.controls(False); self.output=b''; self.deadline=time.monotonic()+60
            proc=QtCore.QProcess(self.w); self.process=proc
            proc.readyReadStandardOutput.connect(self.read_output); proc.finished.connect(self.planned)
            proc.errorOccurred.connect(lambda *_:self.fail('规划进程未能启动') if self.state=='planning' else None)
            proc.started.connect(lambda:self.send_request(proc))
            proc.start('/home/ubuntu/miniconda3/envs/robot_env/bin/python', ['-m','autolife_handeye_calibration_306.verify_plan_worker'])
            self.say('已锁定目标，正在计算位置IK和局部自碰撞检查；没有发送运动命令。')
        except Exception as exc:self.say('未开始规划：'+str(exc))

    def send_request(self,proc):
        if self.process is proc:
            proc.write(json.dumps(self.request).encode()); proc.closeWriteChannel()

    def read_output(self):
        if self.process:self.output+=bytes(self.process.readAllStandardOutput())

    def planned(self,code,status):
        if self.state!='planning':return
        try:
            self.read_output(); proc=self.process; self.process=None
            errors=bytes(proc.readAllStandardError()).decode(errors='replace'); proc.deleteLater()
            if code:raise ValueError(errors[-300:] or '离线路径检查失败')
            result=json.loads(self.output.decode().strip().splitlines()[-1])
            if not result['ok']:raise ValueError(result['reason'])
            self.state='idle'; self.plan=result['plan']; self.planned_at=time.monotonic(); self.controls(True)
            self.say(f'规划完成：位移{self.plan["distance_mm"]:.0f}mm，模型终点残差{self.plan["nominal_error_mm"]:.1f}mm。'
                     '检查蓝色虚影和外部空间；已解锁可点击开始运动，未解锁则先解锁并重新规划。')
        except Exception as exc:self.fail(str(exc))

    def start(self):
        if self.active or self.plan is None:return
        try:
            if camera_geometry(self.data)!=self.comparison:raise ValueError('外参来源已改变，请重新规划')
            text=('本次外参：'+self.comparison['camera_source_label']+'。\n'
                  '只移动装板侧手臂，头颈、腰腿和夹爪开度保持。\n'
                  '多关节连续平滑运动，最长30cm局部路径，限速24°/s且跟随落后时减缓进度。\n'
                  '开始前复核靶标位置，开始后按锁定坐标执行，不持续追踪靶标。\n'
                  '执行中请勿移动盒子、靶标或机器人底盘；若目标被移动，请立即停止重新规划。\n'
                  '模型未包含桌面、盒子和人员，必须确认整条路径畅通。\n'
                  f'本次停在靶标前{self.standoff.value():g}mm。'
                  + ('0mm可能接触盒子；标定误差可能导致碰撞，请优先先试30mm。\n' if self.standoff.value()==0 else '\n')
                  + '请守在实体急停旁。确认开始？')
            if Q.QMessageBox.question(self.w,'确认验证运动',text,defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
            frame,_=self.w.camera.get(); q,target,_=self.snapshot(frame,True)
            if not self.w.node._hardware_ready:raise ValueError('请先保持式解锁并等待就绪，然后重新规划')
            if time.monotonic()-self.planned_at>60:raise ValueError('规划超过60秒，请重新规划')
            if max(abs(q[n]-v) for n,v in self.source.items())>.5:raise ValueError('机器人姿态已变化，请重新规划')
            if (not self.is_depth and frame['detection']['marker_id']!=self.marker_id) or np.linalg.norm(target[:3,3]-self.target[:3,3])>.008:
                raise ValueError('靶标位置/ID已变化，请重新规划')
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.run_path=self.path/'verification'/stamp
            atomic_json(self.run_path/'plan.json',{'calibration':str(self.path),'plan':self.plan,
                'comparison':self.comparison,
                'target_T_torso_marker':self.target.tolist(),'marker_id':self.marker_id,
                'tcp_offset_mm':self.offset.value(),'tcp_offset_frame':'old_marker_negative_Z',
                'standoff_mm':self.standoff.value(),'external_obstacles_checked':False,
                'target_mode':'locked_before_execution','vision_required_during_motion':False,
                'target_source':'aligned_depth' if self.is_depth else 'aruco',
                'gripper_start_deg':q[f'Joint_{self.data["config"]["arm"].title()}_Gripper'],
                'tcp_reference_gripper_deg':self.data['config']['calibration_gripper_deg'],
                'gripper_opening_required':False,
                'depth_selection':self.depth.result if self.is_depth else None})
            if not cv2.imwrite(str(self.run_path/'before.png'),frame['image']):raise ValueError('起始图像保存失败')
            if self.plan.get('execution')!='continuous_joint_path_v1':raise ValueError('旧版路径不能执行，请重新规划')
            path=JointPath(ARMS[self.data['config']['arm']],self.plan['waypoints'])
            self.started=time.monotonic()
            self.follower=PathFollower(path,self.started);self.settled_since=None
            self.edge={'continuous':True}; self.state='moving'; self.controls(False)
            self.say('按已锁定坐标执行；不追踪靶标、不操作夹爪。不要移动盒子和底盘，遇危险按实体急停。')
        except Exception as exc:
            self.invalidate(); self.say('尚未运动：'+str(exc))

    def cancel(self):
        was_active=self.active
        if self.state=='selecting':self.depth.clear()
        self.state='idle'; self.edge=None
        if self.process:
            proc=self.process; self.process=None; proc.blockSignals(True); proc.kill(); proc.deleteLater()
        if was_active:
            self.invalidate(); self.controls(True)
        return was_active

    def fail(self,reason):
        moving=self.state=='moving'
        if moving and self.run_path:
            try:atomic_json(self.run_path/'result.json',{'status':'stopped','reason':reason,'q':dict(self.w.node._current),
                                                      'comparison':getattr(self,'comparison',None)})
            except OSError:pass
        self.cancel(); self.invalidate(); self.controls(True)
        if moving:self.w.node._lock('精度验证停止：'+reason)
        self.say('精度验证停止：'+reason+'；不会自动继续，需重新规划。')

    def tick(self,frame):
        if self.w.pages.currentIndex()!=3 and not self.active:return
        now=time.monotonic(); self.model.connected=self.w.node._is_feedback_fresh()
        q=dict(self.w.node._current)
        self.model.set_joint_states(q,self.plan['end_q'] if self.plan else q)
        if self.state=='planning':
            if now>self.deadline:self.fail('规划超时，没有执行运动')
        self.depth.tick(frame)
        try:
            # Live detection is for setup only. Motion uses the immutable target
            # and prechecked joint path; occlusion/re-PnP never changes it.
            if self.state=='moving':target=self.target
            else:q,target,goal=self.snapshot(frame)
            p=target[:3,3]*1000
            if self.is_depth:
                r=self.depth.result or {}
                self.pose.setText(f'深度目标 / 躯干坐标 mm：X {p[0]:.1f}  Y {p[1]:.1f}  Z {p[2]:.1f}'
                                  +f' · {r.get("inlier_frames",0)}/7帧一致 · 波动{r.get("spread_mm",0):.1f}mm · 仅位置，不指定朝向')
            else:
                self.show_marker_pose(frame,target,p)
        except Exception as exc:
            self.pose.setText('目标暂不可用：'+str(exc))
        if frame is not None and now-frame['received']<.4:
            if frame['received']!=self.seen:
                from .app import pixmap
                image=frame['image'].copy(); d=frame['detection']
                if not self.is_depth and 'corners' in d:
                    pts=np.array(d['corners'],np.int32); cv2.polylines(image,[pts],True,(60,220,90),2)
                    center=tuple(np.mean(pts,axis=0).astype(int)); cv2.drawMarker(image,center,(0,180,255),cv2.MARKER_CROSS,24,2)
                if self.is_depth and self.depth.pixel:
                    cv2.drawMarker(image,self.depth.pixel,(0,190,255),cv2.MARKER_CROSS,20,1)
                    x,y=self.depth.pixel;cv2.rectangle(image,(x-4,y-4),(x+4,y+4),(0,190,255),1)
                self.video.setPixmap(pixmap(image).scaled(self.video.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation)); self.seen=frame['received']
                self.video.source_shape=image.shape;self.display_frame=frame
        else:
            self.display_frame=None;self.video.source_shape=None
            self.video.setText('RGB暂不可用；正在按已锁定坐标执行' if self.state=='moving' else '实时RGB已过期；不能锁定新目标')
        self.motion_tick(frame,now)

    def show_marker_pose(self,frame,target,p):
            from scipy.spatial.transform import Rotation
            rpy=Rotation.from_matrix(target[:3,:3]).as_euler('xyz',degrees=True)
            shown_id=self.marker_id if self.state=='moving' else frame['detection']['marker_id']
            self.pose.setText(f'ID {shown_id} · 靶标中心 / 躯干坐标 mm：X {p[0]:.1f}  Y {p[1]:.1f}  Z {p[2]:.1f}'
                              f' · RPY {rpy[0]:.1f} / {rpy[1]:.1f} / {rpy[2]:.1f}°'
                              + (' · 已锁定坐标（不持续视觉追踪）' if self.state=='moving' else
                                 f' · 黑框最短边 {frame["detection"].get("edge_px",0):.1f}px'
                                 +('（小靶标，可测试；位置误差可能增大）' if frame['detection'].get('edge_px',48)<48 else '')))

    def motion_tick(self,frame,now):
        if self.state!='moving':return
        try:
            q=self.w.node.snapshot()
            if not self.w.node._hardware_ready:raise ValueError('控制器已锁定：'+self.w.node._last_event)
            if now-self.w.node._feedback_monotonic>.3:raise ValueError('运动关节反馈过期')
            if now-self.started>180:raise ValueError('验证运动超时')
            names=ARMS[self.data['config']['arm']]
            fixed={n:v for n,v in self.source.items() if n not in names}
            check_path_state(q,fixed,fixed,None)
            values=self.follower.step(q,now)
            tol=float(self.w.node.get_parameter('goal_tolerance_deg').value)
            arrived=(self.follower.final and not self.w.node._active_goals and
                     max(abs(q[n]-self.plan['end_q'][n]) for n in names)<=tol+.01)
            if arrived:
                if self.settled_since is None:self.settled_since=now
            else:self.settled_since=None
            if self.follower.end_since is not None and now-self.follower.end_since>15:
                raise ValueError('终点15秒仍未到位')
            if self.settled_since is not None and now-self.settled_since>=.3:
                measured=tcp_position(self.w.fk,self.data['config'],q,np.array(self.plan['tcp']))
                error=float(np.linalg.norm(measured-np.array(self.plan['goal']))*1000)
                result={'status':'completed','q':q,'measured_tcp_torso_m':measured.tolist(),
                        'comparison':getattr(self,'comparison',None),
                        'kinematic_target_error_mm':error,'physical_accuracy_measured':False,
                        'target_mode':'locked_before_execution',
                        'target_source':'aligned_depth' if self.is_depth else 'aruco',
                        'note':'请用尺量夹爪中心与靶标中心的实际差距；FK残差不是独立精度验证'}
                self.w.node._lock('实物验证完成，保持并锁定')
                # End image is optional evidence, never a condition for arrival.
                result['end_image_status']='RGB过期或不可用，未保存终点图像'
                if frame is not None and now-frame['received']<.4:
                    try:
                        result['end_image_status']='saved' if cv2.imwrite(str(self.run_path/'after.png'),frame['image']) else '终点图像写入失败'
                    except (cv2.error,OSError) as exc:result['end_image_status']='终点图像未保存：'+str(exc)
                atomic_json(self.run_path/'result.json',result)
                self.state='idle'; self.edge=None; self.invalidate(); self.controls(True)
                self.say(f'验证运动结束。模型反馈到目标差{error:.1f}mm；请目视/尺量实际误差，图像和记录已保存。不会自动纠偏或抓取。')
                return
            if values is not None:
                # Rate/acceleration/lead are controlled above. Do not add the
                # former 20% progressive limiter and a stop at every sample.
                self.w.node.apply(self.data['config']['arm']+'_arm',values,speed_scale=1.)
                if not self.w.node._active_goals and max(abs(q[n]-values[n]) for n in names)>tol:
                    raise ValueError('控制器未接收目标：'+self.w.node._last_event)
            percent=100*self.follower.progress/self.follower.path.length
            self.status.setText(f'固定坐标验证运动 {percent:.0f}% · 不持续视觉追踪 · 夹爪不开合 · 仅终点停稳')
        except Exception as exc:self.fail(str(exc))
