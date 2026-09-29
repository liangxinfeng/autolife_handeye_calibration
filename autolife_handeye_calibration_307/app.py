"""Native Qt guided workflow; only explicit group Apply actions can move hardware."""
import copy
import datetime
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from types import SimpleNamespace
import cv2
import numpy as np
import yaml
import rclpy
from ament_index_python.packages import get_package_share_directory
from PyQt5 import QtCore, QtGui, QtWidgets as Q
from control_center_joint_control.joint_definitions import GROUPS, JOINT_TO_GROUP
from .bridge import CalibrationBridge
from .camera import Camera
from .core import (Session,Kinematics,TORSO,MOUNT,NECK,BODY,ARMS,detect,sample_guard,
                   compute_report,pose_error,atomic_json)

STYLE='''
QWidget {background:#17212d;color:#e5edf5;font-family:"Noto Sans CJK SC","DejaVu Sans";font-size:14px;}
QPushButton {background:#29445b;border:1px solid #426078;border-radius:7px;padding:9px;}
QPushButton:hover {background:#355b78;} QPushButton:disabled {color:#718096;background:#222e3b;}
QPushButton#capture {background:#167866;font-size:17px;font-weight:bold;padding:14px;}
QPushButton#stop {background:#963c43;}
QLineEdit,QDoubleSpinBox,QSpinBox,QComboBox {background:#223142;padding:5px;border:1px solid #4b6476;border-radius:4px;}
QListWidget,QTableWidget {background:#101a24;border:1px solid #354c61;}
QListWidget::item:selected {background:#365b78;} QHeaderView::section {background:#29445b;padding:7px;}
QTabBar::tab {padding:9px;background:#223142;} QTabBar::tab:selected {background:#356078;}
QLabel#guide {background:#24374b;border-radius:8px;padding:12px;font-size:16px;}
'''

def pixmap(image):
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data,w,h,rgb.strides[0],QtGui.QImage.Format_RGB888).copy())

class Solver(QtCore.QObject):
    done=QtCore.pyqtSignal(object,object)
    def run(self,data):
        try:self.done.emit(compute_report(data),None)
        except Exception as exc:self.done.emit(None,str(exc))

class Window(Q.QMainWindow):
    def __init__(self,node,config,camera=None):
        super().__init__(); self.node=node; self.cfg=config; self.base_cfg=copy.deepcopy(config); self.fk=Kinematics(config['urdf_path'])
        self.camera=camera or Camera(config); self.session=None; self.index=0; self.last_draw=None
        self.busy=False; self.collect=None; self.rows={}; self.current={}; self.result_rows=[]
        self.node.gate=self.motion_gate
        self.changing=False; self.pending_groups=set(); self.last_live_send=0.; self.resetting=False
        from .automation import Automation
        self.automation=Automation(self)
        from .workflow import build
        build(self); self.update_guide()
        from .verification import Verification
        self.verification=Verification(self)
        self.solver=Solver(); self.solver.done.connect(self.solved)
        self.tick_timer=QtCore.QTimer(); self.tick_timer.timeout.connect(self.tick); self.tick_timer.start(60)
        self.ros_timer=QtCore.QTimer(); self.ros_timer.timeout.connect(self.spin_ros); self.ros_timer.start(5)
        for widget in (self.size,self.marker):widget.valueChanged.connect(self.update_config)
        self.arm.currentIndexChanged.connect(self.update_config)
    def message(self,text):
        self.banner.setText(text)

    def goto_page(self,index):
        if self.busy or self.collect or self.automation.active or self.verification.active:return
        if self.pages.currentIndex()==3 and index!=3:self.verification.leave()
        if index!=1:
            self.realtime.setChecked(False)
            if self.node._active_goals:self.stop()
        self.pages.setCurrentIndex(index)
        if index==3:self.verification.enter()
        if index==2:
            from .workflow import render_report
            render_report(self)

    def live_toggled(self,enabled):
        self.pending_groups.clear()
        if enabled:
            if not self.node._hardware_ready or self.busy or self.collect or self.resetting:
                self.realtime.setChecked(False); self.message('请先解锁，等待“就绪”，再勾选实时调整'); return
            for group in ('left_arm','right_arm','neck','grippers'):self.sync_group(group)
            self.message('实时调整已开启：拖动滑块或模型即可低速跟随')
        elif self.node._active_goals:
            self.stop()

    def model_selected(self,name):
        if name not in self.rows:
            self.message('本工具不控制腰腿，请选择手臂或头颈'); self.model.set_selected_joint(''); return
        self.tabs.setCurrentIndex(self.group_indexes[JOINT_TO_GROUP[name]])

    def model_dragged(self,name,value):
        if name not in self.rows:return
        self.model_selected(name); self.rows[name][2].setValue(value)

    def target_changed(self,name,value):
        if self.changing or self.automation.active:return
        group=JOINT_TO_GROUP[name]
        reason=self.motion_gate(group)
        if reason:
            self.message(reason); self.changing=True
            try:self.rows[name][2].setValue(self.current.get(name,value))
            finally:self.changing=False
            return
        if self.realtime.isChecked():self.pending_groups.add(group)

    def live_tick(self):
        if self.realtime.isChecked() and not self.node._hardware_ready:
            self.realtime.setChecked(False); return
        if time.monotonic()-self.last_live_send<.1:return
        self.last_live_send=time.monotonic()
        pending=tuple(self.pending_groups); self.pending_groups.clear()
        for group in pending:
            if self.realtime.isChecked():self.apply_group(group)
        if self.resetting and not self.node._active_goals:
            self.resetting=False; self.reset_button.setEnabled(True)
            self.message('复位流程已结束，请核对模型实测状态：'+self.node._last_event)

    def reset_pose(self):
        if self.automation.active:return
        if self.session:
            self.message('请先结束本轮再复位，避免改变本轮固定头部基准'); return
        if self.busy or self.collect or self.resetting:return
        if not self.node._hardware_ready:self.message('先解锁并等待就绪，复位不会自动使能'); return
        targets=copy.deepcopy(self.cfg['initial_pose_deg'])
        try:q=self.node.snapshot()
        except Exception as exc:self.message('无法准备复位：'+str(exc));return
        jaw=f'Joint_{self.cfg["arm"].title()}_Gripper'
        targets['grippers']=[self.cfg.get('calibration_gripper_deg',240.) if n==jaw else q[n]
                             for n,*_ in GROUPS['grippers']['joints']]
        # Explicit user-confirmed preparation only; fixed throughout sampling.
        self.changing=True
        try:
            for group,values in targets.items():
                for (name,*_),value in zip(GROUPS[group]['joints'],values):self.rows[name][2].setValue(value)
        finally:self.changing=False
        self.model.set_joint_states(self.current,{**self.current,**{n:row[2].value() for n,row in self.rows.items()}})
        if Q.QMessageBox.question(self,'确认回到初始姿态','双臂收拢，头部俯视20°；装板侧夹爪到240°，另一侧开度与腰腿保持。\n准备复位不是路径避障规划。确认板只固定在单片手指，支架、桌面和双臂运动空间无碰撞后继续？',defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
        self.realtime.setChecked(False); self.pending_groups.clear()
        # Unchecking real-time while moving locks output; require a new explicit unlock.
        if not self.node._hardware_ready:self.message('已停止原运动，请重新解锁后再复位'); return
        for group in targets:self.apply_group(group,speed_scale=.5)
        self.resetting=bool(self.node._active_goals); self.reset_button.setEnabled(not self.resetting)

    def eval_select(self,row,col):
        report=self.session.data.get('report') if self.session else None
        if report:
            self.select_sample(report['samples'][row]['slot']); self.goto_page(1)

    def update_config(self):
        if self.session:return
        self.cfg.update(marker_size_mm=self.size.value(),marker_id=self.marker.value(),arm='right' if self.arm.currentIndex()==0 else 'left')
        self.camera.configure(self.cfg)

    def show_intrinsics(self):
        frame,error=self.camera.get()
        if frame is None:self.message(error or '相机尚未就绪'); return
        info=frame['intrinsics']
        Q.QMessageBox.information(self,'当前工厂相机参数',
            f'原始RGB：{frame["image"].shape[1]}×{frame["image"].shape[0]}\n'
            +json.dumps(info,ensure_ascii=False,indent=2,default=str)
            +'\n这些参数来自当前头部相机共享内存。当前未对RGB做缩放或旋转。'
            '\n码尺寸需用尺量黑框外边。固定板、光照良好后开始。'
            '\n若更换了相机、图像模式或内参，必须重新核对并新建实验；不要把内参缺失当成零畸变。')

    def motion_gate(self,group):
        verification=getattr(self,'verification',None)
        if verification and verification.active:
            if verification.state!='moving' or verification.edge is None:return '精度验证规划/等待阶段禁止运动'
            return '' if group==verification.data['config']['arm']+'_arm' else '精度验证不开放夹爪、头颈或另一手臂'
        if self.automation.active:
            if self.automation.state not in ('moving','next_edge'):return '自动流程当前阶段禁止运动'
            edge=self.automation.edge
            allowed=JOINT_TO_GROUP[edge['joint']] if edge else self.cfg['arm']+'_arm'
            if group!=allowed:return '自动流程只允许当前已检查关节运动'
        if self.busy or self.collect is not None:return '计算/采样期间不允许调整'
        if group not in ('left_arm','right_arm','neck','grippers'):return '不开放腰腿运动'
        if self.session:
            phase=self.session.data['steps'][self.index]['phase']
            if group=='grippers':return '本轮已锁定夹爪开度。如需开合，请结束本轮并新建实验'
            if group in ('left_arm','right_arm') and group!=self.cfg['arm']+'_arm':return '只允许调整本轮安装标定板的一侧手臂'
            if group=='neck' and (phase!='head' or not self.session.data.get('fixed_solution')):
                return '头部固定采样期间不允许转头；完成手臂训练与验证并计算后进入转头验证'
        return ''

    def unlock(self):
        if self.busy or self.collect or self.automation.active:return
        answer=Q.QMessageBox.question(self,'真机控制确认','确认周围无人、板和支架牢固、另一套遥操已退出？\n将保持当前姿态准备手动控制，不执行系统复位。\n本工具不提供路径碰撞规划，请小幅单关节调整。',Q.QMessageBox.Yes|Q.QMessageBox.No,Q.QMessageBox.No)
        if answer==Q.QMessageBox.Yes:
            response=self.node.unlock(); self.message(response.message)

    def stop(self):
        verification=getattr(self,'verification',None)
        if verification and verification.active:verification.fail('用户停止')
        self.automation.cancel()
        self.realtime.blockSignals(True); self.realtime.setChecked(False); self.realtime.blockSignals(False)
        self.pending_groups.clear(); self.resetting=False; self.reset_button.setEnabled(True)
        self.collect=None; self.node.sampling=False; self.node._lock('用户停止标定调整')
        self.message('已停止发送后续目标；不是硬件急停。如有危险请按实体急停。')

    def sync_group(self,group):
        self.changing=True
        try:
            for name,*_ in GROUPS[group]['joints']:
                if name in self.current:self.rows[name][2].setValue(self.current[name])
        finally:self.changing=False

    def apply_group(self,group,speed_scale=None):
        if self.automation.active:return
        try:
            reason=self.motion_gate(group)
            if reason:raise ValueError(reason)
            q=self.node.snapshot()
            if not self.node._hardware_ready:raise ValueError('请先解锁并等待硬件准备完成')
            target={n:self.rows[n][2].value() for n,*_ in GROUPS[group]['joints']}
            if speed_scale is None:self.node.apply(group,target)
            else:self.node.apply(group,target,speed_scale=speed_scale)
            self.message(self.node._last_event)
        except Exception as exc:self.message(str(exc))

    def set_session(self,s):
        self.session=s; self.cfg=copy.deepcopy(s.data['config']); self.camera.configure(self.cfg)
        self.arm.setCurrentIndex(0 if self.cfg['arm']=='right' else 1); self.size.setValue(self.cfg['marker_size_mm']); self.marker.setValue(self.cfg['marker_id'])
        for w in (self.arm,self.size,self.marker):w.setEnabled(False)
        self.board_check.setChecked(True); self.intr_check.setChecked(True)
        self.index=next((i for i in range(len(s.data['steps'])) if str(i) not in s.data['samples']),0)
        self.refresh_samples(); self.update_guide(); self.pages.setCurrentIndex(1); self.message('实验已保存至 '+str(s.path))

    def new_session(self):
        if self.automation.active:self.message('请先停止自动采集');return
        if self.busy or self.collect:
            self.message('正在计算或记录，请等待当前操作完成'); return
        if self.session:
            if Q.QMessageBox.question(self,'结束本轮','当前数据会保留。结束后可调整头部、夹爪、板尺寸，再新建一轮？',defaultButton=Q.QMessageBox.No)!=Q.QMessageBox.Yes:return
            self.stop(); self.session=None
            for w in (self.arm,self.size,self.marker):w.setEnabled(True)
            self.samples.clear(); self.update_guide(); self.goto_page(0); self.message('本轮已保存。可重新调姿，再开始新一轮。'); return
        try:
            self.update_config()
            if not self.board_check.isChecked() or not self.intr_check.isChecked():
                self.goto_page(0)
                raise ValueError('已返回①准备页。请勾选“板已固定，本轮开度不变”和“已核对相机参数”，再进入调姿开始采集。')
            if self.node._active_goals or getattr(self.node,'_hardware_prepare',None):
                raise ValueError('关节仍在调整或使能中。请等待到位，或点“停止/锁定”，停稳后再开始。')
            q=self.node.snapshot(); frame,error=self.camera.get()
            if frame is None or time.monotonic()-frame['received']>.5:raise ValueError(error or 'RGB画面过期')
            cfg=copy.deepcopy(self.cfg)
            found=frame['detection'].get('detected_ids',[])
            if cfg['marker_id']<0:
                if len(found)!=1:raise ValueError('请在画面内完整露出一个标定码，自动识别唯一ID后再开始。')
                cfg['marker_id']=int(found[0])
            elif found.count(cfg['marker_id'])!=1:raise ValueError('当前画面没有唯一的所选ID，请核对标定码。')
            # Starting a round needs a board identity, not a perfect first PnP.
            # Full corner/pose quality remains mandatory for every saved sample.
            reason=sample_guard(q,q,'train',cfg,self.node.history,time.monotonic())
            if reason:raise ValueError(reason)
            directory=Path.home()/'ros2_ws/src/autolife_handeye_calibration_307/sessions'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.stop(); self.set_session(Session(directory,cfg,q,self.fk.sha))
            self.start_status.setText('本轮已开始 · ID '+str(cfg['marker_id'])+' · 请按第1张提示采集')
        except Exception as exc:
            text=str(exc); self.start_status.setText('尚未开始：'+text); self.message(text)
            self.node._event('开始采集被阻止：'+text,'warn')
            Q.QMessageBox.information(self,'还差一步即可开始',text)

    def load_session(self):
        if self.busy or self.collect or self.automation.active:return
        directory=Q.QFileDialog.getExistingDirectory(self,'选择包含session.json的实验目录',str(Path.home()/'ros2_ws/src/autolife_handeye_calibration_307/sessions'))
        if not directory:return
        try:
            s=Session.load(directory)
            if s.data['urdf_sha256']!=self.fk.sha or s.data['config']['robot_id']!=283:raise ValueError('机器人或URDF与此实验不一致，不能继续混合采集')
            self.stop(); self.set_session(s)
        except Exception as exc:self.message(str(exc))

    def refresh_samples(self):
        if not self.session:return
        self.samples.blockSignals(True); self.samples.clear()
        report=self.session.data.get('report') or {}
        bad={r['slot'] for r in report.get('samples',[]) if r['replace_suggested']}
        for i,s in enumerate(self.session.data['steps']):
            sample=self.session.data['samples'].get(str(i)); item=Q.QListWidgetItem(f'{i+1:02d} '+{'train':'训练','validation':'验证','head':'转头'}[s['phase']])
            item.setToolTip(s['hint'])
            if sample:
                item.setIcon(QtGui.QIcon(str(self.session.path/sample['image'])))
                item.setText(item.text()+ (' · 排除' if not sample.get('included',True) else ' · 重拍' if i in bad else ' ✓'))
            if i in bad:item.setForeground(QtGui.QColor('#ffac7b'))
            self.samples.addItem(item)
        self.samples.setCurrentRow(self.index); self.samples.blockSignals(False)

    def select_sample(self,index):
        if not self.session or self.collect or self.busy or self.automation.active or index<0 or index>=len(self.session.data['steps']):return
        self.index=index; self.samples.blockSignals(True); self.samples.setCurrentRow(index); self.samples.blockSignals(False); self.update_guide()

    def update_guide(self):
        self.begin_button.setVisible(self.session is None); self.capture.setVisible(self.session is not None)
        self.start_status.setVisible(self.session is None)
        if not self.session:
            self.guide.setText('调好手臂、头部和夹爪，让码完整可见。\n准备好后，点击下方开始采集。'); return
        step=self.session.data['steps'][self.index]
        phase={'train':'固定头部训练','validation':'独立验证（不参与拟合）','head':'转头跨姿态验证'}[step['phase']]
        self.guide.setText(f'{self.index+1} / {len(self.session.data["steps"])} · {phase}\n{step["hint"]}\n'+('自动停稳、检查并记录。' if self.automation.active else '停稳后点记录。'))
        self.capture.setText(('替换当前样本' if str(self.index) in self.session.data['samples'] else '记录当前图像')+' → 下一步')

    def validate_frame(self,frame):
        if frame is None or time.monotonic()-frame['received']>.4:raise ValueError('RGB过期，请等待相机恢复')
        if not frame['detection']['ok']:raise ValueError(frame['detection']['reason'])
        if any(frame['config'][k]!=self.cfg[k] for k in ('marker_id','marker_size_mm','arm')):raise ValueError('等待检测器应用本轮配置')
        q=self.node.snapshot(); phase=self.session.data['steps'][self.index]['phase']
        if time.monotonic()-self.node._feedback_monotonic>self.cfg['maximum_feedback_age_sec']:raise ValueError('关节反馈过期')
        error=sample_guard(q,self.session.data['reference'],phase,self.cfg,self.node.history,time.monotonic())
        if error:raise ValueError(error)
        for s in self.session.data['samples'].values():
            if s['image_shape']!=list(frame['image'].shape) or not np.allclose(s['K'],frame['K'],rtol=0,atol=1e-5) or len(s['D'])!=len(frame['D']) or not np.allclose(s['D'],frame['D'],atol=1e-7):
                raise ValueError('相机分辨率或内参变化，请新建一轮，不能混合计算')
            break
        return q

    def start_capture(self):
        if not self.session:self.message('请先新建实验'); return
        if self.busy or self.collect:return
        try:
            frame,_=self.camera.get(); self.validate_frame(frame)
            if self.node._active_goals:raise ValueError('仍有运动目标未完成，请等待到位，或点击停止调整后重新解锁')
            self.pending_groups.clear()
            self.collect={'started':time.monotonic(),'frames':[],'seen':set()}; self.node.sampling=True
            self.message('正在采集8个独立帧，保持不动；通过质量检查后才自动进入下一步。')
        except Exception as exc:self.message(str(exc))

    def capture_tick(self,frame):
        try:
            if time.monotonic()-self.collect['started']>4:raise ValueError('采样超时，未获得足够新帧')
            q=self.validate_frame(frame); ident=(frame['frame_id'],frame['received'])
            if ident in self.collect['seen']:return
            self.collect['seen'].add(ident); self.collect['frames'].append((frame,q))
            frames=self.collect['frames']
            if len(frames)<8:return
            first=np.array(frames[0][0]['detection']['corners'])
            if max(float(np.max(np.linalg.norm(np.array(f['detection']['corners'])-first,axis=1))) for f,_ in frames)>1.5:
                raise ValueError('图像角点仍在移动，请停稳后重拍')
            # Record an actual frame, never an averaged synthetic picture/pose pair.
            frame,q=min(frames,key=lambda fq:fq[0]['detection']['reprojection_px'])
            qfk=dict(q)
            for n,offset in zip(NECK,self.cfg['head_offsets_deg']):qfk[n]+=offset
            G=self.fk.fk(TORSO,f'Link_{self.cfg["arm"].title()}_Wrist_Lower_to_Gripper',qfk)
            H=self.fk.fk(TORSO,MOUNT,qfk)
            phase=self.session.data['steps'][self.index]['phase']
            for s in self.session.data['samples'].values():
                if s['slot']==self.index or not s.get('included',True):continue
                pe,ae=pose_error(G,np.array(s['G'])); _,he=pose_error(H,np.array(s['H']))
                if pe<15 and ae<5 and (phase!='head' or he<3):raise ValueError(f'与第{s["slot"]+1}张姿态太相似，请换位置或倾斜角度')
            sample={**frame['detection'],'G':G.tolist(),'H':H.tolist(),'q_deg':q,'K':frame['K'].tolist(),'D':frame['D'].tolist(),
                    'frame_id':frame['frame_id'],'image_shape':list(frame['image'].shape),'time':time.time(),
                    'image_receive_monotonic':frame['received'],'time_source':frame['time_source'],
                    'batch_frame_ids':[f['frame_id'] for f,_ in frames],'included':True}
            self.index=self.session.record(self.index,frame['image'],sample)
            self.collect=None; self.node.sampling=False; self.refresh_samples(); self.update_guide()
            if self.automation.active:
                self.automation.captured();return
            self.message('已记录并保存。请按新提示手动调整下一姿态；没有自动运动。')
            if (self.session.data['steps'][self.index]['phase']=='head' and not self.session.data.get('fixed_solution')) or len(self.session.data['samples'])==len(self.session.data['steps']):
                self.goto_page(2); self.message('本阶段采集完成，请计算并评估；通过后再做转头验证。')
        except Exception as exc:
            self.collect=None; self.node.sampling=False; self.message('本张未记录：'+str(exc))
            if self.automation.active:self.automation.capture_failed(str(exc))

    def solve(self):
        if not self.session or self.busy or self.collect:return
        if self.automation.active and self.automation.state!='solving':return
        if not self.automation.active:self.node._lock('计算标定，已锁定手动调整')
        self.busy=True
        self.message('正在计算：使用训练集求外参，独立验证集评估；不会应用到抓取系统。')
        threading.Thread(target=self.solver.run,args=(copy.deepcopy(self.session.data),),daemon=True).start()

    def solved(self,report,error):
        self.busy=False
        if error:
            if self.automation.active:self.automation.solved(None,error)
            else:self.message('无法可靠计算：'+error)
            return
        try:
            self.session.data['report']=report
            self.session.data['fixed_solution']={'T_mount_camera':report['T_mount_camera']}
            self.session.save()
            atomic_json(self.session.path/'calibration_candidate.json',{'robot_id':307,'urdf_sha256':self.fk.sha,
                'config':self.cfg,'report':report,'applied_to_robot':False})
            self.message(report['status']); self.refresh_samples(); self.goto_page(2)
            if self.automation.active:self.automation.solved(report,None)
        except Exception as exc:
            if self.automation.active:self.automation.fail('报告保存失败：'+str(exc))
            else:self.message('报告保存失败：'+str(exc))

    def show_report(self):
        if not self.session or not self.session.data.get('report'):self.message('请先采集训练样本并计算'); return
        report=self.session.data['report']; dialog=Q.QDialog(self); dialog.setWindowTitle('本轮标定质量 · 双击样本行去重拍'); dialog.resize(1000,700)
        layout=Q.QVBoxLayout(dialog); summary=Q.QLabel(report['status']+'\n转头验证：'+('通过' if report['moving_head_pass'] else '未通过/不足')+
            '\n这是内部一致性验证，不是实际抓取精度保证；尚未标定TCP和深度。'); summary.setWordWrap(True); layout.addWidget(summary)
        stats=Q.QPlainTextEdit(json.dumps({k:report[k] for k in ['training','validation','head','method_agreement_mm_deg']},ensure_ascii=False,indent=2)); stats.setReadOnly(True); stats.setMaximumHeight(185); layout.addWidget(stats)
        table=Q.QTableWidget(len(report['samples']),6); table.setHorizontalHeaderLabels(['样本','类型','位置误差 mm','角度误差 °','像素误差','建议'])
        for row,e in enumerate(report['samples']):
            for col,value in enumerate([e['slot']+1,e['phase'],f'{e["position_mm"]:.2f}',f'{e["angle_deg"]:.2f}',f'{e["reprojection_px"]:.2f}','建议重拍' if e['replace_suggested'] else '保留']):
                item=Q.QTableWidgetItem(str(value)); item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if e['replace_suggested']:item.setForeground(QtGui.QColor('#ffac7b'))
                table.setItem(row,col,item)
        table.horizontalHeader().setSectionResizeMode(Q.QHeaderView.Stretch); layout.addWidget(table)
        def select(row,col):self.select_sample(report['samples'][row]['slot']); dialog.accept()
        table.cellDoubleClicked.connect(select); layout.addWidget(Q.QLabel('替换会保留旧图并使旧报告失效；不要仅删除误差大的验证样本来“凑通过”。'))
        close=Q.QPushButton('返回实验'); close.clicked.connect(dialog.accept); layout.addWidget(close); dialog.exec_()

    def show_photo(self):
        if not self.session:return
        s=self.session.data['samples'].get(str(self.index))
        if not s:return
        d=Q.QDialog(self); d.setWindowTitle(f'第{self.index+1}张原始标定图（非实时）'); l=Q.QVBoxLayout(d); label=Q.QLabel()
        label.setPixmap(QtGui.QPixmap(str(self.session.path/s['image'])).scaled(950,650,QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation)); l.addWidget(label); d.exec_()

    def exclude(self):
        if not self.session or self.busy or self.collect or self.automation.active:return
        s=self.session.data['samples'].get(str(self.index))
        if not s:return
        reason,ok=Q.QInputDialog.getText(self,'样本排除/恢复','请注明原因（例如板松动/遮挡）；不能靠排除验证失败样本掩盖误差：')
        if not ok or not reason.strip():return
        s['included']=not s.get('included',True); s['exclusion_note']=reason
        self.session.data['report']=None
        if s['phase']!='head':self.session.data.pop('fixed_solution',None)
        self.session.save(); self.refresh_samples()

    def spin_ros(self):
        started=time.monotonic()
        for _ in range(20):
            rclpy.spin_once(self.node,timeout_sec=0)
            if time.monotonic()-started>.004:break

    def tick(self):
        try:
            self.current=dict(self.node._current)
            self.live_tick()
            for n,(lab,slider,box,initialized) in list(self.rows.items()):
                value=self.current.get(n)
                lab.setText('--' if value is None else f'实测 {value:.1f}°')
                if value is not None and not initialized:
                    self.changing=True
                    try:box.setValue(value); self.rows[n]=(lab,slider,box,True)
                    finally:self.changing=False
            self.model.connected=self.node._is_feedback_fresh()
            if self.pages.currentIndex()==1:
                self.model.set_joint_states(self.current,{**self.current,**{n:r[2].value() for n,r in self.rows.items()}})
            self.control_status.setText(('反馈在线' if self.node._is_feedback_fresh() else '反馈断开')+' · '+
                ('已解锁就绪' if self.node._hardware_ready else '准备中' if self.node._hardware_enabled else '只读锁定')+'\n'+self.node._last_event)
            frame,error=self.camera.get()
            self.verification.tick(frame)
            fresh=frame is not None and time.monotonic()-frame['received']<.5
            if fresh and frame['received']!=self.last_draw:
                image=frame['image'].copy(); d=frame['detection']
                self.id_status.setText('识别ID：'+str(d.get('marker_id',d.get('detected_ids','未识别')))+' · '+d['reason'])
                if 'corners' in d:cv2.polylines(image,[np.array(d['corners'],np.int32)],True,(60,220,90) if d['ok'] else (0,160,255),2)
                self.video.setPixmap(pixmap(image).scaled(self.video.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation)); self.last_draw=frame['received']
                self.prep_video.setPixmap(pixmap(image).scaled(self.prep_video.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
                self.quality.setText(f'头部RGB {image.shape[1]}×{image.shape[0]} · '+d['reason']+
                    (f' · 最短边 {d["edge_px"]:.0f}px · 角点误差 {d["reprojection_px"]:.2f}px' if 'edge_px' in d else ''))
            if not fresh:
                self.video.setText('RGB已过期/未连接\n'+(error or '请检查系统视觉服务'))
                self.prep_video.setText('RGB已过期，请检查相机'); self.quality.setText('禁止用过期画面采样')
            self.capture.setEnabled(bool(self.session and fresh and not self.busy and not self.collect and not self.automation.active))
            self.automation.tick(frame)
            if self.collect:self.capture_tick(frame)
        except Exception as exc:
            if self.automation.active:self.automation.fail('状态检查：'+str(exc))
            elif self.verification.active:self.verification.fail('状态检查：'+str(exc))
            else:self.message('状态检查：'+str(exc))

    def closeEvent(self,event):
        if self.busy:
            self.message('请等待计算结束再关闭，避免丢失报告'); event.ignore(); return
        self.stop(); self.tick_timer.stop(); self.ros_timer.stop(); self.camera.close(); event.accept()

def main():
    rclpy.init(); app=Q.QApplication(sys.argv); app.setStyleSheet(STYLE)
    share=Path(get_package_share_directory('autolife_handeye_calibration_307'))
    config=yaml.safe_load((share/'config/experiment.yaml').read_text())
    node=None
    try:
        node=CalibrationBridge(); window=Window(node,config); window.show()
        signal.signal(signal.SIGINT,lambda *_:window.close()); signal.signal(signal.SIGTERM,lambda *_:window.close())
        app.exec_()
    except Exception as exc:
        Q.QMessageBox.critical(None,'标定工具启动失败',str(exc)+'\n请先退出控制中心的全身关节模块，避免两套桥接冲突。')
        traceback.print_exc()
    finally:
        if node is not None:node._lock('标定程序退出'); node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
