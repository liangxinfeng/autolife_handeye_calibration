"""Step-based native UI. This module does not publish robot commands."""
from pathlib import Path
from PyQt5 import QtCore, QtWidgets as Q
from ament_index_python.packages import get_package_share_directory
from control_center_joint_control.joint_definitions import GROUPS
from .model_view import CalibrationModel


def button(layout, text, fn):
    b=Q.QPushButton(text); b.clicked.connect(fn); layout.addWidget(b); return b


def build(w):
    w.setWindowTitle('云蝶 · 手眼标定'); w.resize(1500,940)
    root=Q.QWidget(); w.setCentralWidget(root); main=Q.QVBoxLayout(root)
    header=Q.QHBoxLayout(); title=Q.QLabel('300  手眼标定'); title.setStyleSheet('font-size:24px;font-weight:bold')
    header.addWidget(title); header.addStretch()
    for i,text in enumerate(('① 准备','② 调姿与采集','③ 评估与重拍','④ 精度验证')):
        button(header,text,lambda _,n=i:w.goto_page(n))
    main.addLayout(header)
    w.banner=Q.QLabel('先确认标定板尺寸，再进入调姿。'); w.banner.setWordWrap(True); main.addWidget(w.banner)
    w.pages=Q.QStackedWidget(); main.addWidget(w.pages,1)

    prep=Q.QWidget(); p=Q.QHBoxLayout(prep); w.prep_video=Q.QLabel('等待RGB'); w.prep_video.setAlignment(QtCore.Qt.AlignCenter)
    w.prep_video.setMinimumSize(400,300); w.prep_video.setSizePolicy(Q.QSizePolicy.Ignored,Q.QSizePolicy.Expanding)
    w.prep_video.setStyleSheet('background:#050b10'); p.addWidget(w.prep_video,3)
    card=Q.QWidget(); c=Q.QVBoxLayout(card); c.addStretch()
    intro=Q.QLabel('开始前，只确认三件事\n\n1. 板固定在单片手指上\n2. 量黑框外边，不含白边\n3. 退出VR和其他关节控制界面')
    intro.setStyleSheet('font-size:19px'); c.addWidget(intro)
    form=Q.QFormLayout(); w.arm=Q.QComboBox(); w.arm.addItems(['右臂','左臂'])
    w.arm.setCurrentIndex(0 if w.cfg['arm']=='right' else 1)
    w.size=Q.QDoubleSpinBox(); w.size.setRange(20,300); w.size.setValue(w.cfg['marker_size_mm']); w.size.setSuffix(' mm')
    w.marker=Q.QSpinBox(); w.marker.setRange(-1,249); w.marker.setSpecialValueText('自动识别'); w.marker.setValue(w.cfg['marker_id'])
    form.addRow('装板的手臂',w.arm); form.addRow('黑框边长',w.size); form.addRow('6×6 / 250 · ID',w.marker); c.addLayout(form)
    w.id_status=Q.QLabel('将码放入RGB画面，自动识别ID'); w.id_status.setWordWrap(True); c.addWidget(w.id_status)
    w.board_check=Q.QCheckBox('板已固定，本轮夹爪开度保持不变'); c.addWidget(w.board_check)
    w.intr_check=Q.QCheckBox('已核对相机参数'); c.addWidget(w.intr_check)
    button(c,'查看相机参数',w.show_intrinsics)
    button(c,'进入调姿 →',lambda:w.goto_page(1))
    button(c,'继续已有实验',w.load_session); c.addStretch(); p.addWidget(card,2); w.pages.addWidget(prep)

    work=Q.QWidget(); wl=Q.QVBoxLayout(work)
    bar=Q.QHBoxLayout()
    w.unlock_button=button(bar,'解锁手动调整（保持当前）',w.unlock)
    w.realtime=Q.QCheckBox('实时调整'); w.realtime.toggled.connect(w.live_toggled); bar.addWidget(w.realtime)
    bar.addWidget(Q.QLabel('速度'))
    w.speed=Q.QSpinBox(); w.speed.setRange(10,100); w.speed.setSingleStep(10); w.speed.setValue(100); w.speed.setSuffix('%')
    w.speed.setToolTip('100%与控制中心普通关节调整一致；改变速度对下一次发送生效')
    w.speed.valueChanged.connect(lambda v:setattr(w.node,'manual_speed_scale',v/100)); bar.addWidget(w.speed)
    w.reset_button=button(bar,'回到VR初始姿态…',w.reset_pose)
    stop=button(bar,'停止 / 锁定',w.stop); stop.setObjectName('stop')
    w.control_status=Q.QLabel('反馈连接中'); w.control_status.setWordWrap(True); bar.addWidget(w.control_status,1); wl.addLayout(bar)
    split=Q.QSplitter(); wl.addWidget(split,1)
    model=Q.QWidget(); ml=Q.QVBoxLayout(model); ml.setContentsMargins(0,0,4,0)
    toolbar=Q.QHBoxLayout(); button(toolbar,'观察',lambda:w.model.set_interaction_mode('camera'))
    button(toolbar,'点模型调关节',lambda:w.model.set_interaction_mode('joint'))
    button(toolbar,'正视',lambda:w.model.front_camera()); button(toolbar,'重置视角',lambda:w.model.reset_camera()); ml.addLayout(toolbar)
    urdf=Path(get_package_share_directory('control_center_joint_control'))/'urdf/robot_v2_2.urdf'
    w.model=CalibrationModel(str(urdf)); w.model.setMinimumSize(350,300)
    w.model.joint_selected.connect(w.model_selected); w.model.joint_target_dragged.connect(w.model_dragged); ml.addWidget(w.model,1)
    ml.addWidget(Q.QLabel('实体色＝实测 · 蓝色＝目标；夹爪开合仅示意\n拖空白转视角 / 滚轮缩放；选关节后看旋转方向箭头'))
    split.addWidget(model)
    visual=Q.QWidget(); vl=Q.QVBoxLayout(visual); vl.setContentsMargins(3,0,3,0)
    w.video=Q.QLabel('等待头部RGB'); w.video.setAlignment(QtCore.Qt.AlignCenter); w.video.setMinimumSize(320,250)
    w.video.setSizePolicy(Q.QSizePolicy.Ignored,Q.QSizePolicy.Expanding); w.video.setStyleSheet('background:#050b10'); vl.addWidget(w.video,1)
    w.quality=Q.QLabel(); w.quality.setWordWrap(True); vl.addWidget(w.quality)
    w.guide=Q.QLabel(); w.guide.setObjectName('guide'); w.guide.setWordWrap(True); vl.addWidget(w.guide)
    w.begin_button=button(vl,'姿态准备好了 · 开始本轮采集',w.new_session)
    w.auto_button=button(vl,'全自动：复位 → 24张采集 → 输出结果',w.automation.start)
    w.start_status=Q.QLabel('开始后锁定本轮ID、头部位置与夹爪开度'); w.start_status.setWordWrap(True); vl.addWidget(w.start_status)
    w.capture=button(vl,'记录当前图像 → 下一步',w.start_capture); w.capture.setObjectName('capture')
    split.addWidget(visual)
    controls=Q.QWidget(); controls.setMinimumWidth(270); controls.setMaximumWidth(330); cl=Q.QVBoxLayout(controls)
    w.tabs=Q.QTabWidget(); cl.addWidget(w.tabs,1); w.group_indexes={}
    for group in ('right_arm','left_arm','neck','grippers'):
        page=Q.QWidget(); pl=Q.QVBoxLayout(page); scroll=Q.QScrollArea(); scroll.setWidgetResizable(True)
        contents=Q.QWidget(); rows=Q.QVBoxLayout(contents)
        for name,label,low,high in GROUPS[group]['joints']:
            row=Q.QVBoxLayout(); top=Q.QHBoxLayout(); top.addWidget(Q.QLabel(label)); cur=Q.QLabel('--'); top.addWidget(cur); row.addLayout(top)
            line=Q.QHBoxLayout(); slider=Q.QSlider(QtCore.Qt.Horizontal); slider.setRange(round(low*10),round(high*10))
            box=Q.QDoubleSpinBox(); box.setRange(low,high); box.setDecimals(1); box.setSingleStep(.5); box.setSuffix('°'); box.setKeyboardTracking(False)
            slider.valueChanged.connect(lambda v,b=box:b.setValue(v/10))
            box.valueChanged.connect(lambda v,s=slider:s.setValue(round(v*10)))
            box.valueChanged.connect(lambda v,n=name:w.target_changed(n,v))
            line.addWidget(slider,1); line.addWidget(box); row.addLayout(line); rows.addLayout(row)
            w.rows[name]=(cur,slider,box,False)
        rows.addStretch(); scroll.setWidget(contents); pl.addWidget(scroll)
        if group=='grippers':
            label=Q.QLabel('10°＝全开 · 360°＝全闭\n实体模型跟随实测值，蓝色是目标'); label.setWordWrap(True); pl.addWidget(label)
            for side in ('Left','Right'):
                line=Q.QHBoxLayout(); name=f'Joint_{side}_Gripper'; label='左' if side=='Left' else '右'
                button(line,label+'全开',lambda _,n=name:w.rows[n][2].setValue(10.))
                button(line,label+'全闭',lambda _,n=name:w.rows[n][2].setValue(360.)); pl.addLayout(line)
        button(pl,'同步当前实测',lambda _,g=group:w.sync_group(g))
        button(pl,'发送预览目标',lambda _,g=group:w.apply_group(g))
        w.group_indexes[group]=w.tabs.addTab(page,GROUPS[group]['label'])
    cl.addWidget(Q.QLabel('未勾实时：只预览，需点发送\n已勾实时：拖动即低速跟随\n松开滑块仍会完成当前目标'))
    split.addWidget(controls); split.setSizes([460,600,300])
    w.samples=Q.QListWidget(); w.samples.setViewMode(Q.QListView.IconMode); w.samples.setResizeMode(Q.QListView.Adjust)
    w.samples.setIconSize(QtCore.QSize(64,48)); w.samples.setGridSize(QtCore.QSize(104,76)); w.samples.setMaximumHeight(100)
    w.samples.currentRowChanged.connect(w.select_sample); wl.addWidget(w.samples)
    foot=Q.QHBoxLayout(); button(foot,'上一张',lambda:w.select_sample(max(0,w.index-1)))
    button(foot,'下一张',lambda:w.select_sample(w.index+1)); button(foot,'查看原图',w.show_photo)
    button(foot,'结束本轮 / 重新准备',w.new_session); button(foot,'进入评估 →',lambda:w.goto_page(2)); wl.addLayout(foot)
    w.pages.addWidget(work)

    evaluation=Q.QWidget(); e=Q.QVBoxLayout(evaluation)
    w.eval_summary=Q.QLabel('完成固定头部采集后，计算并检查误差。'); w.eval_summary.setWordWrap(True); w.eval_summary.setStyleSheet('font-size:21px'); e.addWidget(w.eval_summary)
    actions=Q.QHBoxLayout(); button(actions,'计算并评估',w.solve); button(actions,'详细报告',w.show_report)
    button(actions,'返回采集 / 重拍',lambda:w.goto_page(1)); button(actions,'排除/恢复选中样本',w.exclude); e.addLayout(actions)
    w.eval_table=Q.QTableWidget(0,6); w.eval_table.setHorizontalHeaderLabels(['样本','用途','位置误差 mm','角度误差 °','像素误差','处理建议'])
    w.eval_table.horizontalHeader().setSectionResizeMode(Q.QHeaderView.Stretch); w.eval_table.cellDoubleClicked.connect(w.eval_select); e.addWidget(w.eval_table,1)
    e.addWidget(Q.QLabel('双击异常行回到该张重拍。原图保留；结果只保存，不自动应用到抓取。')); w.pages.addWidget(evaluation)


def render_report(w):
    report=(w.session.data.get('report') if w.session else None) or {}
    if not report:
        n=len(w.session.data['samples']) if w.session else 0
        total=len(w.session.data['steps']) if w.session else 24
        w.eval_summary.setText(f'已采集 {n} / {total} 张。先计算，再按提示重拍。'); w.eval_table.setRowCount(0); return
    count=len(w.session.data['samples']);total=len(w.session.data['steps'])
    head='通过' if report['moving_head_pass'] else ('精度未达推荐值' if report['head']['count']>=w.session.data['config'].get('auto_head_validation_samples',4) else '尚未完成')
    w.eval_summary.setText(f'已采集 {count} / {total} 张 · '+report['status']+'\n转头验证：'+head+'\n结果文件：'+str(w.session.path/'calibration_candidate.json'))
    w.eval_table.setRowCount(len(report['samples']))
    for row,s in enumerate(report['samples']):
        for col,value in enumerate([s['slot']+1,{'train':'训练','validation':'独立验证','head':'转头'}[s['phase']],
            f"{s['position_mm']:.2f}",f"{s['angle_deg']:.2f}",f"{s['reprojection_px']:.2f}",'双击重拍' if s['replace_suggested'] else '保留']):
            item=Q.QTableWidgetItem(str(value)); item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable); w.eval_table.setItem(row,col,item)
