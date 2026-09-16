# 云蝶306 · 手眼标定与精度验证

306机器人 `~/handeye306` 对应ROS 2功能包，提供可视化调姿、ArUco自动采集、标定计算、样本复查和夹爪触点验证。源码取自306当前使用版本（2026-09-16）。

## 功能

- 实时头部RGB画面与机器人三维模型，调整双臂、头颈和夹爪。
- 自动识别ArUco ID，按引导手动采集或自动复位、规划视野内姿态并采集。
- 根据图像和关节反馈计算外参，保存结果、质量报告及可替换的采集样本。
- 精度验证支持ArUco目标或RGB图像手动选点，利用深度定位并规划夹爪触点运动。
- 可切换标定外参与URDF名义参数，进行同场景对比。

## 环境与安装

Ubuntu 24.04、ROS 2 Jazzy、Python 3.12。需要系统ROS/Qt、OpenGL、OpenCV ArUco、NumPy、SciPy、YAML；自动规划还需要 `robot_env` 的Pinocchio及碰撞库。

依赖 [控制中心](https://github.com/liangxinfeng/autolife_robot_control_center)中的 `control_center_joint_control` 和 [遥操V4](https://github.com/liangxinfeng/openarmx_teleop_vr_306_v4) 的规划/碰撞模块。厂商SDK、完整URDF、相机共享内存和机械臂服务须在机器人上另行安装，本仓库不分发。

```bash
cd ~/ros2_ws/src
git clone https://github.com/liangxinfeng/autolife_handeye_calibration_306.git
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select autolife_handeye_calibration_306 --symlink-install
source install/setup.bash
ros2 launch autolife_handeye_calibration_306 calibration.launch.py
```

已有306环境直接运行 `~/handeye306`。首次安装可在检查路径后，将 `scripts/handeye306` 复制到主目录并添加执行权限。

## 实验流程

1. 刚性固定标定板，设置黑框实际边长、装板手臂及相机参数。
2. 在准备界面检查模型、图像、关节反馈与运动空间，按界面确认解锁。
3. 选择自动采集；程序检查路径并执行复位、采集、计算，异常时停止并保留已采数据。
4. 查看质量报告和异常样本，必要时补拍或替换后重新计算。
5. 在精度验证页选择标定结果或URDF对照，预览路径后由用户确认运动。

当前配置为左手、`DICT_6X6_250`、黑框57mm、板外形75×75mm、标定夹爪开度240°；配置入口：[config/experiment.yaml](config/experiment.yaml)。这些是306当前实验参数，请按实物修改。板装在活动手指上时，采集期间必须固定开度，重新安装后重新标定。

## 结果与安全

本地 `sessions/` 保存图像、会话和候选结果。变换约定为 `T_A_B` 将B坐标变换到A，平移单位米；输出包括相机及标定板相关外参。**有数值结果不代表精度通过，内部残差也不等于真实抓取精度。**程序不会自动把候选结果部署到抓取系统。

自动规划有模型自碰撞和视野检查，但不具备完整外部环境避障。必须清空运动区域并有人看护实体急停，不要同时运行其他机械臂控制程序。软件停止不是硬件急停；启动界面本身不会自动使能或运动。

本仓库只发布程序、配置和测试源码，不含现场照片、实际标定会话、日志、厂商模型/二进制。部分回放测试需要额外本地数据，不要批量执行连接真机的诊断脚本。

进一步说明：[使用说明与历史变更](README_ZH.md)、[高分辨率自动采集](HD_AUTO_CALIBRATION_ZH.md)、[验证模块](VERIFICATION_ZH.md)、[深度选点](DEPTH_VERIFICATION.md)。历史说明保留各阶段参数，当前行为以源码、配置及界面为准。依赖组件遵循各自许可，仓库公开不改变厂商SDK的授权范围。
