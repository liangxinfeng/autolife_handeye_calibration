"""Read-only geometry for the physical TCP verification experiment.

T_A_B maps B into A; translations are metres. The original camera calibration
is never edited. The 15 mm change belongs to the tool, not the observed box.
"""
import numpy as np
from .core import ARMS, NECK, TORSO, MOUNT, has_solution


CAMERA_SOURCES = {'calibrated':'手眼标定外参', 'urdf':'URDF名义外参（未标定）'}


def camera_geometry(data):
    """Only the camera extrinsic is an A/B variable; never change the TCP.

    The vendor URDF ends at the CAD forehead mount, not an RGB optical link.
    Use the same zero-translation/+90deg-Z optical convention as the existing
    visibility predictor and adaptive_object_grasping/config/hardware.yaml.
    This nominal convention is not a measured optical-centre calibration.
    """
    source=data.get('_verification_camera_source','calibrated')
    if source not in CAMERA_SOURCES:raise ValueError('未知相机外参来源，禁止规划')
    if source=='urdf':
        transform=np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    else:transform=np.array(data['report']['T_mount_camera'],dtype=float)
    if (transform.shape!=(4,4) or not np.all(np.isfinite(transform)) or
            not np.allclose(transform[3],[0,0,0,1]) or
            not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5) or
            not np.isclose(np.linalg.det(transform[:3,:3]),1.,atol=1e-5)):
        raise ValueError('相机外参不是有效刚体变换')
    return {'camera_source':source,'camera_source_label':CAMERA_SOURCES[source],
            'T_mount_camera_used':transform.tolist(),'camera_mount_frame':MOUNT,
            'urdf_sha256':data['urdf_sha256'],
            'head_offsets_deg_used':list(data['config']['head_offsets_deg']),
            'tcp_source':'same_calibrated_T_wrist_marker_plus_tool_offset',
            'comparison_scope':'camera_extrinsic_only'}


def wrist_frame(cfg):
    return f'Link_{cfg["arm"].title()}_Wrist_Lower_to_Gripper'


def q_for_fk(q, cfg):
    q = dict(q)
    for n, offset in zip(NECK, cfg['head_offsets_deg']):
        q[n] += offset
    return q


def load_geometry(data, fk, offset_mm=15.):
    if data['config']['robot_id'] != 283 or data['urdf_sha256'] != fk.sha:
        raise ValueError('实验机器人或URDF不匹配')
    report = data.get('report') or {}
    if not has_solution(report):
        raise ValueError('该实验没有有效的标定结果，请先计算')
    if not np.isfinite(offset_mm) or not 0 <= offset_mm <= 50:
        raise ValueError('工具偏移必须在0～50毫米内')
    # Marker is pasted on the upper gripper surface. ArUco +Z points out of
    # the printed face, hence the grasp centre is 15 mm along marker -Z.
    Y = np.array(report['T_wrist_marker'])
    correction = Y[:3, :3] @ np.array([0., 0., -offset_mm / 1000.])
    tcp = Y[:3, 3] + correction
    return tcp


def tcp_position(fk, cfg, q, tcp):
    G = fk.fk(TORSO, wrist_frame(cfg), q_for_fk(q, cfg))
    return G[:3, 3] + G[:3, :3] @ tcp


def observed_target(data, fk, q, frame, standoff_mm):
    if not np.isfinite(standoff_mm) or not 0 <= standoff_mm <= 100:
        raise ValueError('预留距离必须在0～100毫米')
    cfg = data['config']; d = frame['detection']
    if not d.get('ok'):
        raise ValueError(d.get('reason', '靶标不可用'))
    if frame['config']['marker_size_mm'] != cfg['marker_size_mm'] or frame['config']['dictionary'] != cfg['dictionary']:
        raise ValueError('靶标尺寸/字典尚未更新')
    camera = camera_transform(data, fk, q, frame)
    target = camera @ np.array(d['C'])
    toward_camera = camera[:3, 3] - target[:3, 3]
    toward_camera /= np.linalg.norm(toward_camera)
    goal = target[:3, 3] + toward_camera * standoff_mm / 1000.
    return target, goal


def camera_transform(data, fk, q, frame):
    """Shared calibration/intrinsics gate for ArUco and aligned-depth targets."""
    cfg=data['config']
    if cfg['robot_id']!=283 or data['urdf_sha256']!=fk.sha:
        raise ValueError('实验机器人或URDF不匹配')
    sample = next(iter(data['samples'].values()))
    if (list(frame['image'].shape) != sample['image_shape'] or
            np.shape(frame['K']) != np.shape(sample['K']) or
            not np.allclose(frame['K'], sample['K'], atol=1e-5, rtol=0) or
            np.shape(frame['D']) != np.shape(sample['D']) or
            not np.allclose(frame['D'], sample['D'], atol=1e-7, rtol=0)):
        raise ValueError('相机模式/内参与标定时不同')
    H = fk.fk(TORSO, MOUNT, q_for_fk(q, cfg))
    return H @ np.array(camera_geometry(data)['T_mount_camera_used'])


def checked_observation(data, fk, history, frame, now, standoff_mm):
    if frame is None or now - frame['received'] > .4:
        raise ValueError('RGB过期；请等待实时画面恢复')
    nearest = min(history, key=lambda x: abs(x[0] - frame['received'])) if history else None
    if nearest is None or abs(nearest[0] - frame['received']) > .12:
        raise ValueError('图像与关节反馈未配对，等待新数据')
    return observed_target(data, fk, nearest[1], frame, standoff_mm)


def check_path_state(q, start, end, moving_joint, tolerance=.7):
    for name, value in start.items():
        if name not in q or not np.isfinite(q[name]):
            raise ValueError('关节反馈缺失：' + name)
        tol = 1. if 'Gripper' in name else tolerance
        lo, hi = sorted((value, end[name])) if name == moving_joint else (value, value)
        if not lo - tol <= q[name] <= hi + tol:
            raise ValueError('关节偏离预检路径：' + name)


def solve_position(fk, cfg, source, tcp, target, bounds):
    """Position-first bounded IK; tiny joint regularizer avoids arbitrary flips."""
    from scipy.optimize import least_squares
    names = ARMS[cfg['arm']]
    seed = np.deg2rad([source[n] for n in names])
    low, high = np.deg2rad(np.array(bounds).T)
    if np.any(seed < low) or np.any(seed > high):
        raise ValueError('当前手臂已在限位外，不能自动裁剪到另一姿态')
    def values(x):
        return {**source, **dict(zip(names, np.rad2deg(x)))}
    def residual(x):
        return np.r_[tcp_position(fk, cfg, values(x), tcp) - target, .0005 * (x - seed)]
    fit = least_squares(residual, seed, bounds=(low, high), max_nfev=100,
                        ftol=1e-9, xtol=1e-9, gtol=1e-9, diff_step=1e-5)
    q = values(fit.x)
    error = np.linalg.norm(tcp_position(fk, cfg, q, tcp) - target)
    if not fit.success or error > .003:
        raise ValueError(f'位置IK无法达到目标（残差{error*1000:.1f}mm），请移动盒子或手动调起姿')
    if np.max(np.abs(fit.x - seed)) > np.deg2rad(8):
        raise ValueError('IK关节变化突增/接近奇异位姿，请换较弯曲的起姿')
    return q
