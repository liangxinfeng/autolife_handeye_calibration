"""Local position-IK path planning. No ROS node and no hardware publishers."""
import json
import sys
import time
import numpy as np
from control_center_joint_control.joint_definitions import GROUPS
from .core import ARMS, Kinematics
from .auto_plan_worker import CollisionGuard
from .verification_core import solve_position, tcp_position
from .verification_trajectory import JointPath


def plan(request):
    started = time.monotonic()
    cfg = request['config']; fk = Kinematics(cfg['urdf_path'])
    if fk.sha != request['urdf_sha256']:
        raise ValueError('URDF已变更，请重新核对标定')
    source = request['q']; tcp = np.asarray(request['tcp']); target = np.asarray(request['goal'])
    if tcp.shape != (3,) or target.shape != (3,) or not np.isfinite(np.r_[tcp, target]).all():
        raise ValueError('目标/TCP无效')
    start = tcp_position(fk, cfg, source, tcp)
    distance = float(np.linalg.norm(target - start))
    if distance > .30:
        raise ValueError('单次验证位移超过30厘米，请先在调姿页移到靶标附近')
    guard = CollisionGuard(cfg); guard.check(source)
    names = ARMS[cfg['arm']]
    # Use intersection of GUI limits and collision-model URDF limits.
    limits = {n: (lo, hi) for n, _, lo, hi in GROUPS[cfg['arm']+'_arm']['joints']}
    bounds = [(max(limits[n][0], np.rad2deg(guard.kin.lower[guard.kin.joint_q[n]])),
               min(limits[n][1], np.rad2deg(guard.kin.upper[guard.kin.joint_q[n]]))) for n in names]
    previous = dict(source); waypoints = [dict(source)]; checked = 1
    for point in np.linspace(start, target, max(2, int(np.ceil(distance/.008))+1))[1:]:
        if time.monotonic() - started > 50:
            raise ValueError('局部规划超时，请调整起始位姿')
        goal = solve_position(fk, cfg, previous, tcp, point, bounds)
        guard.check(goal);checked+=1
        waypoints.append(goal);previous=goal
    path=JointPath(names,waypoints)
    for values in path.checks():
        if time.monotonic()-started>50:raise ValueError('平滑轨迹检查超时，没有执行运动')
        q={**source,**values};guard.check(q);checked+=1
        xyz=tcp_position(fk,cfg,q,tcp)
        axis=target-start
        fraction=np.clip(np.dot(xyz-start,axis)/max(np.dot(axis,axis),1e-12),0,1)
        if np.linalg.norm(xyz-(start+fraction*axis))>.015:
            raise ValueError('平滑路径偏离末端直线超过15mm，请重新调起姿')
    error = float(np.linalg.norm(tcp_position(fk,cfg,previous,tcp)-target))
    if error > .005:
        raise ValueError('完整执行路径终点误差超出5mm，请调整起姿')
    return {'source_q':source, 'end_q':previous, 'waypoints':waypoints, 'goal':target.tolist(),
            'execution':'continuous_joint_path_v1','path_max_speed':path.max_speed,
            'tcp':tcp.tolist(), 'nominal_error_mm':error*1000, 'distance_mm':distance*1000,
            'checked_states':checked, 'planning_seconds':time.monotonic()-started,
            'external_obstacles_checked':False}


if __name__ == '__main__':
    try:
        result = {'ok':True, 'plan':plan(json.load(sys.stdin))}
    except Exception as exc:
        result = {'ok':False, 'reason':str(exc)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
