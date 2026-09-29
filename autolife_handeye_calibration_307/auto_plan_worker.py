"""Offline collision preflight in robot_env. Never creates ROS nodes/publishers."""
import json
import itertools
import sys
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from ament_index_python.packages import get_package_share_directory
from openarmx_teleop_vr_306_v4.kinematics import IndependentArmKinematics
from .core import ARMS,BODY,NECK
from .visibility import Visibility

GROUPS={'leg_waist':BODY,'left_arm':ARMS['left'],'right_arm':ARMS['right'],'neck':NECK}


def make_targets(source,cfg):
    """One arm per experiment; neck variations are independent holdout only."""
    home=dict(source)
    group=cfg['arm']+'_arm'
    for n,v in zip(GROUPS[group],cfg['initial_pose_deg'][group]):home[n]=float(v)
    # Preserve measured fixed head reference. Only the final holdout changes it.
    # Other arm, waist/legs and both grippers remain unchanged throughout.
    # Wrist/forearm rotation is integrated into existing arm samples, never
    # appended as wrist-only photographs. Full excursions are checked below.
    deltas=[{}, {2:22,4:18,6:10}, {2:-22,4:-18,6:-10},
            {2:10,3:-8,5:18,6:14}, {2:-10,3:8,5:-18,6:-14},
            {0:10,3:-14,4:22,6:-12}, {0:-10,3:14,4:-22,6:12},
            {1:10,2:16,4:14,6:16}, {1:-10,2:-16,4:-14,6:-16},
            {2:18,5:16,4:-18,6:10}, {2:-18,5:-16,4:18,6:-10},
            {3:-16,4:-12,6:18},
            {0:7,2:-14,5:10,4:12,6:9}, {1:7,2:14,5:-10,4:-12,6:-9},
            {3:10,2:-7,4:16,6:-14}, {0:-7,3:-10,5:12,4:-16,6:14}]
    arm=ARMS[cfg['arm']];targets=[]
    for i,delta in enumerate(deltas):
        q=dict(home)
        for index,value in delta.items():q[arm[index]]+=value
        targets.append({'phase':'train' if i<12 else 'validation','q':q,
                        'hint':('自动手臂采集' if i<12 else '独立手臂验证')+f' {i+1}'})
    # Eight holdouts: pitch/yaw plus diagonals, not extra wrist samples.
    for pitch,yaw in [(-12,0),(12,0),(0,-16),(0,16),(-8,-12),(-8,12),(8,-12),(8,12)]:
        q=dict(home);q[NECK[1]]+=pitch;q[NECK[2]]+=yaw
        targets.append({'phase':'head','q':q,'hint':f'自动转头验证：俯仰{pitch:+}°、左右{yaw:+}°（按视野自适应）'})
    return targets


class CollisionGuard:
    def __init__(self,cfg):
        share=Path(get_package_share_directory('openarmx_teleop_vr_306_v4'))
        srdf=ET.parse(share/'urdf/robot_v2_2.srdf')
        # The VR SRDF excludes the two grippers. Restore this pair for automation.
        for el in list(srdf.getroot().findall('disable_collisions')):
            if all('Wrist_Lower_to_Gripper' in el.get(k,'') for k in ('link1','link2')):
                srdf.getroot().remove(el)
        with tempfile.TemporaryDirectory(prefix='handeye_collision_') as tmp:
            path=Path(tmp)/'guard.srdf';srdf.write(path)
            self.kin=IndependentArmKinematics(str(share/'urdf/robot_v2_2_simplified.urdf'),str(path),
                                              collision_check=True,limit_margin_rad=0)
        if not self.kin.collision_available:raise ValueError('碰撞模型加载失败：'+self.kin.collision_error)
        self.side=cfg['arm'].title()
        self.frame=self.kin.model.getFrameId(f'Link_{self.side}_Wrist_Lower_to_Gripper')
        # Conservative envelope: all opening states + protruding 75mm board.
        # Sphere centre 0.20m ahead of wrist, radius 0.14m; includes 2cm reserve.
        self.radius=float(cfg.get('board_envelope_radius_m',.14))
        self.sphere=fcl.Sphere(self.radius)
        self.offset=np.array([.20,0,0])
        self.skip=(f'Link_{self.side}_Wrist_Lower_to_Gripper',f'Link_{self.side}_Wrist_Upper_to_Wrist_Lower')
        self.boxes=[];self.cache=set();self.exact_distance_calls=0
        for obj in self.kin.geom_model.geometryObjects:
            obj.geometry.computeLocalAABB()
            box=obj.geometry.aabb_local
            self.boxes.append((np.array(box.min_),np.array(box.max_)))

    def check(self,values):
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('关节数值不是有限数')
        key=tuple(sorted(values.items()))
        if key in self.cache:return
        groups={g:[values[n] for n in names] for g,names in GROUPS.items()}
        q=self.kin.q_from_feedback(groups,clip=False)
        if not np.all(np.isfinite(q)) or np.any(q<self.kin.lower-1e-6) or np.any(q>self.kin.upper+1e-6):raise ValueError('姿态超出URDF限位')
        pairs=self.kin.collision_pairs(q)
        if pairs:raise ValueError('机器人自碰撞：'+'; '.join(pairs[:3]))
        pin.forwardKinematics(self.kin.model,self.kin.data,q);pin.updateFramePlacements(self.kin.model,self.kin.data)
        pin.updateGeometryPlacements(self.kin.model,self.kin.data,self.kin.geom_model,self.kin.geom_data,q)
        wrist=self.kin.data.oMf[self.frame];center=wrist.translation+wrist.rotation@self.offset
        a=fcl.Transform3f(np.eye(3),center)
        for i,obj in enumerate(self.kin.geom_model.geometryObjects):
            if any(obj.name.startswith(s) for s in self.skip):continue
            t=self.kin.geom_data.oMg[i]
            # Exact conservative broad phase: enclosing AABB separation implies
            # mesh separation. Do not change density/margins or collision pairs.
            local=t.rotation.T@(center-t.translation);low,high=self.boxes[i]
            if np.linalg.norm(local-np.clip(local,low,high))>self.radius+.005:continue
            res=fcl.DistanceResult();self.exact_distance_calls+=1
            distance=fcl.distance(self.sphere,a,obj.geometry,fcl.Transform3f(t.rotation,t.translation),fcl.DistanceRequest(),res)
            if distance<.005:raise ValueError('夹爪/标定板安全包络靠近：'+obj.name)
        self.cache.add(key)


def _ordered_edges(previous,goal,names,guard,vision=None):
    edges=[];current=dict(previous);count=0
    for name in names:
            finish=goal[name];start=current[name]
            if abs(finish-start)<.1:continue
            endpoint=dict(current);endpoint[name]=finish
            steps=max(2,int(np.ceil(abs(finish-start)/.5))+1)
            for value in np.linspace(start,finish,steps):
                sample=dict(current);sample[name]=float(value)
                if vision is not None and not vision.check(sample):raise ValueError('标定板将靠近画面边界或变得太小')
                guard.check(sample)
                count+=1
            edges.append({'joint':name,'target':finish,'start_q':dict(current),'end_q':endpoint})
            current=endpoint
    return edges,count


def edges_between(previous,goal,names,guard,vision=None):
    """Choose a visible joint order, without changing either endpoint.

    Returning the arm before turning the neck is not always visible. Try a
    bounded set of orders; every intermediate state uses the same checks.
    Reset paths keep the original order because there is no visual anchor yet.
    """
    if vision is None:return _ordered_edges(previous,goal,names,guard)
    active=[n for n in names if abs(goal[n]-previous[n])>=.1]
    orders=[active,list(reversed(active)),
            sorted(active,key=lambda n:(n not in NECK,abs(goal[n]-previous[n]))),
            sorted(active,key=lambda n:abs(goal[n]-previous[n])),
            sorted(active,key=lambda n:-abs(goal[n]-previous[n]))]
    seen=set();last=None
    for order in orders:
        key=tuple(order)
        if key in seen:continue
        seen.add(key)
        try:return _ordered_edges(previous,goal,order,guard,vision)
        except ValueError as exc:last=exc
    raise ValueError('中间路径无法保持视野/安全：'+str(last))


def plan(source,cfg,observation=None,mode='collect'):
    started=time.monotonic();guard=CollisionGuard(cfg);guard.check(source)
    names=ARMS[cfg['arm']]+NECK
    if mode=='reset':
        q=dict(source)
        for group,values in cfg['initial_pose_deg'].items():
            for n,v in zip(GROUPS[group],values):q[n]=float(v)
        jaw=f'Joint_{cfg["arm"].title()}_Gripper';q[jaw]=cfg['calibration_gripper_deg']
        names=names+ARMS['right' if cfg['arm']=='left' else 'left']+[jaw]
        edges,count=edges_between(source,q,names,guard)
        targets=[{'phase':'reset','q':q,'edges':edges,'hint':'自动返回初始姿态'}]
    else:
        if observation is None:raise ValueError('自动采集必须提供当前画面和关节状态用于视野检查')
        vision=Visibility(cfg,observation);home=dict(source)
        if not vision.check(source):raise ValueError('初始标定板离边界太近或太小，请先将板完整放入画面')
        # Center with a small local joint search, not an uncalibrated IK target.
        centres=[];arm=ARMS[cfg['arm']]
        for inner,upper,elbow in itertools.product([-8,0,8],[0,6,12],[-8,0,8]):
            q=dict(source)
            for n,d in [(arm[0],inner),(arm[2],upper),(arm[3],elbow)]:q[n]+=d
            if vision.check(q):centres.append((vision.centering_cost(q),q))
        for _,candidate in sorted(centres,key=lambda x:x[0]):
            try:edges_between(source,candidate,names,guard,vision)
            except ValueError:continue
            home=candidate;break
        templates=make_targets(home,cfg)
        # make_targets normally references VR home; transfer deltas to the
        # visibility-centered anchor without changing the fixed head reference.
        original=make_targets(source,cfg)[0]['q']
        for t in templates:
            for n in arm:t['q'][n]+=home[n]-original[n]
        previous=source;targets=[];count=0;selected=[]
        for index,template in enumerate(templates):
            candidates=[template]+[t for t in templates if t['phase']==template['phase'] and t is not template]
            accepted=None
            for item in candidates:
                for scale in [1.,.8,.6,.45]:
                    q={n:home[n]+scale*(v-home[n]) for n,v in item['q'].items()}
                    if not vision.check(q) or not vision.distinct(q,selected,template['phase']=='head'):continue
                    try:edges,ncheck=edges_between(previous,q,names,guard,vision)
                    except ValueError:continue
                    accepted={**template,'q':q,'edges':edges,'view_scale':scale};count+=ncheck;break
                if accepted:break
            if accepted is None:raise ValueError(f'第{index+1}张无法兼顾视野、姿态差异和碰撞；请把起始码放在更居中的位置后重试')
            targets.append(accepted);selected.append(accepted['q']);previous=accepted['q']
        arm_samples=[t['q'] for t in targets if t['phase']=='train']
        wrist_spans={n:float(np.ptp([q[n] for q in arm_samples])) for n in (arm[4],arm[6])}
        if min(wrist_spans.values())<cfg.get('auto_wrist_span_min_deg',18.):
            raise ValueError('当前视野无法保留足够左右旋转幅度；请将标定板重新安装得更居中、正对相机后重试')
    return {'targets':targets,'checked_states':count,'source_q':source,
            'kind':mode,'planning_seconds':time.monotonic()-started,'exact_distance_calls':guard.exact_distance_calls,
            'collision_model':'V4 + restored gripper pair + conservative board envelope',
            'external_obstacles_checked':False}


if __name__=='__main__':
    try:
        request=json.load(sys.stdin);result={'ok':True,'plan':plan(request['q'],request['config'],request.get('observation'),request.get('mode','collect'))}
    except Exception as exc:result={'ok':False,'reason':str(exc)}
    print(json.dumps(result,ensure_ascii=False))
