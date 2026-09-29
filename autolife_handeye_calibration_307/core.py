"""Pure numerical/storage layer; never imports ROS or sends robot commands."""
import copy
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

TORSO = 'Link_Waist_Yaw_to_Shoulder_Inner'
MOUNT = 'Link_Camera_Head_Forehead'
NECK = ['Joint_Neck_Roll', 'Joint_Neck_Pitch', 'Joint_Neck_Yaw']
BODY = ['Joint_Ankle', 'Joint_Knee', 'Joint_Waist_Pitch', 'Joint_Waist_Yaw']
ARMS = {side: [f'Joint_{side.title()}_{s}' for s in
    ['Shoulder_Inner', 'Shoulder_Outer', 'UpperArm', 'Elbow', 'Forearm', 'Wrist_Upper', 'Wrist_Lower']]
    for side in ('left', 'right')}

def matrix(R, t):
    out = np.eye(4); out[:3,:3] = R; out[:3,3] = np.asarray(t).ravel(); return out

def average_pose(poses):
    poses = np.asarray(poses)
    return matrix(Rotation.from_matrix(poses[:,:3,:3]).mean().as_matrix(),
                  np.median(poses[:,:3,3], axis=0))

def pose_error(a, b):
    d = np.linalg.inv(a) @ b
    return float(np.linalg.norm(d[:3,3])*1000), float(np.rad2deg(Rotation.from_matrix(d[:3,:3]).magnitude()))

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()

class Kinematics:
    def __init__(self, path):
        self.sha = digest(path)
        self.joints = {}
        for j in ET.parse(path).getroot().findall('joint'):
            def values(tag, key, default):
                el = j.find(tag)
                return np.array([float(x) for x in (el.get(key, default) if el is not None else default).split()])
            origin = matrix(Rotation.from_euler('xyz', values('origin','rpy','0 0 0')).as_matrix(),
                            values('origin','xyz','0 0 0'))
            self.joints[j.find('child').get('link')] = (j.get('name'), j.get('type'),
                j.find('parent').get('link'), origin, values('axis','xyz','1 0 0'))

    def fk(self, root, tip, qdeg):
        chain=[]; cur=tip
        while cur != root:
            if cur not in self.joints: raise ValueError(f'URDF缺少 {root} → {tip}')
            j=self.joints[cur]; chain.append(j); cur=j[2]
        out=np.eye(4)
        for name, kind, _, origin, axis in reversed(chain):
            out=out@origin
            if kind in ('revolute','continuous','prismatic'):
                if name not in qdeg or not math.isfinite(qdeg[name]): raise ValueError('缺少关节反馈 '+name)
                axis=axis/np.linalg.norm(axis)
                motion = (matrix(Rotation.from_rotvec(axis*np.deg2rad(qdeg[name])).as_matrix(),[0,0,0])
                          if kind != 'prismatic' else matrix(np.eye(3),axis*qdeg[name]))
                out=out@motion
        return out

def object_corners(size_m):
    h=size_m/2
    return np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],np.float64)

def detect(image, K, D, config):
    ar=cv2.aruco
    dictionary=ar.getPredefinedDictionary(getattr(ar,config['dictionary']))
    # OpenCV 4.6 exposes a DetectorParameters type whose direct constructor
    # is unsafe; its factory must be used. New API accompanies ArucoDetector.
    params=ar.DetectorParameters() if hasattr(ar,'ArucoDetector') else ar.DetectorParameters_create()
    params.cornerRefinementMethod=ar.CORNER_REFINE_SUBPIX
    if hasattr(ar,'ArucoDetector'): corners,ids,_=ar.ArucoDetector(dictionary,params).detectMarkers(image)
    else: corners,ids,_=ar.detectMarkers(image,dictionary,parameters=params)
    found=[] if ids is None else [int(v) for v in ids.ravel()]
    wanted=int(config['marker_id'])
    if wanted<0:
        if len(found)!=1:
            return {'ok':False,'detected_ids':found,'reason':('检测到多个码，请只露出本次标定的一个码，或手动选择ID' if found else '请将标定码完整放入画面')}
        wanted=found[0]
    idx=[] if ids is None else np.flatnonzero(ids.ravel()==wanted)
    if len(idx)!=1: return {'ok':False,'detected_ids':found,'reason':'未找到目标码，或画面中有重复ID；完整露出黑框与白边'}
    points=np.asarray(corners[int(idx[0])][0],np.float64)
    edge=float(np.min(np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)))
    obj=object_corners(config['marker_size_mm']/1000.)
    results=cv2.solvePnPGeneric(obj,points,K,D,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    options=[]
    for r,t in zip(results[1],results[2]):
        R=cv2.Rodrigues(r)[0]
        if np.min((obj@R.T+t.ravel())[:,2])<=0: continue
        proj=cv2.projectPoints(obj,r,t,K,D)[0].reshape(-1,2)
        error=float(np.sqrt(np.mean(np.sum((proj-points)**2,axis=1))))
        options.append((error,matrix(R,t)))
    if not options: return {'ok':False,'reason':'板位姿求解失败，请改变角度'}
    options.sort(key=lambda x:x[0]); error,C=options[0]
    ambiguous=(len(options)>1 and options[1][0]-error<0.05 and pose_error(C,options[1][1])[1]>10)
    reason=('码太小，请靠近或使用更大码' if edge<config['minimum_marker_pixels'] else
            '角点不稳定，请停稳、调整光照或擦净镜头' if error>config['maximum_reprojection_px'] else
            '平面姿态有歧义，请轻微倾斜标定板后再拍' if ambiguous else '')
    return {'ok':not reason,'reason':reason or '识别正常','marker_id':wanted,'detected_ids':found, 'C':C.tolist(), 'corners':points.tolist(),
            'edge_px':edge,'reprojection_px':error,'ambiguous':ambiguous}

def guides():
    tips = [
      '让标定板完整朝向相机，放在安全的胸前工作区，保持肘部弯曲。',
      '在上一步附近稍向左移动，板仍需完整可见。',
      '稍向右移动，避免跨过身体中线或碰到另一只手。',
      '稍抬高手臂，注意板不要触碰头部。',
      '稍降低手臂，保持板在图像中。',
      '稍靠近相机，避开头部和机身；不要完全伸直肘。',
      '稍远离相机，码的四边仍需清晰。',
      '轻微倾斜手腕，让板上边缘远离相机。',
      '换另一方向倾斜，让板下边缘远离相机。',
      '让板左边缘稍远离相机，保持在视野中。',
      '让板右边缘稍远离相机，不要扭到腕部限位。',
      '轻微绕板中心旋转，改变码的朝向。',
    ]
    steps=[{'phase':'train','hint':s} for s in tips]
    steps += [{'phase':'train','hint':'换一个安全位置后：'+s} for s in tips]
    steps += [{'phase':'validation','hint':'独立验证，不参与拟合：'+s} for s in tips[:8]]
    steps += [{'phase':'head','hint':'头部跨姿态验证：保持板可见，'+s} for s in
              ['颈部稍低头。','颈部稍抬头。','头稍向左转。','头稍向右转。','低头并稍左转。','低头并稍右转。']]
    return steps

def sample_guard(q, reference, phase, config, history, now):
    fixed=BODY+[f'Joint_{config["arm"].title()}_Gripper']+(NECK if phase!='head' else [])
    for name in fixed:
        tol=config['fixed_gripper_tolerance_deg'] if 'Gripper' in name else config['fixed_head_body_tolerance_deg']
        if name not in q or name not in reference or abs(q[name]-reference[name])>tol:
            return '固定姿态发生变化：'+name+'；恢复本轮基准后再拍，或新建一轮'
    window=config['stationary_window_sec']
    recent=[(t,v) for t,v in history if now-t<=window+0.2]
    if not recent or now-recent[0][0]<window or now-recent[-1][0]>.2:
        return '正在等待连续关节反馈和停稳'
    names=BODY+NECK+ARMS[config['arm']]+[f'Joint_{config["arm"].title()}_Gripper']
    for n in names:
        values=[v.get(n,float('nan')) for _,v in recent]
        if not np.all(np.isfinite(values)) or np.ptp(values)>config['stationary_range_deg']:
            return '请停稳约1秒再记录：'+n
    return ''

class Session:
    def __init__(self, directory, config, reference, urdf_sha, steps=None):
        self.path=Path(directory); self.path.mkdir(parents=True, exist_ok=False)
        self.data={'schema':1,'created':time.time(),'config':copy.deepcopy(config),'reference':dict(reference),
                   'urdf_sha256':urdf_sha,'steps':copy.deepcopy(steps) if steps is not None else guides(),'samples':{},'revisions':[], 'report':None}
        self.save()

    @classmethod
    def load(cls, directory):
        s=cls.__new__(cls); s.path=Path(directory)
        s.data=json.loads((s.path/'session.json').read_text(encoding='utf-8'))
        if s.data.get('schema')!=1: raise ValueError('实验文件版本不兼容')
        return s

    def save(self):
        atomic_json(self.path/'session.json',self.data)
        candidate=self.path/'calibration_candidate.json'
        if self.data.get('report') is None and candidate.exists():
            atomic_json(candidate,{'valid':False,'reason':'样本已变更，请重新计算；旧候选已失效','applied_to_robot':False})

    def record(self,index,image,sample):
        key=str(index); name=f'sample_{index+1:02d}_{uuid.uuid4().hex[:8]}.png'
        target=self.path/name
        if not cv2.imwrite(str(target),image): raise OSError('图像写入失败')
        previous=copy.deepcopy(self.data)
        if key in self.data['samples']: self.data['revisions'].append(self.data['samples'][key])
        self.data['samples'][key]={**sample,'image':name,'slot':index,'phase':self.data['steps'][index]['phase']}
        self.data['report']=None
        if self.data['steps'][index]['phase']!='head':self.data.pop('fixed_solution',None)
        try:self.save()
        except Exception:self.data=previous; raise
        return next((i for i in range(index+1,len(self.data['steps'])) if str(i) not in self.data['samples']),
                    next((i for i in range(len(self.data['steps'])) if str(i) not in self.data['samples']),index))

def has_solution(report):
    """A computable rigid transform is distinct from passing accuracy checks."""
    try:
        for key in ('T_torso_camera','T_mount_camera','T_wrist_marker'):
            T=np.asarray(report[key],dtype=float)
            if T.shape!=(4,4) or not np.isfinite(T).all():return False
            R=T[:3,:3]
            if not np.allclose(T[3],[0,0,0,1],atol=1e-6):return False
            if not np.allclose(R.T@R,np.eye(3),atol=.01) or abs(np.linalg.det(R)-1)>.01:return False
        return True
    except (KeyError,TypeError,ValueError):return False


def compute_report(data):
    """Camera-to-torso from inverse wrist FK (eye-to-hand); independent holdout tests."""
    cfg=data['config']; samples=list(data['samples'].values())
    train=[s for s in samples if s['phase']=='train' and s.get('included',True)]
    if len(train)<cfg['minimum_training_samples']: raise ValueError('训练样本不足，请继续采集或替换异常样本')
    gs=[np.array(s['G']) for s in train]; cs=[np.array(s['C']) for s in train]
    # Non-parallel rotation excitation is essential. Two independent axes suffice.
    rot=np.array([Rotation.from_matrix(gs[0][:3,:3].T@g[:3,:3]).as_rotvec() for g in gs[1:]])
    singular=np.linalg.svd(rot,compute_uv=False)
    if len(singular)<2 or singular[1]<0.15 or np.max(np.linalg.norm(rot,axis=1))<0.25:
        raise ValueError('旋转方向太单一：请绕至少两个不同方向转动手腕/手臂，重拍部分训练样本')
    candidates=[]
    for label,method in [('Park',cv2.CALIB_HAND_EYE_PARK),('Tsai',cv2.CALIB_HAND_EYE_TSAI)]:
        inverse=[np.linalg.inv(g) for g in gs]
        try:
            R,t=cv2.calibrateHandEye([g[:3,:3] for g in inverse],[g[:3,3] for g in inverse],
                [c[:3,:3] for c in cs],[c[:3,3] for c in cs],method=method)
            X=matrix(R,t)
            if not np.all(np.isfinite(X)) or abs(np.linalg.det(R)-1)>0.01: continue
            Y=average_pose([np.linalg.inv(g)@X@c for g,c in zip(gs,cs)])
            errors=[pose_error(g@Y,X@c) for g,c in zip(gs,cs)]
            score=float(np.median([e[0]+3*e[1] for e in errors]))
            candidates.append((score,label,X,Y))
        except (cv2.error,ValueError,np.linalg.LinAlgError):continue
    if not candidates: raise ValueError('标定求解退化；请增加多方向转动并核对码尺寸、关节反馈')
    candidates.sort(key=lambda x:x[0]); _,method,X,Y=candidates[0]
    H0=average_pose([s['H'] for s in train]); mount=np.linalg.inv(H0)@X
    detail=[]
    for s in samples:
        if not s.get('included',True): continue
        # Head verification dynamically propagates the calibrated mount through neck FK.
        camera=np.array(s['H'])@mount if s['phase']=='head' else X
        pred=np.linalg.inv(camera)@np.array(s['G'])@Y
        pe,ae=pose_error(pred,np.array(s['C']))
        obj=object_corners(cfg['marker_size_mm']/1000.)
        K=np.array(s['K']); D=np.array(s['D'])
        if np.min((obj@pred[:3,:3].T+pred[:3,3])[:,2])<=0: px=1000000.
        else:
            pts=cv2.projectPoints(obj,cv2.Rodrigues(pred[:3,:3])[0],pred[:3,3],K,D)[0].reshape(-1,2)
            px=float(np.sqrt(np.mean(np.sum((pts-np.array(s['corners']))**2,axis=1))))
        bad=pe>cfg['validation_position_mm'] or ae>cfg['validation_angle_deg'] or px>3.
        detail.append({'slot':s['slot'],'phase':s['phase'],'position_mm':pe,'angle_deg':ae,
                       'reprojection_px':px,'replace_suggested':bool(bad)})
    def stats(phase):
        es=[e for e in detail if e['phase']==phase]
        return {'count':len(es), **({} if not es else {
            'position_median_mm':float(np.median([e['position_mm'] for e in es])),
            'position_p95_mm':float(np.percentile([e['position_mm'] for e in es],95)),
            'angle_p95_deg':float(np.percentile([e['angle_deg'] for e in es],95)),
            'pixel_p95':float(np.percentile([e['reprojection_px'] for e in es],95))})}
    valid=stats('validation'); head=stats('head')
    def passes(s):
        return s.get('position_p95_mm',1e9)<=cfg['validation_position_mm'] and s.get('angle_p95_deg',1e9)<=cfg['validation_angle_deg'] and s.get('pixel_p95',1e9)<=3.
    agreement=None if len(candidates)<2 else pose_error(candidates[0][2],candidates[1][2])
    fixed_ok=bool(valid['count']>=cfg['minimum_validation_samples'] and passes(valid)
                  and agreement is not None and agreement[0]<=10 and agreement[1]<=2)
    report={'computed':time.time(),'method':method,'fixed_head_pass':fixed_ok,
        'moving_head_pass':bool(fixed_ok and head['count']>=cfg.get('auto_head_validation_samples',4) and passes(head)),
        'status':('已计算标定结果，固定头部验证通过；仍需实物对准验证' if fixed_ok else '已计算标定结果，精度未达推荐值；可查看外参，不自动用于抓取'),
        'T_torso_camera':X.tolist(),'T_mount_camera':mount.tolist(),'T_wrist_marker':Y.tolist(),
        'method_agreement_mm_deg':agreement,'rotation_singular_values':singular.tolist(),
        'training':stats('train'),'validation':valid,'head':head,'samples':detail,
        'transform_convention':'T_A_B maps B coordinates to A; quaternion export xyzw; translations metres',
        'not_calibrated':['actual_grasp_TCP','depth_scale_and_alignment','joint_zero_offsets','physical_accuracy']}
    if not has_solution(report):raise ValueError('外参不是有效的有限刚体变换，无法输出标定结果')
    report['result_available']=True
    report['quality_grade']='validated' if report['moving_head_pass'] else 'needs_validation'
    return report
