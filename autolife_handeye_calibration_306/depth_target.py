"""Robust RGB-pixel -> aligned-depth point; no robot commands or side effects."""
import cv2
import numpy as np
from .verification_core import camera_transform


def aligned_depth(color, depth, color_id, depth_id, intr, depth_intr):
    if color_id!=depth_id:raise ValueError('等待同帧RGB/深度配对')
    if depth.dtype!=np.uint16 or depth.shape!=color.shape[:2]:
        raise ValueError('深度必须为与RGB同尺寸的Z16图像')
    for key in ('fx','fy','ppx','ppy','model_id'):
        if abs(float(intr[key])-float(depth_intr[key]))>1e-3:
            raise ValueError('RGB与深度内参不一致，不能直接使用同一像素')
    if not np.allclose(intr['coeffs'],depth_intr['coeffs'],atol=1e-6,rtol=0):
        raise ValueError('RGB与深度畸变参数不一致')


def surface_point(frame, pixel, now):
    if frame is None or now-frame['received']>.4:raise ValueError('RGB已过期')
    if frame.get('depth') is None:raise ValueError(frame.get('depth_error') or '等待已对齐深度')
    aligned_depth(frame['image'],frame['depth'],frame['frame_id'],frame['depth_id'],
                  frame['intrinsics'],frame['depth_intrinsics'])
    scale=float(frame['depth_scale_m'])
    if not np.isfinite(scale) or not 0<scale<.01:raise ValueError('深度单位无效')
    x,y=map(int,pixel);h,w=frame['depth'].shape;r=4
    if not r<=x<w-r or not r<=y<h-r:raise ValueError('请不要点击图像边缘')
    patch=frame['depth'][y-r:y+r+1,x-r:x+r+1].astype(float)*scale
    valid=(patch>=.15)&(patch<=2.)
    if np.mean(valid)<.65:raise ValueError('附近深度空洞太多，请选物体内部清晰区域')
    center=patch[r-1:r+2,r-1:r+2];cv=valid[r-1:r+2,r-1:r+2]
    if cv.sum()<6:raise ValueError('点击中心深度不完整，请避开空洞')
    zs=center[cv]
    if np.ptp(np.percentile(zs,[10,90]))>.02:raise ValueError('点击处跨越前后景，请避开物体轮廓')
    seed=float(np.median(zs));gate=max(.008,.012*seed)
    inliers=valid&(np.abs(patch-seed)<=gate)
    if np.mean(inliers)<.6:raise ValueError('附近表面混杂，不能把背景当作物体位置；请向物体内部重选')
    z=float(np.median(patch[inliers]))
    mad=float(np.median(np.abs(patch[inliers]-z)))
    if mad>max(.003,.004*z):raise ValueError('邻域深度不稳定，请重选')
    # Current D435i color/depth stream has zero distortion coefficients.
    # Do not apply OpenCV Brown undistortion to nonzero inverse-Brown models.
    if frame['intrinsics']['model_id'] not in (0,4) and np.any(np.abs(frame['D'])>1e-9):
        raise ValueError('当前畸变模型暂不支持深度选点')
    ray=cv2.undistortPoints(np.array([[[x,y]]],np.float64),frame['K'],frame['D']).reshape(2)
    return np.r_[ray*z,z], {'depth_m':z,'spatial_mad_mm':mad*1000,'inlier_pixels':int(inliers.sum())}


def torso_point(data,fk,history,frame,pixel,now):
    point,stats=surface_point(frame,pixel,now)
    nearest=min(history,key=lambda h:abs(h[0]-frame['received'])) if history else None
    if nearest is None or abs(nearest[0]-frame['received'])>.12:
        raise ValueError('深度图像与关节反馈未完成120ms配对')
    camera=camera_transform(data,fk,nearest[1],frame)
    return camera[:3,:3]@point+camera[:3,3],camera[:3,3],stats


class PointSamples:
    def __init__(self):self.clear()
    def clear(self):self.points=[];self.ids=set();self.result=None
    def add(self,frame_id,point,camera,stats):
        if frame_id in self.ids:return None
        self.ids.add(frame_id);self.points.append((np.array(point),np.array(camera),stats))
        if len(self.points)<7:return None
        points=np.array([p[0] for p in self.points]);median=np.median(points,axis=0)
        errors=np.linalg.norm(points-median,axis=1);keep=errors<=.008
        if keep.sum()<6 or np.max(errors)>.025:
            raise ValueError('多帧位置波动过大，请保持相机/物体不动并重新点击')
        median=np.median(points[keep],axis=0)
        self.result={'point':median.tolist(),'camera':np.median([p[1] for p in self.points],axis=0).tolist(),
                     'frames':len(points),'inlier_frames':int(keep.sum()),'spread_mm':float(np.max(errors[keep])*1000),
                     'spatial_mad_mm':float(np.median([p[2]['spatial_mad_mm'] for p in self.points])),
                     'depth_m':float(np.median([p[2]['depth_m'] for p in self.points]))}
        return self.result
