"""Continuous position targets on a prechecked, shape-preserving joint path.

No ROS/hardware access. Clock progress never runs ahead while feedback lags.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


class JointPath:
    def __init__(self, names, waypoints):
        self.names=list(names)
        raw=np.array([[q[n] for n in names] for q in waypoints],float)
        if raw.ndim!=2 or not np.isfinite(raw).all():raise ValueError('轨迹关节数据无效')
        keep=np.r_[True,np.max(np.abs(np.diff(raw,axis=0)),axis=1)>1e-6]
        self.points=raw[keep]
        if len(self.points)==1:self.points=np.vstack([self.points,self.points])
        self.s=np.r_[0,np.cumsum(np.max(np.abs(np.diff(self.points,axis=0)),axis=1))]
        self.length=float(self.s[-1])
        if self.length<1e-6:self.s[-1]=1e-6;self.length=1e-6
        self.curve=PchipInterpolator(self.s,self.points,axis=0)
        # Bound each cubic analytically, including both sides of IK knots:
        # curvature can jump there and a sampled estimate can miss its peak.
        d1=1.;d2=1e-6
        for i,h in enumerate(np.diff(self.s)):
            a,b,c,_=self.curve.c[:,i,:]
            d1=max(d1,float(np.max(np.abs(c))),float(np.max(np.abs(3*a*h*h+2*b*h+c))))
            d2=max(d2,float(np.max(np.abs(2*b))),float(np.max(np.abs(6*a*h+2*b))))
            nonzero=np.abs(a)>1e-15
            roots=np.zeros_like(a);roots[nonzero]=-b[nonzero]/(3*a[nonzero])
            valid=nonzero&(roots>0)&(roots<h)
            if np.any(valid):
                d1=max(d1,float(np.max(np.abs((3*a*roots*roots+2*b*roots+c)[valid]))))
        # 2x time scaling of the previous 12deg/s, 13.5+13.5deg/s² profile:
        # velocities scale by 2, accelerations by 2². Geometry, endpoint
        # tolerance and feedback lead protection are unchanged.
        self.max_speed=min(24./d1,np.sqrt(54./d2))
        self.acceleration=54./d1

    def at(self,s):return np.asarray(self.curve(np.clip(s,0,self.length)),float)

    def values(self,s):return dict(zip(self.names,map(float,self.at(s))))

    def checks(self):
        # Oversample the SMOOTHED curve, not just the original IK vertices.
        samples=np.unique(np.r_[self.s,np.linspace(0,self.length,max(2,int(np.ceil(self.length/.1))+1))])
        return [self.values(float(s)) for s in samples]


class PathFollower:
    def __init__(self,path,now):
        self.path=path;self.progress=0.;self.velocity=0.;self.last=now
        self.command=path.at(0);self.end_since=None

    @property
    def final(self):return self.progress>=self.path.length-1e-6

    def step(self,q,now,valid=True):
        actual=np.array([q[n] for n in self.path.names],float)
        if not np.isfinite(actual).all():raise ValueError('跟随反馈无效')
        dt=min(.1,max(0.,now-self.last));self.last=now
        if not valid:self.velocity=0.;return None
        # Retain a gentle first/recovery step even after a delayed UI tick.
        if self.velocity==0.:dt=min(dt,.05)
        # Actual joint combinations must remain near the recently issued curve,
        # not anywhere inside the large start/end joint bounding box.
        samples=np.linspace(max(0,self.progress-4),self.progress,41)
        distance=float(np.min(np.max(np.abs(self.path.curve(samples)-actual),axis=1)))
        if distance>1.0:raise ValueError('实测关节偏离连续预检轨迹超过1°')
        if self.final:return None
        remaining=self.path.length-self.progress
        speed=min(self.path.max_speed,np.sqrt(max(0.,2*self.path.acceleration*remaining)))
        self.velocity=min(speed,self.velocity+self.path.acceleration*dt)
        proposed=min(self.path.length,self.progress+self.velocity*dt)
        candidate=self.path.at(proposed)
        # Never accumulate wall-clock backlog. Lag pauses the path clock;
        # do not jump forward when data/motors recover.
        if np.max(np.abs(candidate-actual))>1.2:
            self.velocity=0.;return None
        if np.max(np.abs(candidate-self.command))<.015 and proposed<self.path.length:
            # Suppress tiny publications, not time progress: otherwise a slow
            # final approach can remain forever below the publication threshold.
            self.progress=proposed
            return None
        self.progress=proposed;self.command=candidate
        if self.final and self.end_since is None:self.end_since=now
        return dict(zip(self.path.names,map(float,candidate)))
