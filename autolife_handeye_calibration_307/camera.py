"""Read-only latest-frame camera worker. No vendor service start/restart."""
import importlib.util
import sys
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from .core import detect

class Camera:
    def __init__(self, config):
        self.config=dict(config); self.lock=threading.Lock(); self.latest=None
        self.error='等待头部RGB共享内存（请先启动系统视觉服务）'
        self.stop_event=threading.Event(); self.consumer=None
        self.depth_consumer=None
        self.thread=threading.Thread(target=self.run,daemon=True); self.thread.start()

    def get(self):
        with self.lock: return self.latest,self.error

    def configure(self, cfg):
        with self.lock: self.config=dict(cfg)

    def close_depth(self):
        if self.depth_consumer is not None:
            try:self.depth_consumer.close()
            except Exception:pass
        self.depth_consumer=None

    def depth_data(self,image,frame_id,intr,cfg):
        if not cfg.get('read_depth',False):
            self.close_depth();return {}
        try:
            if self.depth_consumer is None:
                suffix='rgbd_head_depth'
                opts={'data_shm_name':f'/camera_image_buffer_{suffix}',
                      'meta_shm_name':f'/camera_metadata_struct_{suffix}',
                      'read_mutex_name':f'/read_mutex_lock_{suffix}',
                      'write_lock_name':f'/write_data_lock_{suffix}',
                      'writer_queue_name':f'/writer_queue_sem_{suffix}',
                      'intrinsics_shm_name':f'/camera_intrinsics_struct_{suffix}'}
                self.depth_consumer=type(self.consumer)({'selected_protocol':'v1','protocols':{'v1':opts}},name='handeye_depth_readonly')
            item=self.depth_consumer.get_latest(nonblock=True)
            if item is None:return {'depth_error':'等待深度帧'}
            depth,ident=item;di=dict(self.depth_consumer.get_intrinsics())
            from .depth_target import aligned_depth
            aligned_depth(image,depth,frame_id,ident,intr,di)
            return {'depth':np.array(depth,copy=True),'depth_id':int(ident),'depth_intrinsics':di,
                    'depth_scale_m':float(cfg.get('depth_scale_m',.001)),'depth_error':''}
        except ValueError as exc:return {'depth_error':str(exc)}
        except Exception as exc:
            self.close_depth();return {'depth_error':'深度不可用：'+str(exc)}

    def run(self):
        last_id=None; last_intr=0; intr=None
        while not self.stop_event.is_set():
            try:
                with self.lock: cfg=dict(self.config)
                if self.consumer is None:
                    # Use only the installed ABI-compatible extension, never
                    # prepend all Conda packages (would shadow system Qt/NumPy).
                    if importlib.util.find_spec('posix_ipc') is None:
                        ipc=next(Path(cfg['vendor_python_path']).glob('posix_ipc*.so'),None)
                        if ipc is None: raise RuntimeError('缺少posix_ipc，请安装python3-posix-ipc或厂商相机依赖')
                        spec=importlib.util.spec_from_file_location('posix_ipc',ipc)
                        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
                        sys.modules['posix_ipc']=module
                    candidates=sorted((Path(cfg['vendor_python_path'])/'autolife_robot_sdk/utils').glob('camera_shm*.so'))
                    if not candidates: raise RuntimeError('未找到厂商相机共享内存适配器')
                    spec=importlib.util.spec_from_file_location('camera_shm',candidates[0])
                    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                    suffix='rgbd_head_color'
                    options={'data_shm_name':f'/camera_image_buffer_{suffix}',
                             'meta_shm_name':f'/camera_metadata_struct_{suffix}',
                             'read_mutex_name':f'/read_mutex_lock_{suffix}',
                             'write_lock_name':f'/write_data_lock_{suffix}',
                             'writer_queue_name':f'/writer_queue_sem_{suffix}',
                             'intrinsics_shm_name':f'/camera_intrinsics_struct_{suffix}'}
                    self.consumer=mod.SHMCameraFrameConsumer({'selected_protocol':'v1','protocols':{'v1':options}},name='handeye_readonly')
                    last_id=None; last_intr=0
                if time.monotonic()-last_intr>2:
                    intr=dict(self.consumer.get_intrinsics()); last_intr=time.monotonic()
                pair=self.consumer.get_latest(nonblock=True)
                if pair is None or pair[1]==last_id:
                    self.stop_event.wait(.04); continue
                image, frame_id=pair; image=np.array(image,copy=True)
                received=time.monotonic(); last_id=frame_id
                depth=self.depth_data(image,frame_id,intr,cfg)
                if image.ndim!=3 or image.shape[2]!=3: raise ValueError('RGB图像格式不是三通道')
                K=np.array([[intr['fx'],0,intr['ppx']],[0,intr['fy'],intr['ppy']],[0,0,1.]],float)
                D=np.array(intr.get('coeffs',[]),float)
                if len(D) not in (0,4,5,8,12,14): raise ValueError('相机畸变参数长度不支持')
                if not np.all(np.isfinite(K)) or K[0,0]<=0 or K[1,1]<=0: raise ValueError('相机内参无效')
                result=detect(image,K,D,cfg)
                snapshot={'image':image,'K':K,'D':D,'intrinsics':intr,'received':received,
                          'frame_id':int(frame_id),'detection':result,'config':cfg,
                          'time_source':'host_monotonic_after_SHM_read_not_hardware_capture',**depth}
                with self.lock:self.latest=snapshot; self.error=''
                self.stop_event.wait(.09) # bounded 10Hz CPU detection, capacity-one latest output
            except Exception as exc:
                self.close_depth()
                with self.lock:self.error=str(exc); self.latest=None
                if self.consumer is not None:
                    try:self.consumer.close()
                    except Exception:pass
                self.consumer=None; self.stop_event.wait(1.)
        self.close_depth()
        if self.consumer is not None:
            try:self.consumer.close()
            except Exception:pass

    def close(self):
        self.stop_event.set(); self.thread.join(timeout=2.)
