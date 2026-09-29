import time
import copy
import numpy as np
import pytest
from test_ui import win
from test_verification import data_for
from autolife_handeye_calibration_307.depth_target import surface_point,PointSamples,aligned_depth
from autolife_handeye_calibration_307.depth_selection import image_pixel


def depth_frame():
    intr={'fx':600.,'fy':600.,'ppx':20.,'ppy':20.,'model_id':2,'coeffs':[0.]*5}
    return {'image':np.zeros((40,40,3),np.uint8),'depth':np.full((40,40),800,np.uint16),
            'frame_id':1,'depth_id':1,'intrinsics':intr,'depth_intrinsics':dict(intr),
            'K':np.array([[600.,0,20.],[0,600.,20.],[0,0,1]]),'D':np.zeros(5),
            'received':time.monotonic(),'depth_scale_m':.001}


def test_neighborhood_ignores_bad_single_pixel():
    f=depth_frame();f['depth'][20,20]=4000
    p,stats=surface_point(f,(20,20),time.monotonic())
    assert np.allclose(p,[0,0,.8]) and stats['inlier_pixels']==80


@pytest.mark.parametrize('case',['hole','edge','mismatch','unaligned','stale'])
def test_bad_depth_rejected(case):
    f=depth_frame()
    if case=='hole':f['depth'][18:23,18:23]=0
    if case=='edge':f['depth'][:,20:]=1600
    if case=='mismatch':f['depth_id']=2
    if case=='unaligned':f['depth_intrinsics']['fx']=500
    if case=='stale':f['received']-=1
    with pytest.raises(ValueError):surface_point(f,(20,20),time.monotonic())


def test_seven_unique_frames_and_one_outlier():
    s=PointSamples();stats={'spatial_mad_mm':1.,'depth_m':.8}
    assert s.add(1,[0,0,.8],[0,0,0],stats) is None
    for _ in range(9):assert s.add(1,[0,0,.8],[0,0,0],stats) is None
    for i in range(2,7):assert s.add(i,[0,0,.8],[0,0,0],stats) is None
    result=s.add(7,[0,0,.819],[0,0,0],stats)
    assert result['inlier_frames']==6 and np.allclose(result['point'],[0,0,.8])


def test_multiframe_moving_object_rejected():
    s=PointSamples();stats={'spatial_mad_mm':1.,'depth_m':.8}
    with pytest.raises(ValueError):
        for i in range(7):s.add(i,[0,0,.8+i*.01],[0,0,0],stats)


def test_click_coordinate_handles_letterboxing():
    assert image_pixel((300,200),(100,50,400,300),(480,640,3))==(320,240)
    assert image_pixel((20,200),(100,50,400,300),(480,640,3)) is None
    assert image_pixel((500,200),(100,50,400,300),(480,640,3)) is None


def test_depth_mode_does_not_mutate_calibration_or_send_commands(win):
    v=win.verification;d=data_for(win);d['report'].update(status='test',validation={})
    win.session.data=d;original=copy.deepcopy(d['config']);v.load()
    v.mode.setCurrentIndex(1)
    assert v.is_depth and win.camera.config['read_depth']
    assert d['config']==original and not win.node.sent
    v.mode.setCurrentIndex(0)
    assert not win.camera.config['read_depth'] and v.depth.result is None


def test_depth_selection_collects_then_plans_without_marker(win):
    v=win.verification;d=data_for(win);d['report'].update(status='test',validation={})
    win.session.data=d;v.load();v.mode.setCurrentIndex(1);win.pages.setCurrentIndex(3)
    f=win.camera.frame;f['detection'].update(ok=False,reason='无靶标',detected_ids=[])
    intr={'fx':600.,'fy':600.,'ppx':320.,'ppy':240.,'model_id':2,'coeffs':[0.]*5}
    f.update(depth=np.full((480,640),800,np.uint16),depth_id=f['frame_id'],intrinsics=intr,
             depth_intrinsics=intr.copy(),depth_scale_m=.001)
    now=time.monotonic();q=win.node._current
    f['received']=now;win.node.history=[(t,dict(q)) for t in np.linspace(now-.9,now,60)]
    v.depth.begin((320,240),f)
    assert v.state=='selecting' and win.motion_gate('right_arm')
    for i in range(7):
        f['frame_id']=f['depth_id']=i+2;f['received']=time.monotonic()
        win.node.history.append((f['received'],dict(q)));v.depth.tick(f)
    assert v.state=='idle' and v.depth.result['frames']==7 and not win.node.sent
    T,goal=v.depth.target(f,False)
    assert np.linalg.norm(T[:3,3]-goal)==pytest.approx(.03)
    # A changed source pose invalidates execution even with an unchanged pixel.
    q['Joint_Neck_Pitch']+=2
    with pytest.raises(ValueError,match='姿态发生变化'):v.depth.target(f,True)
