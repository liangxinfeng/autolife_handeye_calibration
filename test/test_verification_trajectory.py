import numpy as np
import pytest
from autolife_handeye_calibration_307.verification_trajectory import JointPath,PathFollower


def path():return JointPath(['a','b'],[{'a':0,'b':0},{'a':3,'b':2},{'a':6,'b':4},{'a':10,'b':6}])


def test_multiple_joints_advance_without_intermediate_arrival():
    p=path();f=PathFollower(p,0);q={'a':0.,'b':0.};commands=[]
    for now in np.arange(.05,20,.05):
        cmd=f.step(q,float(now))
        if cmd is not None:
            assert max(abs(cmd[n]-q[n]) for n in q)<=1.2+1e-8
            commands.append(cmd)
        q={n:q[n]+.6*(f.command[i]-q[n]) for i,n in enumerate(q)}
        if f.final:break
    assert f.final and len(commands)>10
    assert all(c['a']>0 and c['b']>0 for c in commands)
    assert commands[-1]==pytest.approx({'a':10,'b':6})
    assert any(2.8<c['a']<3.2 for c in commands)


def test_feedback_lag_freezes_clock_no_catch_up_jump():
    f=PathFollower(path(),0);q={'a':0.,'b':0.}
    for now in np.arange(.05,5,.05):f.step(q,float(now))
    stalled=f.progress
    assert stalled<2
    for now in np.arange(5,15,.05):f.step(q,float(now))
    assert f.progress==pytest.approx(stalled)
    q=f.path.values(f.progress)
    before=f.command.copy();cmd=f.step(q,100.)
    assert np.max(np.abs(f.command-before))<=.135+1e-8


def test_invalid_observation_no_new_goals_and_no_time_backlog():
    f=PathFollower(path(),0)
    assert f.step({'a':0,'b':0},5,valid=False) is None
    assert f.progress==0 and f.velocity==0
    f.step({'a':0,'b':0},100)
    assert f.progress<=.135+1e-8


def test_off_curve_feedback_rejected():
    f=PathFollower(path(),0)
    with pytest.raises(ValueError,match='预检轨迹'):f.step({'a':0,'b':3},.1)


def test_smooth_curve_checked_between_all_vertices():
    p=path();samples=p.checks()
    assert len(samples)>len(p.points)
    assert p.at(0)==pytest.approx([0,0]) and p.at(p.length)==pytest.approx([10,6])
    assert np.isfinite(p.curve(p.s,1)).all()
    assert p.max_speed<=24 and p.acceleration<=54


def test_faster_profile_respects_joint_derivative_budgets():
    p=path();s=np.linspace(0,p.length,2001)
    d1=np.max(np.abs(p.curve(s,1)));d2=np.max(np.abs(p.curve(s,2)))
    assert d1*p.max_speed<=24.+1e-6
    assert d2*p.max_speed**2+d1*p.acceleration<=108.+1e-6


def test_faster_profile_arrives_earlier_without_changing_geometry():
    fast=path();slow=path();slow.max_speed/=2.;slow.acceleration/=4.
    def duration(p):
        f=PathFollower(p,0)
        for t in np.arange(.02,30,.02):
            q=dict(zip(p.names,f.command))
            f.step(q,float(t))
            if f.final:return float(t)
        pytest.fail('trajectory failed to finish')
    assert duration(fast)<.6*duration(slow)
    assert fast.checks()==slow.checks()


def test_final_target_not_reissued_forever():
    f=PathFollower(path(),0)
    f.progress=f.path.length;f.command=f.path.at(f.progress)
    assert f.step({'a':10,'b':6},.1) is None
