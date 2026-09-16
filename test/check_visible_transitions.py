"""Run with robot_env: pure preflight replay; no ROS node or motor commands."""
import json
import numpy as np
from autolife_handeye_calibration_306.auto_plan_worker import edges_between,CollisionGuard,plan
from autolife_handeye_calibration_306.visibility import Visibility

class Guard:
    def check(self,q):pass
class View:
    def check(self,q):return q['arm']<=q['neck']+.01
source={'arm':0.,'neck':0.};goal={'arm':5.,'neck':5.}
edges,_=edges_between(source,goal,['arm','neck'],Guard(),View())
assert edges[0]['joint']=='neck' and edges[-1]['end_q']==goal
class Blocked:
    def check(self,q):return q['arm']==q['neck']
try:edges_between(source,goal,['arm','neck'],Guard(),Blocked())
except ValueError:pass
else:raise AssertionError('unsafe intermediate path was accepted')
d=json.load(open('/tmp/handeye_live_preflight.json'))
r=json.load(open('/tmp/handeye_live_fixed_plan.json'));assert r['ok'];p=r['plan']
v=Visibility(d['config'],d['observation']);guard=CollisionGuard(d['config']);current=d['q'];count=0
for t in p['targets']:
    for e in t['edges']:
        assert e['start_q']==current
        n=e['joint']
        for a in np.linspace(e['start_q'][n],e['end_q'][n],max(2,int(abs(e['end_q'][n]-e['start_q'][n])/.25)+2)):
            q={**current,n:float(a)}
            assert v.check(q),'out-of-view intermediate sample'
            guard.check(q);count+=1
        current=e['end_q']
    assert max(abs(current[n]-t['q'][n]) for n in current)<.1
assert len(p['targets'])==24
print(json.dumps({'rechecked_at_quarter_degree':count,'targets':24,'fallback_order_test':'passed','blocked_path_test':'passed'}))
# Small observation shifts emulate mounting/pose estimation differences.
for dx in [-.002,.002]:
    obs={**d['observation'],'C':np.array(d['observation']['C']).copy()}
    obs['C'][0,3]+=dx;obs['C']=obs['C'].tolist()
    r=plan(d['q'],d['config'],obs)
    assert len(r['targets'])==24
    print(json.dumps({'observation_shift_mm':dx*1000,'targets':24,'seconds':r['planning_seconds']}))
