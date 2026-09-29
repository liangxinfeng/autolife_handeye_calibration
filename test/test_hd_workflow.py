import copy
from test_automation import dataset
from autolife_handeye_calibration_307.core import compute_report


def test_new_round_does_not_pass_head_validation_with_only_four_samples():
    data,_,_=dataset();data=copy.deepcopy(data)
    keys=list(range(12))+list(range(24,28))+list(range(32,36))
    data['samples']={str(i):copy.deepcopy(data['samples'][str(k)]) for i,k in enumerate(keys)}
    for i,s in enumerate(data['samples'].values()):s['slot']=i
    data['config'].update(minimum_training_samples=12,minimum_validation_samples=4,auto_head_validation_samples=8)
    r=compute_report(data)
    assert r['fixed_head_pass'] and not r['moving_head_pass']
    assert r['head']['count']==4
