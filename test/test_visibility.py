import json
from pathlib import Path
import numpy as np
import yaml
from autolife_handeye_calibration_307.visibility import Visibility

ROOT=Path(__file__).resolve().parents[1]

def test_model_prediction_against_real_captured_motion():
    s=json.loads((ROOT/'test/capture_regression_20260910.json').read_text())
    a,b=list(s['samples'].values())[:2]
    cfg=yaml.safe_load((ROOT/'config/experiment.yaml').read_text())
    v=Visibility(cfg,{'q':a['q_deg'],'C':a['C'],'K':a['K'],'D':a['D'],'size':[640,480]})
    assert np.max(np.linalg.norm(v.corners(b['q_deg'])-np.array(b['corners']),axis=1))<12
    # The old second capture is already too close to the image bottom.
    assert not v.check(b['q_deg'])
    bad=dict(a['q_deg']);bad['Joint_Left_UpperArm']=-18.
    assert not v.check(bad)
    assert v.check(a['q_deg'])


def test_visibility_anchor_does_not_use_stale_board_transform():
    s=json.loads((ROOT/'test/capture_regression_20260910.json').read_text())
    a,b=list(s['samples'].values())[:2]
    cfg=s['config'];v=Visibility(cfg,{'q':a['q_deg'],'C':a['C'],'K':a['K'],'D':a['D'],'size':[640,480]})
    old=v.Y.copy();v.anchor(b['q_deg'],b['C'])
    assert not np.allclose(old,v.Y)
    assert np.max(np.linalg.norm(v.corners(b['q_deg'])-np.array(b['corners']),axis=1))<1.
