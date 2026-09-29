"""Only verification relaxes size. Pose-quality and saved calibration stay intact."""
import copy
import numpy as np
from test_ui import win
from autolife_handeye_calibration_307.core import detect


def test_verification_size_is_separate_and_restored_on_leave(win):
    d=win.session.data
    d['report']={k:np.eye(4).tolist() for k in ('T_torso_camera','T_mount_camera','T_wrist_marker')}
    d['report'].update(status='test',validation={'position_p95_mm':8.})
    original=copy.deepcopy(d['config'])
    win.goto_page(3)
    assert win.verification.detect_cfg['minimum_marker_pixels']==32
    assert win.camera.config['minimum_marker_pixels']==32
    assert d['config']==original and win.cfg==original
    assert win.verification.detect_cfg['maximum_reprojection_px']==original['maximum_reprojection_px']
    win.goto_page(1)
    assert win.camera.config['minimum_marker_pixels']==48
    assert not win.node.sent
