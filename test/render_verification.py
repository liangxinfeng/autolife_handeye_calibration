"""UI screenshot with a fake robot and stored image; no ROS init or motion."""
import json
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from PyQt5 import QtWidgets as Q
from test_ui import win
from autolife_handeye_calibration_307.core import Session
from autolife_handeye_calibration_307.app import STYLE

with tempfile.TemporaryDirectory(prefix='verify_ui_') as tmp:
    monkey=pytest.MonkeyPatch(); it=win.__wrapped__(Path(tmp),monkey); w=next(it)
    s=Session.load('/home/ubuntu/ros2_ws/src/autolife_handeye_calibration_307/sessions/20260914_102651_586775')
    w.session=s;w.node._current=dict(s.data['reference']);w.goto_page(3)
    sample=s.data['samples']['0'];f=w.camera.frame
    f.update(image=cv2.imread(str(s.path/sample['image'])),K=np.array(sample['K']),D=np.array(sample['D']),
             config=w.verification.detect_cfg,received=time.monotonic(),detection={**sample,'ok':True,'reason':'存档图像UI测试'})
    w.node.history=[(f['received'],w.node._current)]
    w.verification.tick(f);w.setStyleSheet(STYLE);w.resize(1500,940);w.show()
    app=Q.QApplication.instance()
    for _ in range(10):app.processEvents()
    w.grab().save('/tmp/handeye_verify306_ui.png')
    assert not w.node.sent
    print(json.dumps({'pages':w.pages.count(),'model_error':w.verification.model.render_error,
                      'mesh_count':len(w.verification.model.meshes),'sent_commands':len(w.node.sent)},ensure_ascii=False))
    try:next(it)
    except StopIteration:pass
    monkey.undo()
