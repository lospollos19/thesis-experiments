"""Unit tests for the camera abstraction."""

from __future__ import annotations

import numpy as np
import pytest
from motion_pipeline.camera.camera_interface import BaseCamera, RealCamera


class DummyCamera(BaseCamera):
    """Minimal concrete camera to exercise BaseCamera plumbing."""

    def __init__(self) -> None:
        self.opened = False
        self.released = False

    def open(self) -> None:
        self.opened = True

    def read(self):
        return True, np.zeros((2, 2), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_context_manager_opens_and_releases():
    cam = DummyCamera()
    with cam as c:
        assert c is cam
        assert cam.opened is True
    assert cam.released is True


def test_read_before_open_raises():
    cam = RealCamera(device_id=0)
    with pytest.raises(RuntimeError):
        cam.read()


def test_real_camera_open_failure_is_clear(monkeypatch):
    """Opening a non-existent device raises a clear RuntimeError."""
    import motion_pipeline.camera.camera_interface as ci

    class FakeCapture:
        def __init__(self, *_):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    if ci.cv2 is None:
        pytest.skip("OpenCV not installed")
    monkeypatch.setattr(ci.cv2, "VideoCapture", FakeCapture)
    with pytest.raises(RuntimeError, match="Cannot open camera"):
        RealCamera(device_id=99).open()


def test_real_camera_without_opencv(monkeypatch):
    import motion_pipeline.camera.camera_interface as ci

    monkeypatch.setattr(ci, "cv2", None)
    with pytest.raises(RuntimeError, match="OpenCV is not installed"):
        RealCamera().open()


def test_real_camera_full_lifecycle(monkeypatch):
    """Drive open/read/release success paths through a fake VideoCapture."""
    from types import SimpleNamespace

    import motion_pipeline.camera.camera_interface as ci

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self, device_id):
            self.device_id = device_id
            self.released = False
            self._props = {}

        def isOpened(self):
            return True

        def set(self, prop, value):
            self._props[prop] = value

        def read(self):
            return True, frame

        def release(self):
            self.released = True

    fake_cv2 = SimpleNamespace(
        VideoCapture=FakeCapture,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FRAME_WIDTH=3,
    )
    monkeypatch.setattr(ci, "cv2", fake_cv2)

    cam = RealCamera(device_id=1, resolution=(480, 640))
    cam.open()
    ok, out = cam.read()
    assert ok is True
    np.testing.assert_array_equal(out, frame)
    cam.release()
    assert cam._capture is None


def test_real_camera_read_returns_false(monkeypatch):
    from types import SimpleNamespace

    import motion_pipeline.camera.camera_interface as ci

    class FakeCapture:
        def __init__(self, *_):
            pass

        def isOpened(self):
            return True

        def set(self, *_):
            pass

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(
        ci,
        "cv2",
        SimpleNamespace(VideoCapture=FakeCapture, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FRAME_WIDTH=3),
    )
    cam = RealCamera()
    cam.open()
    ok, out = cam.read()
    assert ok is False and out is None
