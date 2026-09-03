"""Tests for the Display overlay.

Per the project constraints Display is never instantiated against a real GUI: we
exercise the headless RuntimeError path and drive the drawing logic through a
fake ``cv2`` module so no window is ever opened.
"""

from __future__ import annotations

from types import SimpleNamespace

import motion_pipeline.ui.display as display_mod
import numpy as np
import pytest
from motion_pipeline.processing.motion_detector import MotionResult
from motion_pipeline.ui.display import Display


def _result(motion: bool) -> MotionResult:
    mask = np.zeros((20, 20), dtype=np.uint8)
    if motion:
        mask[5:10, 5:10] = 255
    return MotionResult(motion_detected=motion, diff_score=12.5, mask=mask)


def _fake_cv2(waitkey_return: int) -> SimpleNamespace:
    """Minimal stand-in for the OpenCV API surface Display touches."""
    calls: list[str] = []

    def find_contours(mask, mode, method):
        contour = np.array([[[5, 5]], [[9, 9]]], dtype=np.int32)
        return [contour], None

    ns = SimpleNamespace(
        error=type("error", (Exception,), {}),
        WINDOW_NORMAL=0,
        COLOR_GRAY2BGR=8,
        RETR_EXTERNAL=0,
        CHAIN_APPROX_SIMPLE=2,
        FONT_HERSHEY_SIMPLEX=0,
        namedWindow=lambda *a, **k: calls.append("namedWindow"),
        cvtColor=lambda img, code: np.dstack([img] * 3),
        findContours=find_contours,
        boundingRect=lambda c: (5, 5, 4, 4),
        rectangle=lambda *a, **k: calls.append("rectangle"),
        putText=lambda *a, **k: calls.append("putText"),
        getTextSize=lambda *a, **k: ((30, 10), 2),
        imshow=lambda *a, **k: calls.append("imshow"),
        waitKey=lambda n: waitkey_return,
        destroyWindow=lambda *a, **k: calls.append("destroyWindow"),
    )
    ns._calls = calls
    return ns


def test_headless_open_raises(monkeypatch):
    monkeypatch.setattr(display_mod, "cv2", None)
    with pytest.raises(RuntimeError, match="Display not available in headless mode"):
        Display().open()


def test_headless_show_raises(monkeypatch):
    monkeypatch.setattr(display_mod, "cv2", None)
    with pytest.raises(RuntimeError, match="Display not available in headless mode"):
        Display().show(np.zeros((4, 4, 3), np.uint8), _result(False))


def test_show_returns_true_when_not_quit(monkeypatch):
    fake = _fake_cv2(waitkey_return=ord("x"))
    monkeypatch.setattr(display_mod, "cv2", fake)
    frame = np.zeros((20, 20, 3), np.uint8)
    with Display(backend="NumPy") as disp:
        assert disp.show(frame, _result(motion=True)) is True
        # Second frame drives the FPS branch (last_time already set).
        assert disp.show(frame, _result(motion=False)) is True
    assert "imshow" in fake._calls
    assert "rectangle" in fake._calls  # motion -> bounding box drawn
    assert "destroyWindow" in fake._calls


def test_show_returns_false_on_quit(monkeypatch):
    fake = _fake_cv2(waitkey_return=ord("q"))
    monkeypatch.setattr(display_mod, "cv2", fake)
    disp = Display()
    assert disp.show(np.zeros((20, 20, 3), np.uint8), _result(False)) is False


def test_show_converts_grayscale_frame(monkeypatch):
    fake = _fake_cv2(waitkey_return=ord("x"))
    monkeypatch.setattr(display_mod, "cv2", fake)
    disp = Display()
    gray = np.zeros((20, 20), np.uint8)  # 2-D triggers cvtColor path
    assert disp.show(gray, _result(False)) is True
