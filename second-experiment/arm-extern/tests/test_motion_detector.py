"""Unit tests for the motion detection logic."""

from __future__ import annotations

import numpy as np
from motion_pipeline.processing.motion_detector import MotionDetector, MotionResult


def _frame(value: int, shape=(80, 80, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_first_frame_never_detects(gpu_processor):
    det = MotionDetector(gpu_processor)
    result = det.detect(_frame(30))
    assert isinstance(result, MotionResult)
    assert result.motion_detected is False
    assert result.diff_score == 0.0
    assert result.mask.shape == (80, 80)


def test_static_frames_no_motion(gpu_processor):
    det = MotionDetector(gpu_processor, threshold=25, min_area=50)
    det.detect(_frame(30))
    result = det.detect(_frame(30))
    assert result.motion_detected is False


def test_large_change_triggers_motion(gpu_processor):
    det = MotionDetector(gpu_processor, threshold=25, min_area=50)
    det.detect(_frame(0))
    result = det.detect(_frame(255))
    assert result.motion_detected is True
    assert result.diff_score > 0
    assert result.mask.max() == 255


def test_small_change_below_min_area(gpu_processor):
    det = MotionDetector(gpu_processor, threshold=25, min_area=10_000)
    det.detect(_frame(0))
    frame = _frame(0)
    frame[0:3, 0:3] = 255  # tiny bright patch
    result = det.detect(frame)
    assert result.motion_detected is False


def test_reset_clears_previous(gpu_processor):
    det = MotionDetector(gpu_processor)
    det.detect(_frame(30))
    det.reset()
    result = det.detect(_frame(200))
    assert result.motion_detected is False  # treated as first frame again


def test_detect_over_simulated_stream(simulated_camera, gpu_processor):
    det = MotionDetector(gpu_processor, threshold=25, min_area=100)
    simulated_camera.open()
    events = 0
    while True:
        ok, frame = simulated_camera.read()
        if not ok:
            break
        if det.detect(frame).motion_detected:
            events += 1
    assert events > 0  # motion injected on frames 10-15
