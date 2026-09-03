"""Unit tests for the simulated camera."""

from __future__ import annotations

import numpy as np
import pytest
from motion_pipeline.simulation.data_stream import SimulatedCamera


def test_deterministic_with_seed():
    a = SimulatedCamera(resolution=(50, 60), num_frames=5, seed=1)
    b = SimulatedCamera(resolution=(50, 60), num_frames=5, seed=1)
    a.open()
    b.open()
    for _ in range(5):
        oka, fa = a.read()
        okb, fb = b.read()
        assert oka and okb
        np.testing.assert_array_equal(fa, fb)


def test_exhaustion_returns_false():
    cam = SimulatedCamera(resolution=(10, 10), num_frames=3)
    cam.open()
    for _ in range(3):
        ok, frame = cam.read()
        assert ok and frame is not None
    ok, frame = cam.read()
    assert ok is False and frame is None


def test_frame_shape_and_dtype_rgb():
    cam = SimulatedCamera(resolution=(40, 30), num_frames=1, channels=3)
    cam.open()
    ok, frame = cam.read()
    assert ok
    assert frame.shape == (40, 30, 3)
    assert frame.dtype == np.uint8


def test_grayscale_shape():
    cam = SimulatedCamera(resolution=(40, 30), num_frames=1, channels=1)
    cam.open()
    _, frame = cam.read()
    assert frame.shape == (40, 30)


def test_motion_frame_is_brighter():
    cam = SimulatedCamera(resolution=(60, 60), num_frames=5, noise_level=1.0, motion_frames=[2])
    cam.open()
    frames = [cam.read()[1] for _ in range(5)]
    center = np.s_[20:40, 20:40]
    assert frames[2][center].mean() > frames[0][center].mean()


def test_invalid_channels():
    with pytest.raises(ValueError):
        SimulatedCamera(channels=2)


def test_release_resets():
    cam = SimulatedCamera(num_frames=2)
    cam.open()
    cam.read()
    cam.release()
    assert cam._index == 0
    assert cam._frames == []
