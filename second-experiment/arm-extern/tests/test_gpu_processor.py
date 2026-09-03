"""Unit tests for the GPU processor (NumPy fallback path)."""

from __future__ import annotations

import numpy as np


def test_backend_is_numpy_when_forced(gpu_processor):
    assert gpu_processor.backend == "numpy"


def test_to_device_and_host_roundtrip(gpu_processor, sample_frame):
    dev = gpu_processor.to_device(sample_frame)
    host = gpu_processor.to_host(dev)
    np.testing.assert_array_equal(host, sample_frame)


def test_convert_grayscale_reduces_channels(gpu_processor, sample_frame):
    gray = gpu_processor.convert_grayscale(gpu_processor.to_device(sample_frame))
    assert gray.ndim == 2
    assert gray.shape == sample_frame.shape[:2]


def test_convert_grayscale_passthrough_2d(gpu_processor):
    gray_in = np.full((10, 10), 100, dtype=np.uint8)
    out = gpu_processor.convert_grayscale(gray_in)
    assert out.shape == (10, 10)


def test_gaussian_blur_smooths(gpu_processor):
    arr = np.zeros((21, 21), dtype=np.float32)
    arr[10, 10] = 255.0
    blurred = gpu_processor.to_host(gpu_processor.gaussian_blur(arr, kernel_size=5))
    assert blurred[10, 10] < 255.0
    assert blurred[9, 10] > 0.0  # energy spread to neighbours


def test_gaussian_blur_rejects_even_kernel(gpu_processor):
    import pytest

    with pytest.raises(ValueError):
        gpu_processor.gaussian_blur(np.zeros((5, 5), dtype=np.float32), kernel_size=4)


def test_gaussian_blur_identity_for_kernel_one(gpu_processor):
    arr = np.ones((4, 4), dtype=np.uint8)
    out = gpu_processor.to_host(gpu_processor.gaussian_blur(arr, kernel_size=1))
    np.testing.assert_allclose(out, arr.astype(np.float32))


def test_absolute_diff(gpu_processor):
    a = np.full((4, 4), 200, dtype=np.uint8)
    b = np.full((4, 4), 50, dtype=np.uint8)
    diff = gpu_processor.to_host(gpu_processor.absolute_diff(a, b))
    np.testing.assert_allclose(diff, np.full((4, 4), 150.0))


def test_threshold(gpu_processor):
    arr = np.array([[10, 30], [40, 5]], dtype=np.float32)
    mask = gpu_processor.to_host(gpu_processor.threshold(arr, 25))
    assert mask.tolist() == [[0, 255], [255, 0]]
