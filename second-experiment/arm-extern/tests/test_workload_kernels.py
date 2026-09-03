"""Kernel equivalence at realistic frame sizes.

``test_kernels.py`` establishes numerical equivalence on a small, deliberately
tile-misaligned frame. This module repeats it at the resolutions the pipeline
actually runs, where the grid spans many blocks and rounding accumulates over far
more pixels. Same tolerances: they are stated per-pixel, so they do not loosen with
frame size.

Only ``REFERENCE_RESOLUTIONS`` are covered. The NumPy reference convolves with a
Python loop over the kernel taps, so a 720p run at 15 taps would spend most of its
time on the host and measure the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest
import workload
from motion_pipeline.processing.gpu_processor import GPUProcessor
from test_kernels import ATOL_ABSDIFF, ATOL_BLUR, ATOL_GRAYSCALE

from conftest import assert_within

pytestmark = pytest.mark.gpu


@pytest.fixture
def reference() -> GPUProcessor:
    return GPUProcessor(force_cpu=True)


def _frame(resolution: tuple[int, int], channels: int = 3, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = resolution if channels == 1 else (*resolution, channels)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


@pytest.mark.parametrize("resolution", workload.REFERENCE_RESOLUTIONS)
def test_grayscale_matches_numpy_at_scale(cuda_processor, reference, resolution):
    frame = _frame(resolution)
    gpu = cuda_processor.to_host(cuda_processor.convert_grayscale(cuda_processor.to_device(frame)))
    ref = reference.to_host(reference.convert_grayscale(frame))
    assert_within(gpu, ref, ATOL_GRAYSCALE, f"grayscale {resolution}")


@pytest.mark.parametrize("resolution", workload.REFERENCE_RESOLUTIONS)
@pytest.mark.parametrize("kernel_size", workload.BLUR_KERNELS)
def test_gaussian_blur_matches_numpy_at_scale(cuda_processor, reference, resolution, kernel_size):
    gray = _frame(resolution, channels=1).astype(np.float32)
    gpu = cuda_processor.to_host(
        cuda_processor.gaussian_blur(cuda_processor.to_device(gray), kernel_size)
    )
    ref = reference.to_host(reference.gaussian_blur(gray, kernel_size))
    assert_within(gpu, ref, ATOL_BLUR, f"blur k={kernel_size} {resolution}")


@pytest.mark.parametrize("resolution", workload.REFERENCE_RESOLUTIONS)
def test_absolute_diff_matches_numpy_at_scale(cuda_processor, reference, resolution):
    a = _frame(resolution, channels=1, seed=1).astype(np.float32)
    b = _frame(resolution, channels=1, seed=2).astype(np.float32)
    gpu = cuda_processor.to_host(
        cuda_processor.absolute_diff(cuda_processor.to_device(a), cuda_processor.to_device(b))
    )
    ref = reference.to_host(reference.absolute_diff(a, b))
    assert_within(gpu, ref, ATOL_ABSDIFF, f"absdiff {resolution}")


@pytest.mark.parametrize("resolution", workload.REFERENCE_RESOLUTIONS)
@pytest.mark.parametrize("threshold", workload.THRESHOLDS)
def test_threshold_matches_numpy_at_scale(cuda_processor, reference, resolution, threshold):
    arr = _frame(resolution, channels=1).astype(np.float32)
    gpu = cuda_processor.to_host(
        cuda_processor.threshold(cuda_processor.to_device(arr), float(threshold))
    )
    ref = reference.to_host(reference.threshold(arr, float(threshold)))
    np.testing.assert_array_equal(gpu, ref)
