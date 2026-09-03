"""Sustained device-only load on one primitive at a time.

These carry the same weight as the pipeline tests but the narrowest dependency
footprint in the GPU suite: the processor and the kernels, nothing else. No camera,
no detector, no pipeline, and no NumPy reference.

That separation is the point. With load concentrated in the pipeline tests, every
change to the processor or the kernels re-selects essentially the whole runtime and
a selection strategy has nothing to show; conversely a change to an unrelated module
skips all of it. Both outcomes measure the shape of this suite rather than the
strategy. Giving the narrow tier comparable weight is what makes the middle of that
range reachable.

Each test uploads once and then loops on the device, so what is measured is kernel
throughput rather than PCIe or, on Tegra, unified-memory traffic.
"""

from __future__ import annotations

import numpy as np
import pytest
import workload

pytestmark = pytest.mark.gpu

#: Slack for the float32 rounding the kernels introduce, when bounding outputs.
ATOL = 1e-3


def _device_frame(processor, resolution, channels=1, seed=3):
    rng = np.random.default_rng(seed)
    shape = resolution if channels == 1 else (*resolution, channels)
    return processor.to_device(rng.integers(0, 256, size=shape, dtype=np.uint8))


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
def test_grayscale_sustained(cuda_processor, resolution):
    src = _device_frame(cuda_processor, resolution, channels=3)
    for _ in range(workload.frames_for(resolution)):
        out = cuda_processor.convert_grayscale(src)
    host = cuda_processor.to_host(out)
    assert host.shape == resolution
    # Luminance weights sum to 1, so a uint8 input cannot leave [0, 255].
    assert np.isfinite(host).all()
    assert host.min() >= 0.0 and host.max() <= 255.0


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
@pytest.mark.parametrize("kernel_size", workload.BLUR_KERNELS)
def test_gaussian_blur_sustained(cuda_processor, resolution, kernel_size):
    src = _device_frame(cuda_processor, resolution)
    for _ in range(workload.frames_for(resolution)):
        out = cuda_processor.gaussian_blur(src, kernel_size)
    host = cuda_processor.to_host(out)
    src_host = cuda_processor.to_host(src)
    assert host.shape == resolution
    # Normalised positive weights and clamped borders make every output a convex
    # combination of input pixels, so it cannot escape the input's range.
    assert np.isfinite(host).all()
    assert host.min() >= src_host.min() - ATOL and host.max() <= src_host.max() + ATOL


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
def test_absolute_diff_sustained(cuda_processor, resolution):
    a = _device_frame(cuda_processor, resolution, seed=1)
    b = _device_frame(cuda_processor, resolution, seed=2)
    for _ in range(workload.frames_for(resolution)):
        out = cuda_processor.absolute_diff(a, b)
    host = cuda_processor.to_host(out)
    assert host.shape == resolution
    assert np.isfinite(host).all()
    assert host.min() >= 0.0


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
@pytest.mark.parametrize("threshold", workload.THRESHOLDS)
def test_threshold_sustained(cuda_processor, resolution, threshold):
    src = _device_frame(cuda_processor, resolution)
    for _ in range(workload.frames_for(resolution)):
        out = cuda_processor.threshold(src, float(threshold))
    host = cuda_processor.to_host(out)
    assert host.shape == resolution
    assert set(np.unique(host)) <= {0, 255}


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
def test_full_primitive_chain_sustained(cuda_processor, resolution):
    """The four primitives back to back, still without leaving the device.

    Two distinct sources, alternated. Running the chain against a single source
    would make absolute_diff identically zero and leave threshold classifying a
    field of zeros for the whole loop, so two of the four primitives would be
    exercised on degenerate input.
    """
    sources = [
        _device_frame(cuda_processor, resolution, channels=3, seed=7),
        _device_frame(cuda_processor, resolution, channels=3, seed=8),
    ]
    # Seeded from the second source so that iteration 0, which uses the first, already
    # has something to differ from.
    previous = cuda_processor.gaussian_blur(cuda_processor.convert_grayscale(sources[1]), 5)
    for i in range(workload.frames_for(resolution)):
        current = cuda_processor.gaussian_blur(cuda_processor.convert_grayscale(sources[i % 2]), 5)
        mask = cuda_processor.threshold(cuda_processor.absolute_diff(current, previous), 25.0)
        previous = current
    host = cuda_processor.to_host(mask)
    assert host.shape == resolution
    assert set(np.unique(host)) <= {0, 255}
    # Two independent noise fields differ well above the threshold nearly everywhere;
    # an all-zero mask would mean the chain collapsed to comparing a frame with itself.
    assert host.any()
