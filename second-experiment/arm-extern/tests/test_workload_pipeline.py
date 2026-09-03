"""End-to-end pipeline runs at realistic frame sizes.

These are the bulk of the Orin's occupancy: they stream a fixed pixel budget's worth of frames
through camera -> grayscale -> blur -> diff -> threshold on the device and assert
invariants, without ever computing a NumPy reference. Correctness of the kernels is
established in ``test_kernels.py``; what these check is that the whole pipeline holds
together over a realistic stream, at a realistic size, for every parameter
combination the CLI exposes.

Their dependency footprint is deliberately the widest in the suite — camera,
pipeline, detector, processor and kernels — which is what makes them the tests a
selection strategy has the most to gain (and the most to lose) by skipping.
"""

from __future__ import annotations

import pytest
import workload
from motion_pipeline.pipeline import Pipeline
from motion_pipeline.processing.motion_detector import MotionDetector

pytestmark = pytest.mark.gpu


def _run(processor, resolution, channels, blur_kernel, threshold, frames=None):
    # Frame count is derived from a fixed pixel budget so every test costs the same
    # regardless of resolution; see tests/workload.py for why that matters.
    num_frames = frames if frames is not None else workload.frames_for(resolution)
    # motion_frames is fixed by CachedCamera's unique set, not by the stream length.
    camera = workload.CachedCamera(
        resolution=resolution,
        num_frames=num_frames,
        channels=channels,
        seed=17,
    )
    detector = MotionDetector(processor, threshold=threshold, min_area=500, blur_kernel=blur_kernel)
    pipeline = Pipeline(camera, detector)
    results = [result for _, result in pipeline.run()]
    return pipeline.stats, results


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
@pytest.mark.parametrize("channels", workload.CHANNELS)
@pytest.mark.parametrize("blur_kernel", workload.BLUR_KERNELS)
def test_pipeline_streams_every_frame(cuda_processor, resolution, channels, blur_kernel):
    stats, results = _run(cuda_processor, resolution, channels, blur_kernel, threshold=25)
    expected = workload.frames_for(resolution)
    assert stats.total_frames == expected
    assert len(results) == expected
    assert stats.backend == "pycuda"


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
@pytest.mark.parametrize("threshold", workload.THRESHOLDS)
def test_pipeline_detects_the_injected_motion(cuda_processor, resolution, threshold):
    stats, _ = _run(cuda_processor, resolution, channels=3, blur_kernel=5, threshold=threshold)
    # The stream is deterministic and the burst schedule is known, so bound the count
    # on both sides. Three bursts per cycle over the unique set, each producing an
    # onset and an offset, gives the floor. The ceiling matters as much: a detector
    # regression that flags every frame would otherwise pass this test.
    #
    # Both bounds were checked on the NumPy path, which the kernels agree with to
    # ~1e-5, at every resolution and threshold in the grid. The count comes out
    # exactly 2*3*cycles on an exact multiple of UNIQUE_FRAMES and above it on a
    # partial cycle. The two regimes are four orders of magnitude apart: a frame that
    # fires has at least 34_826 changed pixels against min_area=500, and a static one
    # never exceeds 2. Float32 rounding cannot move a frame across that.
    cycles = stats.total_frames // workload.UNIQUE_FRAMES
    assert stats.motion_events >= 2 * 3 * cycles
    assert stats.motion_events < stats.total_frames // 2


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
def test_pipeline_masks_match_the_frame_geometry(cuda_processor, resolution):
    _, results = _run(cuda_processor, resolution, channels=3, blur_kernel=5, threshold=25)
    for result in results:
        assert result.mask.shape == resolution
        assert result.mask.dtype.kind == "u"


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
def test_pipeline_first_frame_reports_no_motion(cuda_processor, resolution):
    _, results = _run(cuda_processor, resolution, channels=3, blur_kernel=5, threshold=25)
    assert results[0].motion_detected is False
    assert results[0].diff_score == 0.0


@pytest.mark.parametrize("resolution", workload.RESOLUTIONS)
@pytest.mark.parametrize("channels", workload.CHANNELS)
def test_pipeline_is_deterministic_across_runs(cuda_processor, resolution, channels):
    """Same seed, same stream, same verdicts — the study depends on reproducibility."""
    _, first = _run(cuda_processor, resolution, channels, blur_kernel=5, threshold=25)
    _, second = _run(cuda_processor, resolution, channels, blur_kernel=5, threshold=25)
    assert [r.motion_detected for r in first] == [r.motion_detected for r in second]
    assert [r.diff_score for r in first] == [r.diff_score for r in second]
