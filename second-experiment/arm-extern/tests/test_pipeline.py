"""Integration test: full pipeline run without any hardware."""

from __future__ import annotations

from motion_pipeline.pipeline import Pipeline, PipelineStats
from motion_pipeline.processing.motion_detector import MotionDetector


def test_pipeline_runs_full_stream(simulated_camera, gpu_processor):
    detector = MotionDetector(gpu_processor, threshold=25, min_area=100)
    pipeline = Pipeline(simulated_camera, detector)

    frames = list(pipeline.run(max_frames=0))
    assert len(frames) == simulated_camera.num_frames

    stats = pipeline.stats
    assert isinstance(stats, PipelineStats)
    assert stats.total_frames == simulated_camera.num_frames
    assert stats.motion_events > 0
    assert stats.avg_processing_ms >= 0.0
    assert stats.backend == "numpy"


def test_pipeline_respects_max_frames(simulated_camera, gpu_processor):
    detector = MotionDetector(gpu_processor)
    pipeline = Pipeline(simulated_camera, detector)
    frames = list(pipeline.run(max_frames=5))
    assert len(frames) == 5
    assert pipeline.stats.total_frames == 5


def test_pipeline_releases_camera(simulated_camera, gpu_processor):
    detector = MotionDetector(gpu_processor)
    pipeline = Pipeline(simulated_camera, detector)
    list(pipeline.run(max_frames=3))
    # release() resets the simulated camera's internal index
    assert simulated_camera._index == 0
