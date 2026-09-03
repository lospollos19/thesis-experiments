"""Pipeline orchestrator: camera -> motion detector -> per-frame results.

Depends only on the :class:`BaseCamera` and :class:`MotionDetector` abstractions
and never touches display/UI code, keeping it fully headless-testable.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from motion_pipeline.camera.camera_interface import BaseCamera
from motion_pipeline.processing.motion_detector import MotionDetector, MotionResult


@dataclass
class PipelineStats:
    total_frames: int
    motion_events: int
    avg_processing_ms: float
    backend: str


class Pipeline:
    """Ties a camera to a motion detector and streams results."""

    def __init__(self, camera: BaseCamera, detector: MotionDetector) -> None:
        self.camera = camera
        self.detector = detector
        self._total_frames = 0
        self._motion_events = 0
        self._total_ms = 0.0

    def run(self, max_frames: int = 0) -> Iterator[tuple[np.ndarray, MotionResult]]:
        """Yield ``(frame, result)`` per frame.

        ``max_frames <= 0`` runs until the camera is exhausted. The camera is
        opened on entry and released when the generator finishes or is closed.
        """
        self._total_frames = 0
        self._motion_events = 0
        self._total_ms = 0.0

        self.camera.open()
        try:
            while max_frames <= 0 or self._total_frames < max_frames:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    break

                start = time.perf_counter()
                result = self.detector.detect(frame)
                self._total_ms += (time.perf_counter() - start) * 1000.0

                self._total_frames += 1
                if result.motion_detected:
                    self._motion_events += 1

                yield frame, result
        finally:
            self.camera.release()

    @property
    def stats(self) -> PipelineStats:
        avg = self._total_ms / self._total_frames if self._total_frames else 0.0
        return PipelineStats(
            total_frames=self._total_frames,
            motion_events=self._motion_events,
            avg_processing_ms=avg,
            backend=self.detector.processor.backend,
        )
