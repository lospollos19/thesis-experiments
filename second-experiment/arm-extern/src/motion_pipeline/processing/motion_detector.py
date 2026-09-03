"""Frame-differencing motion detector built on top of :class:`GPUProcessor`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motion_pipeline.processing.gpu_processor import GPUProcessor


@dataclass
class MotionResult:
    """Outcome of a single :meth:`MotionDetector.detect` call.

    ``mask`` is a host (NumPy) uint8 array (0/255) sized like the input frame.
    """

    motion_detected: bool
    diff_score: float
    mask: np.ndarray


class MotionDetector:
    """Detects motion by thresholding the blurred grayscale frame difference.

    Parameters
    ----------
    processor:
        Backend used for all array math.
    threshold:
        Per-pixel intensity difference above which a pixel counts as changed.
    min_area:
        Minimum number of changed pixels for a frame to count as motion.
    blur_kernel:
        Gaussian blur kernel size applied before differencing (noise robustness).
    """

    def __init__(
        self,
        processor: GPUProcessor,
        threshold: int = 25,
        min_area: int = 500,
        blur_kernel: int = 5,
    ) -> None:
        self.processor = processor
        self.threshold = threshold
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self._previous = None  # blurred grayscale device array

    def _preprocess(self, frame: np.ndarray):
        dev = self.processor.to_device(frame)
        gray = self.processor.convert_grayscale(dev)
        return self.processor.gaussian_blur(gray, self.blur_kernel)

    def detect(self, frame: np.ndarray) -> MotionResult:
        current = self._preprocess(frame)

        if self._previous is None:
            self._previous = current
            zero_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            return MotionResult(motion_detected=False, diff_score=0.0, mask=zero_mask)

        diff = self.processor.absolute_diff(current, self._previous)
        mask = self.processor.threshold(diff, self.threshold)

        changed_pixels = float(self.processor.to_host((mask > 0).sum()))
        diff_score = float(self.processor.to_host(diff.mean()))
        motion = changed_pixels >= self.min_area

        self._previous = current
        return MotionResult(
            motion_detected=motion,
            diff_score=diff_score,
            mask=self.processor.to_host(mask).astype(np.uint8),
        )

    def reset(self) -> None:
        """Forget the previous frame (next detect returns no motion)."""
        self._previous = None
