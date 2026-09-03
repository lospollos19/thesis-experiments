"""Real-time OpenCV display with motion overlay.

This is the only module allowed to call ``cv2.imshow``. It degrades gracefully:
in a headless environment (no GUI backend) :meth:`show` raises a clear
``RuntimeError` rather than a cryptic OpenCV error.
"""

from __future__ import annotations

import contextlib
import time

import numpy as np

from motion_pipeline.processing.motion_detector import MotionResult

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV always present via extras
    cv2 = None  # type: ignore[assignment]

_HEADLESS_MSG = "Display not available in headless mode"


class Display:
    """Renders frames with a motion overlay in an OpenCV window."""

    def __init__(self, window_name: str = "Motion Pipeline", backend: str = "NumPy") -> None:
        self.window_name = window_name
        self.backend = backend
        self._window_created = False
        self._last_time: float | None = None

    def open(self) -> None:
        if cv2 is None:
            raise RuntimeError(_HEADLESS_MSG)
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:  # pragma: no cover - needs headless OpenCV
            raise RuntimeError(_HEADLESS_MSG) from exc
        self._window_created = True

    def _fps(self) -> float:
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return 0.0
        dt = now - self._last_time
        self._last_time = now
        return 1.0 / dt if dt > 0 else 0.0

    def show(self, frame: np.ndarray, result: MotionResult) -> bool:
        """Render ``frame`` with overlay. Returns ``False`` if user pressed ``q``."""
        if cv2 is None:
            raise RuntimeError(_HEADLESS_MSG)
        if not self._window_created:
            self.open()

        canvas = frame.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        self._draw_contours(canvas, result)
        self._draw_hud(canvas, result)

        try:
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:  # pragma: no cover - needs headless OpenCV
            raise RuntimeError(_HEADLESS_MSG) from exc
        return key != ord("q")

    def _draw_contours(self, canvas: np.ndarray, result: MotionResult) -> None:
        if result.mask is None or not result.mask.any():
            return
        contours, _ = cv2.findContours(
            result.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def _draw_hud(self, canvas: np.ndarray, result: MotionResult) -> None:
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        status = "MOTION" if result.motion_detected else "STATIC"
        color = (0, 0, 255) if result.motion_detected else (0, 255, 0)
        cv2.putText(canvas, status, (10, 30), font, 0.8, color, 2)
        cv2.putText(
            canvas, f"diff: {result.diff_score:.1f}", (10, 60), font, 0.6, (255, 255, 255), 1
        )

        fps = self._fps()
        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, h - 15), font, 0.6, (255, 255, 255), 1)

        label = self.backend
        (tw, _), _ = cv2.getTextSize(label, font, 0.6, 1)
        cv2.putText(canvas, label, (w - tw - 10, 30), font, 0.6, (200, 200, 0), 2)

    def release(self) -> None:
        if cv2 is not None and self._window_created:
            with contextlib.suppress(cv2.error):  # pragma: no cover
                cv2.destroyWindow(self.window_name)
            self._window_created = False

    def __enter__(self) -> Display:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
