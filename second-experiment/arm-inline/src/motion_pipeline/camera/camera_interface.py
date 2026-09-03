"""Camera abstraction: abstract BaseCamera + OpenCV-backed RealCamera."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:  # OpenCV is optional (headless CI may install headless variant, still importable)
    import cv2
except ImportError:  # pragma: no cover - exercised only when OpenCV missing
    cv2 = None  # type: ignore[assignment]


class BaseCamera(ABC):
    """Abstract camera source.

    Concrete cameras implement :meth:`open`, :meth:`read` and :meth:`release`.
    Supports use as a context manager so callers always release resources.
    """

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource (device, generator, ...)."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return ``(ok, frame)``. ``ok`` is ``False`` when no frame is available."""

    @abstractmethod
    def release(self) -> None:
        """Release the underlying resource."""

    def __enter__(self) -> BaseCamera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class RealCamera(BaseCamera):
    """Wraps an OpenCV ``VideoCapture``.

    Handles a missing OpenCV import or missing hardware gracefully by raising a
    clear :class:`RuntimeError` instead of a cryptic OpenCV failure.
    """

    def __init__(self, device_id: int = 0, resolution: tuple[int, int] = (480, 640)) -> None:
        self.device_id = device_id
        self.resolution = resolution  # (height, width)
        self._capture: Any = None

    def open(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed; cannot use RealCamera")

        capture = cv2.VideoCapture(self.device_id)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open camera device {self.device_id}")

        height, width = self.resolution
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture = capture

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is None:
            raise RuntimeError("Camera is not open; call open() first")
        ok, frame = self._capture.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
