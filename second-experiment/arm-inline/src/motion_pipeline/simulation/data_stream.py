"""Synthetic camera producing deterministic frames for hardware-free testing."""

from __future__ import annotations

import numpy as np

from motion_pipeline.camera.camera_interface import BaseCamera


class SimulatedCamera(BaseCamera):
    """Deterministic synthetic frame source.

    Generates frames of gaussian noise. On indices listed in ``motion_frames`` a
    bright rectangle is drawn to simulate a motion event. Given a fixed ``seed``
    the whole stream is reproducible, which tests rely on.

    Parameters
    ----------
    resolution:
        ``(height, width)`` of generated frames. Default ``(480, 640)``.
    num_frames:
        Number of frames the stream yields before exhaustion.
    noise_level:
        Standard deviation of the background gaussian noise (0-255).
    motion_frames:
        Frame indices where a bright rectangle is injected.
    channels:
        ``3`` for RGB (default) or ``1`` for grayscale.
    seed:
        RNG seed for deterministic output.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (480, 640),
        num_frames: int = 30,
        noise_level: float = 10.0,
        motion_frames: list[int] | None = None,
        channels: int = 3,
        seed: int = 42,
    ) -> None:
        if channels not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        self.resolution = resolution
        self.num_frames = num_frames
        self.noise_level = noise_level
        self.motion_frames = set(motion_frames or [])
        self.channels = channels
        self.seed = seed

        self._frames: list[np.ndarray] = []
        self._index = 0

    def _generate(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        height, width = self.resolution
        shape = (height, width) if self.channels == 1 else (height, width, self.channels)
        base_level = 40.0

        frames: list[np.ndarray] = []
        for i in range(self.num_frames):
            noise = rng.normal(base_level, self.noise_level, size=shape)
            frame = np.clip(noise, 0, 255).astype(np.uint8)
            if i in self.motion_frames:
                y0, y1 = height // 3, (2 * height) // 3
                x0, x1 = width // 3, (2 * width) // 3
                frame[y0:y1, x0:x1] = 255
            frames.append(frame)
        return frames

    def open(self) -> None:
        self._frames = self._generate()
        self._index = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self._frames = []
        self._index = 0
