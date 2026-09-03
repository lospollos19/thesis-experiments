"""Frame primitives backed by CUDA-C kernels, with a transparent NumPy fallback.

Three modes are supported explicitly:

1. PyCUDA importable **and** a CUDA device present -> the CUDA-C kernels of
   :mod:`motion_pipeline.processing.kernels` are compiled and used (``backend ==
   "pycuda"``);
2. otherwise -> the NumPy reference path (``backend == "numpy"``);
3. ``force_cpu=True`` -> the NumPy path unconditionally.

The NumPy path is the numerical reference: it is the one exercised in CI and the one
the GPU equivalence tests compare against. All operations accept either host NumPy
arrays or device handles returned by :meth:`GPUProcessor.to_device`, so callers do not
need to know which backend is active.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from motion_pipeline.processing import _trace, kernels

try:  # PyCUDA is optional and only present on CUDA machines
    import pycuda.driver as _cuda
    import pycuda.gpuarray as _gpuarray
except ImportError:  # pragma: no cover - PyCUDA absent in CI
    _cuda = None
    _gpuarray = None


def _pycuda_available() -> bool:
    """True when PyCUDA is importable and at least one CUDA device is visible."""
    if _cuda is None:
        return False
    try:  # pragma: no cover - requires a real GPU
        _cuda.init()
        return _cuda.Device.count() > 0
    except Exception:  # pragma: no cover - driver present but no usable device
        return False


class DeviceArray:  # pragma: no cover - only instantiated on the PyCUDA path
    """Thin handle over a ``pycuda.gpuarray.GPUArray``.

    It exists so that device arrays expose the small slice of the NumPy API the
    detector relies on (``ndim``, ``astype``, comparison, ``sum``, ``mean``) without
    depending on which reductions a given PyCUDA release happens to ship. Reductions
    and comparisons round-trip through the host: they are not part of the four
    kernelised primitives and run once per frame on already-small results.
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: Any) -> None:
        self._arr = arr

    @property
    def raw(self) -> Any:
        """The underlying ``GPUArray``."""
        return self._arr

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    @property
    def ndim(self) -> int:
        return len(self._arr.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._arr.dtype

    def get(self) -> np.ndarray:
        return self._arr.get()

    def astype(self, dtype: Any) -> DeviceArray:
        if self._arr.dtype == np.dtype(dtype):
            return self
        return DeviceArray(self._arr.astype(np.dtype(dtype)))

    def __gt__(self, other: Any) -> np.ndarray:
        return self.get() > other

    def sum(self) -> Any:
        return self.get().sum()

    def mean(self) -> Any:
        return self.get().mean()


class GPUProcessor:
    """Array primitives used by the motion detector.

    Attributes
    ----------
    backend:
        ``"pycuda"`` when the CUDA-C kernels are in use, otherwise ``"numpy"``.
    """

    def __init__(self, force_cpu: bool = False) -> None:
        self._xp = np
        self._backend = "numpy"
        #: kernel name -> its own SourceModule (empty on the NumPy path).
        self._module: dict[str, Any] = {}
        self._fn: dict[str, Any] = {}

        if not force_cpu and _pycuda_available():  # pragma: no cover - needs GPU
            self._init_pycuda()

    def _init_pycuda(self) -> None:  # pragma: no cover - needs GPU
        """Enter a CUDA context and compile the kernel module.

        Any failure here (no driver, nvcc missing, compile error) degrades silently to
        the NumPy path rather than breaking the pipeline.
        """
        try:
            # Imported for its side effect: it retains the device's primary context,
            # pushes it, and registers an atexit hook that pops it again. Pushing the
            # context by hand instead leaves the context stack non-empty at interpreter
            # shutdown, at which point PyCUDA aborts the process. Being a module, it
            # also stays a no-op when several processors are constructed.
            import pycuda.autoprimaryctx  # noqa: F401
            from pycuda.compiler import SourceModule

            # One translation unit per kernel, not one for all five. With a single unit
            # every test that touches the GPU pulls in every kernel, so a test-to-kernel
            # dependency map would be dense by construction and a kernel-aware selector
            # would have nothing to select on. Measured cost of the split (step 04a):
            # a flat ~0.5 s per nvcc invocation on a cold cache, ~2 s more than the single
            # unit, under 1 % of the suite; 1 ms in the hot regime a campaign runs in; and
            # cheaper than the single unit under mutation, since only the mutated kernel
            # loses its cache entry.
            self._module = {
                name: SourceModule(kernels.get_kernel_source(name)) for name in kernels.KERNEL_NAMES
            }
            self._fn = {name: module.get_function(name) for name, module in self._module.items()}
            self._backend = "pycuda"
        except Exception:
            self._module = {}
            self._fn = {}
            self._backend = "numpy"

    @property
    def backend(self) -> str:
        return self._backend

    def to_device(self, frame: np.ndarray) -> Any:
        """Move a host array onto the compute device (no-op for NumPy)."""
        if self._backend == "pycuda":  # pragma: no cover - needs GPU
            return DeviceArray(_gpuarray.to_gpu(np.ascontiguousarray(frame)))
        return self._xp.asarray(frame)

    def to_host(self, arr: Any) -> np.ndarray:
        """Copy a device array back to a NumPy host array."""
        if isinstance(arr, DeviceArray):  # pragma: no cover - needs GPU
            return arr.get()
        return np.asarray(arr)

    def convert_grayscale(self, frame: Any) -> Any:
        """Convert an RGB/BGR frame to grayscale using luminance weights."""
        if self._backend == "pycuda":  # pragma: no cover - needs GPU
            return self._grayscale_cuda(frame)
        xp = self._xp
        if frame.ndim == 2:
            return frame.astype(xp.float32)
        weights = xp.asarray([0.114, 0.587, 0.299], dtype=xp.float32)  # BGR order
        gray = frame.astype(xp.float32) @ weights
        return gray

    def gaussian_blur(self, frame: Any, kernel_size: int = 5) -> Any:
        """Separable Gaussian blur. ``kernel_size`` must be a positive odd int."""
        if kernel_size <= 1:
            return frame.astype(np.float32)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if self._backend == "pycuda":  # pragma: no cover - needs GPU
            return self._blur_cuda(frame, kernel_size)
        xp = self._xp
        arr = frame.astype(xp.float32)
        kernel = self._gaussian_kernel(kernel_size)
        # Convolve rows then columns (separable) for O(k) instead of O(k^2).
        blurred = self._convolve1d(arr, kernel, axis=1)
        blurred = self._convolve1d(blurred, kernel, axis=0)
        return blurred

    def absolute_diff(self, frame_a: Any, frame_b: Any) -> Any:
        if self._backend == "pycuda":  # pragma: no cover - needs GPU
            return self._absdiff_cuda(frame_a, frame_b)
        xp = self._xp
        return xp.abs(frame_a.astype(xp.float32) - frame_b.astype(xp.float32))

    def threshold(self, arr: Any, value: float) -> Any:
        """Binary mask (uint8 0/255) where ``arr > value``."""
        if self._backend == "pycuda":  # pragma: no cover - needs GPU
            return self._threshold_cuda(arr, value)
        xp = self._xp
        mask = (arr > value).astype(xp.uint8) * 255
        return mask

    # -- NumPy reference helpers -----------------------------------------
    def _gaussian_kernel(self, size: int) -> Any:
        xp = self._xp
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
        ax = xp.arange(size, dtype=xp.float32) - (size - 1) / 2.0
        kernel = xp.exp(-(ax**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()
        return kernel

    def _convolve1d(self, arr: Any, kernel: Any, axis: int) -> Any:
        xp = self._xp
        pad = kernel.shape[0] // 2
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (pad, pad)
        padded = xp.pad(arr, pad_width, mode="edge")
        out = xp.zeros_like(arr)
        for k in range(kernel.shape[0]):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(k, k + arr.shape[axis])
            out = out + kernel[k] * padded[tuple(sl)]
        return out

    # -- PyCUDA path -------------------------------------------------------
    def _as_gpu(self, arr: Any, dtype: Any = None) -> Any:  # pragma: no cover - needs GPU
        """Accept a DeviceArray or a host array and return a GPUArray of ``dtype``."""
        if isinstance(arr, DeviceArray):
            gpu = arr.raw
        else:
            gpu = _gpuarray.to_gpu(np.ascontiguousarray(arr))
        if dtype is not None and gpu.dtype != np.dtype(dtype):
            gpu = gpu.astype(np.dtype(dtype))
        return gpu

    @staticmethod
    def _grid_2d(width: int, height: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        block = (kernels.TILE_W, kernels.TILE_H, 1)
        grid = (
            (width + kernels.TILE_W - 1) // kernels.TILE_W,
            (height + kernels.TILE_H - 1) // kernels.TILE_H,
            1,
        )
        return block, grid

    def _grayscale_cuda(self, frame: Any) -> DeviceArray:  # pragma: no cover - needs GPU
        src = self._as_gpu(frame)
        if len(src.shape) == 2:
            return DeviceArray(src.astype(np.float32))
        if src.dtype != np.uint8:
            src = src.astype(np.uint8)
        height, width = src.shape[0], src.shape[1]
        dst = _gpuarray.empty((height, width), dtype=np.float32)
        block, grid = self._grid_2d(width, height)
        _trace.record("grayscale_bgr_u8")
        self._fn["grayscale_bgr_u8"](
            src, dst, np.int32(width), np.int32(height), block=block, grid=grid
        )
        return DeviceArray(dst)

    def _blur_cuda(self, frame: Any, kernel_size: int) -> DeviceArray:  # pragma: no cover
        if kernel_size > kernels.MAX_KERNEL_SIZE:
            raise ValueError(
                f"kernel_size {kernel_size} exceeds the shared-memory halo "
                f"(max {kernels.MAX_KERNEL_SIZE})"
            )
        src = self._as_gpu(frame, np.float32)
        height, width = src.shape[0], src.shape[1]
        taps = _gpuarray.to_gpu(np.ascontiguousarray(self._gaussian_kernel(kernel_size)))
        radius = np.int32(kernel_size // 2)
        block, grid = self._grid_2d(width, height)

        tmp = _gpuarray.empty((height, width), dtype=np.float32)
        dst = _gpuarray.empty((height, width), dtype=np.float32)
        args = (radius, np.int32(width), np.int32(height))
        _trace.record("conv1d_horizontal")
        self._fn["conv1d_horizontal"](src, tmp, taps, *args, block=block, grid=grid)
        _trace.record("conv1d_vertical")
        self._fn["conv1d_vertical"](tmp, dst, taps, *args, block=block, grid=grid)
        return DeviceArray(dst)

    def _absdiff_cuda(self, a: Any, b: Any) -> DeviceArray:  # pragma: no cover - needs GPU
        src_a = self._as_gpu(a, np.float32)
        src_b = self._as_gpu(b, np.float32)
        height, width = src_a.shape[0], src_a.shape[1]
        dst = _gpuarray.empty((height, width), dtype=np.float32)
        block, grid = self._grid_2d(width, height)
        _trace.record("absdiff_f32")
        self._fn["absdiff_f32"](
            src_a, src_b, dst, np.int32(width), np.int32(height), block=block, grid=grid
        )
        return DeviceArray(dst)

    def _threshold_cuda(self, arr: Any, value: float) -> DeviceArray:  # pragma: no cover
        src = self._as_gpu(arr, np.float32)
        height, width = src.shape[0], src.shape[1]
        dst = _gpuarray.empty((height, width), dtype=np.uint8)
        block, grid = self._grid_2d(width, height)
        _trace.record("threshold_u8")
        self._fn["threshold_u8"](
            src, dst, np.float32(value), np.int32(width), np.int32(height), block=block, grid=grid
        )
        return DeviceArray(dst)
