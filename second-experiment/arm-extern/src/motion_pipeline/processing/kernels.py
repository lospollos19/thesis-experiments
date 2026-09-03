"""Loader for the CUDA-C sources stored as external ``.cu`` files.

This module is the *external storage* variant of the kernel sources. The CUDA-C
itself lives in ``processing/cuda/``; nothing here contains device code. The public
surface is unchanged from the inline variant, so :mod:`gpu_processor` is untouched:
everything still goes through :func:`get_kernel_source`.

Layout conventions shared by all kernels:

* images are row-major and contiguous, indexed as ``y * width + x``;
* colour input is 8-bit BGR interleaved (3 bytes per pixel), matching OpenCV;
* intermediate results are ``float32``, binary masks are ``uint8`` (0 or 255);
* out-of-image reads are clamped to the border (``edge`` padding), which is what the
  NumPy reference path does via ``np.pad(..., mode="edge")``.

Consequence worth stating explicitly, since it is the object of study: a ``.cu`` file
is not a Python module. Nothing imports it, it never appears in ``sys.modules``, and
an import-graph-based test selector has no edge to it. Editing device code here
changes program behaviour while leaving the Python dependency graph identical.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Directory holding the CUDA-C sources. One ``.cu`` per kernel, plus ``common.cuh``.
KERNEL_DIR = Path(__file__).parent / "cuda"

#: Shared tile-geometry defines, prepended to the kernels that need them.
COMMON_HEADER = "common.cuh"

KERNEL_NAMES: tuple[str, ...] = (
    "grayscale_bgr_u8",
    "conv1d_horizontal",
    "conv1d_vertical",
    "absdiff_f32",
    "threshold_u8",
)

#: Kernels that reference TILE_W / TILE_H / MAX_RADIUS and therefore need the header.
_NEEDS_COMMON = frozenset({"conv1d_horizontal", "conv1d_vertical"})


def _read(filename: str) -> str:
    path = KERNEL_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"CUDA source {filename!r} is missing from {KERNEL_DIR}. "
            "The .cu files must ship with the package: poetry-core picks up every "
            "file under the package directory, so this means the distribution was "
            "built from an incomplete tree."
        ) from None


def _parse_define(source: str, name: str) -> int:
    """Extract an integer ``#define`` from the shared header.

    The header is the single source of truth for the tile geometry: the launch
    configuration in :mod:`gpu_processor` must agree with what the kernels were
    compiled against, and duplicating the numbers in Python would let the two drift
    apart silently.
    """
    match = re.search(rf"^\s*#define\s+{name}\s+(\d+)\s*$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"{COMMON_HEADER} does not define {name}")
    return int(match.group(1))


_COMMON_SOURCE = _read(COMMON_HEADER)

TILE_W = _parse_define(_COMMON_SOURCE, "TILE_W")
TILE_H = _parse_define(_COMMON_SOURCE, "TILE_H")
MAX_RADIUS = _parse_define(_COMMON_SOURCE, "MAX_RADIUS")
MAX_KERNEL_SIZE = 2 * MAX_RADIUS + 1


def get_kernel_source(name: str) -> str:
    """Return the CUDA-C source of kernel ``name``.

    Single point of access to the kernel sources: callers never read the storage
    directly, so the storage mode can be changed here alone.
    """
    if name not in KERNEL_NAMES:
        known = ", ".join(KERNEL_NAMES)
        raise KeyError(f"unknown kernel {name!r}; known kernels: {known}")
    source = _read(f"{name}.cu")
    if name in _NEEDS_COMMON:
        return _COMMON_SOURCE + source
    return source


def get_module_source() -> str:
    """Concatenate every kernel into a single translation unit.

    PyCUDA compiles one ``SourceModule`` per string, so the processor builds all
    kernels at once rather than paying nvcc's start-up cost five times. The shared
    header is emitted once, ahead of the kernels that reference it.
    """
    bodies = (_read(f"{name}.cu") for name in KERNEL_NAMES)
    return _COMMON_SOURCE + "\n".join(bodies)
