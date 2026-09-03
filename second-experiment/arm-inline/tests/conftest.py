"""Shared pytest fixtures. All hardware-free: simulated camera + NumPy backend."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from motion_pipeline.processing import _trace
from motion_pipeline.processing.gpu_processor import GPUProcessor
from motion_pipeline.simulation.data_stream import SimulatedCamera

#: (label, observed deviation, tolerance) collected by the equivalence tests.
_DEVIATIONS: list[tuple[str, float, float]] = []


def record_deviation(label: str, deviation: float, atol: float) -> None:
    """Record a kernel-vs-NumPy deviation for the end-of-run summary."""
    _DEVIATIONS.append((label, deviation, atol))


def assert_within(gpu: np.ndarray, ref: np.ndarray, atol: float, what: str) -> None:
    """Assert a purely absolute deviation bound, and record what was observed.

    Deliberately not ``np.allclose``: its criterion is ``atol + rtol * |ref|`` with
    ``rtol=1e-5`` by default, so on 0-255 intensities it silently adds 2.55e-3 to
    whatever atol is passed. That swamped every tolerance in this suite — absdiff's
    1e-5 became 2.56e-3, 256 times looser than documented — and the tolerances are
    what decide whether a kernel regression is detectable at all.

    Recording rather than printing: pytest captures a passing test's stdout.
    """
    deviation = float(np.max(np.abs(gpu - ref)))
    record_deviation(what, deviation, atol)
    assert deviation <= atol, f"{what}: max deviation {deviation:.3e} exceeds atol {atol:.0e}"


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Print the observed deviations as a table.

    A passing test's stdout is captured, so printing from inside the assertion
    helper produced nothing in CI. The margin between the analytically derived
    tolerances and what the hardware actually does is a result in its own right,
    so it has to survive to the log.
    """
    if not _DEVIATIONS:
        return
    terminalreporter.write_sep("=", "kernel vs NumPy deviation")
    width = max(len(label) for label, _, _ in _DEVIATIONS)
    for label, deviation, atol in _DEVIATIONS:
        margin = deviation / atol if atol else float("inf")
        terminalreporter.write_line(
            f"{label:<{width}}  observed {deviation:.3e}  atol {atol:.0e}  "
            f"({margin:6.1%} of budget)"
        )


# -- kernel-usage tracing (inert unless MP_TRACE_KERNELS=1) ---------------


@pytest.fixture(autouse=True)
def _trace_active_test(request: pytest.FixtureRequest):
    """Attribute kernel launches to the test that caused them.

    Autouse so no test can be traced anonymously: a launch recorded outside a test
    would be dropped, and a test missing from the map is indistinguishable from a
    test with no kernel dependency, which is the failure mode this whole map exists
    to rule out.
    """
    if not _trace.ENABLED:
        yield
        return
    _trace.set_active_test(request.node.nodeid)
    try:
        yield
    finally:
        _trace.set_active_test(None)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Write the traced map, one file per session.

    One session is not enough: a kernel launched only on a rare branch would be
    missing. The campaign runs several sessions and takes the union of the files
    (`tools/merge_kernel_deps.py`).
    """
    if not _trace.ENABLED:
        return
    destination = os.environ.get("MP_TRACE_OUT")
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_trace.dump(), indent=2, sort_keys=True), encoding="utf-8")


@pytest.fixture
def simulated_camera() -> SimulatedCamera:
    """30-frame stream with a motion event on frames 10-15."""
    return SimulatedCamera(
        resolution=(120, 160),
        num_frames=30,
        motion_frames=list(range(10, 16)),
        seed=7,
    )


@pytest.fixture
def gpu_processor() -> GPUProcessor:
    """Processor forced onto the NumPy backend (CI has no CUDA)."""
    return GPUProcessor(force_cpu=True)


@pytest.fixture
def cuda_processor() -> GPUProcessor:
    """Processor on the real CUDA-C kernel path, or skip when no device is available."""
    processor = GPUProcessor()
    if processor.backend != "pycuda":
        pytest.skip("no CUDA device / PyCUDA available")
    return processor


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Single 480x640x3 uint8 frame."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
