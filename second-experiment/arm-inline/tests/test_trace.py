"""The kernel tracer itself, on the CPU: it is instrumentation the results depend on.

Two properties are load-bearing and neither is visible in a GPU run:

* it stays inert unless `MP_TRACE_KERNELS=1`, because the same suite is used for the
  occupancy timings and instrumentation left running would sit inside the measurement;
* a launch outside a test is dropped rather than attributed to a neighbouring test,
  because a wrong edge in the map is worse than a missing one is obvious.
"""

from __future__ import annotations

import pytest
from motion_pipeline.processing import _trace


@pytest.fixture(autouse=True)
def _clean_tracer():
    _trace.reset()
    yield
    _trace.reset()


def test_records_nothing_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_trace, "ENABLED", False)
    _trace.set_active_test("tests/test_x.py::test_y")
    _trace.record("absdiff_f32")
    assert _trace.dump() == {}


def test_records_per_test_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_trace, "ENABLED", True)
    _trace.set_active_test("tests/test_x.py::test_a")
    _trace.record("absdiff_f32")
    _trace.record("threshold_u8")
    _trace.record("absdiff_f32")  # repeated launches collapse to one edge
    _trace.set_active_test("tests/test_x.py::test_b")
    _trace.record("grayscale_bgr_u8")

    assert _trace.dump() == {
        "tests/test_x.py::test_a": ["absdiff_f32", "threshold_u8"],
        "tests/test_x.py::test_b": ["grayscale_bgr_u8"],
    }


def test_drops_launches_outside_a_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_trace, "ENABLED", True)
    _trace.set_active_test(None)
    _trace.record("conv1d_horizontal")
    assert _trace.dump() == {}
