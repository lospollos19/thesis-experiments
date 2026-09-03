"""The selector decides what is *not* run, so its rules are tested one by one.

Every case here is a case where getting it wrong produces a green report on a broken
build — the outcome RQ2 exists to measure. A selector that over-selects wastes device
time and is visible in the numbers; one that under-selects is invisible until something
ships. So the tests are written from the under-selection side.

No artefacts and no device: a manifest is built from the real tree, then individual
hashes are overwritten to stand for "this kernel changed". That keeps the sources
untouched while exercising the real comparison code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import kernel_select  # noqa: E402

MAP = {
    "tests/test_a.py::test_grayscale_only": ["grayscale_bgr_u8"],
    "tests/test_a.py::test_blur_only": ["conv1d_horizontal", "conv1d_vertical"],
    "tests/test_b.py::test_whole_chain": [
        "absdiff_f32",
        "conv1d_horizontal",
        "conv1d_vertical",
        "grayscale_bgr_u8",
        "threshold_u8",
    ],
}
KERNEL_FREE = {"tests/test_c.py::test_backend_only"}
GPU_COLLECTED = [*MAP, *KERNEL_FREE]
#: A CPU test: outside the map's universe, but it asserts on MAX_KERNEL_SIZE.
CPU_ONLY = "tests/test_kernels.py::test_max_blur_contract"
COLLECTED = [*GPU_COLLECTED, CPU_ONLY]


@pytest.fixture
def manifest() -> dict:
    return kernel_select.build_manifest("baseline")


def _with_changed(manifest: dict, *names: str) -> dict:
    changed = {**manifest, "kernels": dict(manifest["kernels"])}
    for name in names:
        changed["kernels"][name] = "0" * 64
    return changed


def test_nothing_changed_selects_nothing(manifest):
    """The control. A selector that cannot say 'nothing' cannot save anything."""
    selected, _ = kernel_select.select(manifest, MAP, COLLECTED, GPU_COLLECTED, KERNEL_FREE, set())
    assert selected == []


def test_a_changed_kernel_selects_its_tests(manifest):
    selected, _ = kernel_select.select(
        _with_changed(manifest, "grayscale_bgr_u8"),
        MAP,
        COLLECTED,
        GPU_COLLECTED,
        KERNEL_FREE,
        set(),
    )
    assert "tests/test_a.py::test_grayscale_only" in selected
    assert "tests/test_b.py::test_whole_chain" in selected
    assert "tests/test_a.py::test_blur_only" not in selected


def test_any_kernel_change_selects_the_kernel_free_tests(manifest):
    """R3: they launch nothing, but they depend on every kernel compiling."""
    selected, _ = kernel_select.select(
        _with_changed(manifest, "absdiff_f32"), MAP, COLLECTED, GPU_COLLECTED, KERNEL_FREE, set()
    )
    assert "tests/test_c.py::test_backend_only" in selected


def test_an_unknown_nodeid_is_selected(manifest):
    """R2, and not hypothetical.

    Parametrised ids embed the value, so halving ``MAX_RADIUS`` turns
    ``...matches_numpy[33]`` into ``[17]`` and the id in the map ceases to exist. The
    natural implementation deselects it exactly when it matters.
    """
    new = "tests/test_a.py::test_new[17]"
    selected, _ = kernel_select.select(
        manifest, MAP, [*COLLECTED, new], [*GPU_COLLECTED, new], KERNEL_FREE, set()
    )
    assert selected == [new]


def test_a_cpu_test_is_out_of_the_map_universe_but_not_out_of_the_geometry_rule(manifest):
    """The two universes, and the violation that showed why they differ.

    ``CPU_ONLY`` asserts on ``MAX_KERNEL_SIZE``, a module attribute reachable from Python
    without any launch. It is absent from the map because it launches nothing, so a
    kernel change must not drag it in — otherwise every kernel edit selects the whole CPU
    suite and the saving disappears. But halving ``MAX_RADIUS`` does change its outcome,
    and on ``variant/extern`` no Python file changes, so testmon cannot cover it: R4 is
    the only rule that can, and it has to range over the full collection to do so.

    Measured, not hypothetical: with the gpu-marked list used as the whole universe, the
    first run of the ground truth had exactly this test failing under
    ``geometry-halve-max-radius`` and unselected.
    """
    kernel_changed, _ = kernel_select.select(
        _with_changed(manifest, "absdiff_f32"), MAP, COLLECTED, GPU_COLLECTED, KERNEL_FREE, set()
    )
    assert CPU_ONLY not in kernel_changed

    geometry = {**manifest, kernel_select.GEOMETRY: "0" * 64}
    geometry_changed, _ = kernel_select.select(
        geometry, MAP, COLLECTED, GPU_COLLECTED, KERNEL_FREE, set()
    )
    assert CPU_ONLY in geometry_changed


def test_changed_geometry_selects_everything(manifest):
    """R4: the shared header reaches the launch config, the guard and the contract."""
    changed = {**manifest, kernel_select.GEOMETRY: "0" * 64}
    selected, reasons = kernel_select.select(
        changed, MAP, COLLECTED, GPU_COLLECTED, KERNEL_FREE, set()
    )
    assert set(selected) == set(COLLECTED)
    assert reasons["geometry_changed"] is True


def test_the_selection_is_a_union_with_testmon(manifest):
    """testmon's edges are real edges over Python code; this tool knows nothing of them."""
    testmon = {"tests/test_a.py::test_blur_only"}
    selected, _ = kernel_select.select(
        manifest, MAP, COLLECTED, GPU_COLLECTED, KERNEL_FREE, testmon
    )
    assert selected == ["tests/test_a.py::test_blur_only"]


def test_testmon_ids_outside_the_collection_are_ignored(manifest):
    """A stale id from another content state must not be reported as selected."""
    selected, _ = kernel_select.select(
        manifest, MAP, COLLECTED, GPU_COLLECTED, KERNEL_FREE, {"tests/gone.py::test_removed"}
    )
    assert selected == []


def test_build_manifest_reloads_the_kernels_module(monkeypatch):
    """The staleness that broke R4 in run 31955528366, locked down.

    ``variant/extern``'s loader binds the shared header and the tile constants at import
    time. A long-lived process — the campaign imports once and then mutates files — hashes
    a header it read minutes ago, so a geometry mutation reports ``geometry_changed:
    false`` and R4 never fires. The same defect as testmon's, in the tool written to
    cover it.

    This asserts the mechanism rather than the effect: demonstrating the effect means
    editing the real sources, and a test that leaves the tree dirty on failure is worse
    than one that is coupled to the fix. The effect itself is checked by the campaign,
    which records ``geometry_changed`` per mutation.
    """
    called = []
    real_reload = kernel_select.importlib.reload
    monkeypatch.setattr(
        kernel_select.importlib, "reload", lambda m: called.append(m) or real_reload(m)
    )
    kernel_select.build_manifest("x")
    assert called, "build_manifest must re-import the kernels module before hashing"
    assert called[0] is kernel_select.kernels
