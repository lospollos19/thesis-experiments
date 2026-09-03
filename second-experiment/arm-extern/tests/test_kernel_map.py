"""Completeness of the traced test -> kernel map (R1, question 3).

An entry missing from the map does not mean "this test depends on no kernel". It
means the tracer missed it. A kernel-aware selector reading such a map would skip a
test that does exercise the mutated kernel and report green: that is precisely the
safety violation RQ2 measures, manufactured by the instrument instead of observed.

So the map is checked against the list of `gpu`-marked tests pytest actually
collects, and both come from the environment rather than from this file:

    MP_KERNEL_MAP          merged map (see tools/merge_kernel_deps.py)
    MP_GPU_NODEIDS         one collected gpu nodeid per line
    MP_KERNEL_FREE_NODEIDS nodeids marked `no_kernel_launch` (optional)

The first two unset -- the normal case on a machine without a device -- and the check
skips. It is a check on a completed tracing campaign, not something a local run can
answer.

A test can legitimately launch nothing: it asserts on `backend`, it exits before the
launch, or it raises on a guard. R1's first run reported exactly three such tests as
INCOMPLETE, which made the criterion wrong rather than the tracer. They are declared
with `@pytest.mark.no_kernel_launch` at the test, where the reason is visible, and
this check exempts only what is declared.

The exemption is narrow on purpose, since it weakens a safety check:

- the denominator stays the full collected set, so a marker cannot shrink the
  property quietly;
- a declared test that *is* in the map with kernels fails the check, because the
  declaration is then false;
- and being absent here does not mean dependency-free. Those tests carry compilation
  and constant dependencies a launch-site tracer cannot see, so a selector must not
  read their absence as a licence to deselect them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} unset: no tracing campaign to check")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"{name} points at {path}, which does not exist")
    return path


def test_every_gpu_test_appears_in_the_kernel_map() -> None:
    kernel_map = json.loads(_env_path("MP_KERNEL_MAP").read_text(encoding="utf-8"))
    nodeids = [
        line.strip()
        for line in _env_path("MP_GPU_NODEIDS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert nodeids, "no gpu nodeids were collected — the campaign traced nothing"

    declared: set[str] = set()
    declared_path = os.environ.get("MP_KERNEL_FREE_NODEIDS")
    if declared_path and Path(declared_path).is_file():
        declared = {
            line.strip()
            for line in Path(declared_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    missing = [n for n in nodeids if n not in kernel_map and n not in declared]
    empty = [n for n in nodeids if n in kernel_map and not kernel_map[n]]
    overclaimed = sorted(n for n in declared if kernel_map.get(n))

    # Named, not counted: which test was missed is what tells you why it was missed.
    report = []
    if missing:
        report.append(f"{len(missing)} gpu test(s) absent from the map and not declared:")
        report += [f"  absent: {n}" for n in missing]
    if empty:
        report.append(f"{len(empty)} gpu test(s) present with no kernel:")
        report += [f"  empty : {n}" for n in empty]
    if overclaimed:
        report.append(f"{len(overclaimed)} test(s) marked no_kernel_launch that launch one:")
        report += [f"  wrong : {n} -> {', '.join(kernel_map[n])}" for n in overclaimed]
    assert not report, "\n".join(report)
