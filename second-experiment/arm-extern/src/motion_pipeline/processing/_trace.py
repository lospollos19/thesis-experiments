"""Dynamic kernel-usage tracer. Inert unless ``MP_TRACE_KERNELS=1``.

Records which CUDA kernels each test actually launches, which is the input a
kernel-aware selector needs and which an import-graph selector cannot produce.

**Instrumented at the launch sites, deliberately, not at the source accessors.**
``get_kernel_source`` is read once per process: PyCUDA compiles a module and keeps
it, so tracing the source read would attribute every kernel to whichever test ran
first and leave every later test looking dependency-free. A selector built on that
map would skip tests that do use the mutated kernel and report green -- the exact
failure this study is meant to detect. A launch happens on every call and cannot be
cached away.

The environment guard is not a micro-optimisation: the same suite is used for the
occupancy timings, and instrumentation left active would sit inside the measured
quantity.
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("MP_TRACE_KERNELS") == "1"

_active_test: str | None = None
_usage: dict[str, set[str]] = {}


def set_active_test(nodeid: str | None) -> None:
    """Bind subsequent launches to a test, or to nothing when passed ``None``.

    Launches outside a test (import-time work, fixtures at session scope) are
    dropped rather than attributed to whichever test happens to follow.
    """
    global _active_test
    _active_test = nodeid


def record(kernel_name: str) -> None:
    """Called at every kernel launch. Must not sit behind a cache."""
    if not ENABLED or _active_test is None:
        return
    _usage.setdefault(_active_test, set()).add(kernel_name)


def dump() -> dict[str, list[str]]:
    """The map collected so far, sorted so runs can be diffed."""
    return {test: sorted(names) for test, names in sorted(_usage.items())}


def reset() -> None:
    """Drop everything recorded. For tests of the tracer itself."""
    global _active_test
    _active_test = None
    _usage.clear()
