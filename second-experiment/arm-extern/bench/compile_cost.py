"""Measure what `nvcc` costs at process start, one translation unit against five.

`SourceModule` shells out to nvcc every time a process builds a module, so the cost is
fixed: a session pays it whether it runs 5 tests or 113. It never shrinks under test
selection and therefore caps the reduction RQ2 can report.

Splitting the five kernels into five translation units would let a test depend only on
the kernels it uses, but may multiply that fixed cost. This script produces the number;
it decides nothing and it is not a refactoring step. No production module is touched:
both configurations are assembled here from the public accessors of
`motion_pipeline.processing.kernels`, which is also what makes the script identical on
both arms of the experiment.

    Config A -- one `SourceModule` holding all five kernels (what the processor does).
    Config B -- five `SourceModule`s, one per kernel. Whatever each kernel needs to
                compile alone (the tile-geometry defines inline, `common.cuh` on the
                external arm) is included five times. That duplication is part of the
                cost being measured, not a distortion to remove.

One process measures one configuration once. Compilation is a start-up cost, so timing
both configurations in a single process would measure the second one against a warm
CUDA context and a warm compiler cache. `run_compile_cost.sh` re-launches this script
for every repetition.

Usage:
    python bench/compile_cost.py --config A --cache cold --rep 1 --cache-dir /path
    python bench/compile_cost.py --probe        # environment only, compiles nothing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from motion_pipeline.processing import kernels


def _nvcc_version() -> str:
    """Report the compiler actually on PATH, since the cost being measured is its own."""
    try:
        out = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    return out.stdout.strip().splitlines()[-1] if out.stdout else f"rc={out.returncode}"


def _default_cache_dir() -> str:
    """Where PyCUDA would cache without `cache_dir=`.

    Reported rather than used: the measurement passes an explicit directory so that
    "cold" is a directory this script controls, not a path that moved between PyCUDA
    releases (tempdir in older versions, platformdirs in current ones).
    """
    try:
        import platformdirs

        return str(Path(platformdirs.user_cache_dir("pycuda", "pycuda")) / "compiler-cache-v1")
    except Exception:  # noqa: BLE001 - probing only; any failure just means "unknown"
        import tempfile

        return str(Path(tempfile.gettempdir()) / "pycuda-compiler-cache-v1-<uid>")


def _cache_entries(cache_dir: Path) -> int:
    """Count cached compilation units, so a claimed cold run can be checked afterwards."""
    if not cache_dir.is_dir():
        return 0
    return sum(1 for _ in cache_dir.rglob("*") if _.is_file())


def _probe(cache_dir: Path) -> dict[str, Any]:
    env_keys = ("PYCUDA_CACHE_DIR", "PYCUDA_DISABLE_CACHE", "CUDA_ROOT", "CUDA_CACHE_DISABLE")
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "nvcc_path": shutil.which("nvcc") or "MISSING",
        "nvcc_version": _nvcc_version(),
        "cache_dir_used": str(cache_dir),
        "cache_dir_pycuda_default": _default_cache_dir(),
        "cache_entries_before": _cache_entries(cache_dir),
        "env": {k: os.environ.get(k, "") for k in env_keys},
        "kernels": list(kernels.KERNEL_NAMES),
        "module_source_bytes": len(kernels.get_module_source()),
        "kernel_source_bytes": {
            name: len(kernels.get_kernel_source(name)) for name in kernels.KERNEL_NAMES
        },
    }
    try:
        import pycuda.driver  # noqa: PLC0415 - probe-time import, keeps --probe cheap

        info["pycuda_version"] = getattr(pycuda, "VERSION_TEXT", "unknown")
    except Exception as exc:  # noqa: BLE001 - reported, not raised: --probe must not fail
        info["pycuda_version"] = f"unavailable: {exc}"
    return info


def _init_context() -> float:
    """Create the CUDA context and return its cost, kept out of the compile figures.

    `autoprimaryctx` rather than a hand-pushed context: a non-empty context stack at
    interpreter shutdown makes PyCUDA abort the process after the work has succeeded.
    """
    start = time.perf_counter()
    import pycuda.autoprimaryctx  # noqa: F401, PLC0415 - imported for its side effect

    return time.perf_counter() - start


def _measure_single(cache_dir: Path) -> dict[str, Any]:
    """Config A: one translation unit, five `get_function` lookups."""
    from pycuda.compiler import SourceModule  # noqa: PLC0415 - after the context exists

    source = kernels.get_module_source()

    build_start = time.perf_counter()
    module = SourceModule(source, cache_dir=str(cache_dir))
    build_s = time.perf_counter() - build_start

    fn_start = time.perf_counter()
    for name in kernels.KERNEL_NAMES:
        module.get_function(name)
    get_function_s = time.perf_counter() - fn_start

    return {
        "units": 1,
        "build_s": build_s,
        "get_function_s": get_function_s,
        "per_kernel": {},
    }


def _measure_split(cache_dir: Path) -> dict[str, Any]:
    """Config B: one translation unit per kernel, timed individually.

    The per-kernel breakdown is what makes the total interpretable. If the five costs
    are each close to the single-unit cost, the price is a fixed per-nvcc-invocation
    overhead and five units cost roughly five times one. If they fall well below it,
    the price tracks the volume of code compiled and the split is nearly free.
    """
    from pycuda.compiler import SourceModule  # noqa: PLC0415 - after the context exists

    per_kernel: dict[str, dict[str, float]] = {}
    build_s = 0.0
    get_function_s = 0.0

    for name in kernels.KERNEL_NAMES:
        source = kernels.get_kernel_source(name)

        build_start = time.perf_counter()
        module = SourceModule(source, cache_dir=str(cache_dir))
        one_build = time.perf_counter() - build_start

        fn_start = time.perf_counter()
        module.get_function(name)
        one_fn = time.perf_counter() - fn_start

        per_kernel[name] = {"build_s": one_build, "get_function_s": one_fn}
        build_s += one_build
        get_function_s += one_fn

    return {
        "units": len(kernels.KERNEL_NAMES),
        "build_s": build_s,
        "get_function_s": get_function_s,
        "per_kernel": per_kernel,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("A", "B"), help="A = one unit, B = five units")
    parser.add_argument(
        "--cache",
        choices=("cold", "hot"),
        default="cold",
        help="label only; clearing the cache is the runner's job",
    )
    parser.add_argument("--rep", type=int, default=0, help="repetition index, recorded as-is")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="mark this run as a discarded cache-populating run",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("BENCH_CACHE_DIR", ""),
        help="explicit PyCUDA cache directory; required so that 'cold' is well defined",
    )
    parser.add_argument(
        "--probe", action="store_true", help="report the environment, compile nothing"
    )
    parser.add_argument("--out", default="", help="append the JSON record here instead of stdout")
    args = parser.parse_args(argv)

    if not args.cache_dir:
        parser.error("--cache-dir is required (or set BENCH_CACHE_DIR)")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = _probe(cache_dir)

    if args.probe:
        record["kind"] = "probe"
        # Report first, then refuse: a probe that prints the environment and *then* fails
        # tells the runner why the measurement was skipped instead of leaving it to
        # produce twenty empty repetitions.
        usable = record["nvcc_path"] != "MISSING" and not str(record["pycuda_version"]).startswith(
            "unavailable"
        )
        record["usable"] = usable
    else:
        if not args.config:
            parser.error("--config is required unless --probe is given")
        record.update(
            {
                "kind": "measurement",
                "config": args.config,
                "cache": args.cache,
                "rep": args.rep,
                "warmup": args.warmup,
            }
        )
        total_start = time.perf_counter()
        record["context_init_s"] = _init_context()
        measured = _measure_single(cache_dir) if args.config == "A" else _measure_split(cache_dir)
        record.update(measured)
        record["total_s"] = time.perf_counter() - total_start
        # build + get_function, excluding context creation: the quantity the split
        # actually changes. context_init_s is reported separately and is paid either way.
        record["compile_s"] = record["build_s"] + record["get_function_s"]
        record["cache_entries_after"] = _cache_entries(cache_dir)

    line = json.dumps(record)
    if args.out:
        with Path(args.out).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print(line)
    return 0 if record.get("usable", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
