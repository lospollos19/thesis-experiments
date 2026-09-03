"""Turn the JSONL emitted by `compile_cost.py` into the reporting table.

Median, min and max -- not the mean. Thermal throttling and contention on a shared
device skew the distribution to the right, and a mean absorbs the skew into a number
that describes neither the typical run nor the worst one.

Usage:
    python bench/aggregate_compile_cost.py results.jsonl > summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

CONFIG_LABEL = {"A": "A — 1 unit", "B": "B — 5 units"}


def _load(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the log into measurements and everything else (probes, warmups)."""
    measurements: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("kind") != "measurement":
            continue
        if record.get("warmup"):
            record["discard_reason"] = "warmup run, populates the cache for the hot block"
            discarded.append(record)
        else:
            measurements.append(record)
    return measurements, discarded


def _stats(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), min(values), max(values)


def _table(measurements: list[dict[str, Any]], field: str) -> list[str]:
    rows = ["| Config | Cache | n | Median (s) | Min | Max |", "|---|---|---|---|---|---|"]
    for config in ("A", "B"):
        for cache in ("cold", "hot"):
            values = [
                m[field] for m in measurements if m["config"] == config and m["cache"] == cache
            ]
            if not values:
                rows.append(f"| {CONFIG_LABEL[config]} | {cache} | 0 | — | — | — |")
                continue
            median, low, high = _stats(values)
            rows.append(
                f"| {CONFIG_LABEL[config]} | {cache} | {len(values)} "
                f"| {median:.3f} | {low:.3f} | {high:.3f} |"
            )
    return rows


def _per_kernel_table(measurements: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| Kernel | Cache | n | Median build (s) | Min | Max | Source bytes |",
        "|---|---|---|---|---|---|---|",
    ]
    split = [m for m in measurements if m["config"] == "B"]
    names: list[str] = []
    for m in split:
        for name in m.get("per_kernel", {}):
            if name not in names:
                names.append(name)
    for name in names:
        for cache in ("cold", "hot"):
            values = [
                m["per_kernel"][name]["build_s"]
                for m in split
                if m["cache"] == cache and name in m.get("per_kernel", {})
            ]
            if not values:
                continue
            median, low, high = _stats(values)
            size = split[0].get("kernel_source_bytes", {}).get(name, "?")
            rows.append(
                f"| `{name}` | {cache} | {len(values)} "
                f"| {median:.3f} | {low:.3f} | {high:.3f} | {size} |"
            )
    return rows


def _ratio_line(measurements: list[dict[str, Any]], cache: str) -> str:
    def median_for(config: str) -> float | None:
        values = [
            m["compile_s"] for m in measurements if m["config"] == config and m["cache"] == cache
        ]
        return statistics.median(values) if values else None

    a, b = median_for("A"), median_for("B")
    if a is None or b is None or a == 0:
        return f"- **{cache}**: not enough data for a ratio."
    return f"- **{cache}**: B / A = **{b / a:.2f}×** ({b:.3f} s against {a:.3f} s, medians)."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSONL written by compile_cost.py")
    args = parser.parse_args(argv)

    measurements, discarded = _load(args.results)
    if not measurements:
        print("No measurements in the log — every run failed or was discarded.")
        return 1

    first = measurements[0]
    out: list[str] = [
        "# Compilation cost — one translation unit against five",
        "",
        "`compile_s` = `SourceModule` construction + `get_function` for all five kernels.",
        "CUDA context creation is excluded (reported separately below): it is paid either way",
        "and is not what the split changes.",
        "",
        "## compile_s",
        "",
        *_table(measurements, "compile_s"),
        "",
        "## Ratio, the number the decision turns on",
        "",
        _ratio_line(measurements, "cold"),
        _ratio_line(measurements, "hot"),
        "",
        "## Per-kernel breakdown, config B",
        "",
        "Five costs each close to the single-unit cost means a fixed per-`nvcc` overhead and a",
        "split that costs about 5×. Costs well below it mean the price tracks compiled volume.",
        "",
        *_per_kernel_table(measurements),
        "",
        "## CUDA context creation (excluded from the figures above)",
        "",
        *_table(measurements, "context_init_s"),
        "",
        "## Environment",
        "",
        f"- nvcc: `{first.get('nvcc_version', '?')}` at `{first.get('nvcc_path', '?')}`",
        f"- PyCUDA: `{first.get('pycuda_version', '?')}`, Python `{first.get('python', '?')}`",
        f"- cache directory used: `{first.get('cache_dir_used', '?')}`",
        f"- PyCUDA default cache directory (not used, reported for context): "
        f"`{first.get('cache_dir_pycuda_default', '?')}`",
        f"- single-unit source: {first.get('module_source_bytes', '?')} bytes",
        "",
        "## Runs discarded",
        "",
    ]
    if discarded:
        out += [
            f"- config {d['config']}, {d['cache']}, rep {d['rep']}: {d['discard_reason']}"
            for d in discarded
        ]
    else:
        out.append("- none")
    out.append("")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
