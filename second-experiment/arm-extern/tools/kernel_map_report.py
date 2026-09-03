"""Report on the traced map and rule on R1 question 2: is it exploitable?

The verdict is not "the map exists". A map where every test depends on all five
kernels is a valid map and a dead end: the fine granularity a kernel-aware selector
needs would not exist, condition C would have no object, and that has to be known
before anything is built on top of it. So the criterion is **non-triviality**: at
least one test must depend on a strict subset of the kernels.

A trivial map is a result to report, not a crash. This script says which it is and
exits non-zero so the job can carry the verdict.

    python tools/kernel_map_report.py kernel_deps.json --gpu-nodeids ids.txt --out verdict.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Kept here rather than imported from the package: this script also runs where the
# CUDA-C sources are not the ones being traced, and a mismatch should be visible.
ALL_KERNELS = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_map", type=Path)
    parser.add_argument("--gpu-nodeids", type=Path, help="one collected gpu nodeid per line")
    parser.add_argument(
        "--kernel-free",
        type=Path,
        help="nodeids marked no_kernel_launch: absent from the map by design, not missed",
    )
    parser.add_argument("--out", type=Path, help="write the markdown report here as well")
    args = parser.parse_args(argv)

    kernel_map: dict[str, list[str]] = json.loads(args.kernel_map.read_text(encoding="utf-8"))
    counts = Counter(len(v) for v in kernel_map.values())
    kernel_use = Counter(k for v in kernel_map.values() for k in v)

    strict_subset = [n for n, v in kernel_map.items() if 0 < len(v) < ALL_KERNELS]
    non_trivial = bool(strict_subset)

    lines = [
        "## Q2 — test to kernel map",
        "",
        f"- tests in the map: **{len(kernel_map)}**",
        f"- edges: **{sum(len(v) for v in kernel_map.values())}**",
        f"- tests depending on a strict subset: **{len(strict_subset)}**",
        f"- verdict: **{'NON-TRIVIAL' if non_trivial else 'TRIVIAL'}**",
        "",
        "| Kernels per test | Tests |",
        "|---|---|",
    ]
    lines += [f"| {n} | {counts.get(n, 0)} |" for n in range(0, ALL_KERNELS + 1)]
    lines += [
        "",
        "| Kernel | Tests depending on it |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {c} |" for k, c in sorted(kernel_use.items())]

    if not non_trivial:
        lines += [
            "",
            "Every traced test depends on the full kernel set. The fine granularity a",
            "kernel-aware selector would exploit does not exist in this suite: condition C",
            "has no object as it stands. This is a finding, not a tooling failure.",
        ]

    complete = True
    overclaimed: list[str] = []
    if args.gpu_nodeids and args.gpu_nodeids.is_file():
        nodeids = [
            line.strip()
            for line in args.gpu_nodeids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        declared = set()
        if args.kernel_free and args.kernel_free.is_file():
            declared = {
                line.strip()
                for line in args.kernel_free.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        # The denominator stays the full collected set. Subtracting the declarations from
        # it would let the safety property be weakened by adding a marker, and the shrink
        # would not appear in the report.
        missing = [n for n in nodeids if n not in kernel_map and n not in declared]
        empty = [n for n in nodeids if n in kernel_map and not kernel_map[n]]
        # A declaration that is wrong in the other direction: it said "no launch" and the
        # tracer saw one. Harmless for safety, but the declaration is false and says so.
        overclaimed = sorted(n for n in declared if kernel_map.get(n))
        complete = not (missing or empty)
        lines += [
            "",
            "## Q3 — completeness",
            "",
            f"- gpu tests collected: **{len(nodeids)}**",
            f"- declared `no_kernel_launch`: **{len(declared)}**",
            f"- absent and not declared: **{len(missing)}**",
            f"- present but with no kernel: **{len(empty)}**",
            f"- verdict: **{'COMPLETE' if complete else 'INCOMPLETE'}**",
        ]
        # Named individually: a count says the tracer failed, a name says why.
        lines += [f"  - absent: `{n}`" for n in missing]
        lines += [f"  - empty: `{n}`" for n in empty]
        if overclaimed:
            lines += [
                "",
                "Declared `no_kernel_launch` but traced launching a kernel — the marker is",
                "wrong and must be removed:",
            ]
            lines += [f"  - `{n}`: {', '.join(kernel_map[n])}" for n in overclaimed]
        if declared and not missing and not empty:
            lines += [
                "",
                "The declared tests are absent by design: they assert on the backend, exit",
                "before a launch, or raise on a guard. They are not dependency-free — they",
                "carry compilation and constant dependencies that a launch-site tracer",
                "cannot observe, and a kernel-aware selector needs a second edge type for",
                "them. Absence from this map is not a licence to deselect them.",
            ]

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
    # A false declaration does not make the map incomplete, but it is not tolerated
    # either: the marker narrows a safety check and has to stay true to be worth reading.
    return 0 if (non_trivial and complete and not overclaimed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
