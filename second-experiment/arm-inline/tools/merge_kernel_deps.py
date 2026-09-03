"""Union several traced sessions into one test -> kernel map.

A single session can miss a launch that only happens on a rare branch (a parametrised
case, an error path). The union over several sessions is the conservative direction:
adding a dependency that is not always exercised over-selects, dropping one that is
produces a silent safety violation. Never take an intersection here.

    python tools/merge_kernel_deps.py --out kernel_deps.json run1.json run2.json run3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(paths: list[Path]) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for nodeid, kernels in data.items():
            merged.setdefault(nodeid, set()).update(kernels)
    return {nodeid: sorted(names) for nodeid, names in sorted(merged.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="per-session trace files")
    parser.add_argument("--out", required=True, type=Path, help="merged map")
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=3,
        help="fail below this many inputs; one session is not a campaign",
    )
    args = parser.parse_args(argv)

    present = [p for p in args.inputs if p.is_file()]
    for path in args.inputs:
        if not path.is_file():
            print(f"missing trace file, ignored: {path}")
    if len(present) < args.min_sessions:
        print(f"only {len(present)} session(s) present, {args.min_sessions} required")
        return 1

    merged = merge(present)
    args.out.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")

    # A session that adds nothing is evidence the union has converged; a session that
    # adds a lot means the campaign is too short to trust.
    running: dict[str, set[str]] = {}
    for path in present:
        before = sum(len(v) for v in running.values())
        for nodeid, kernels in json.loads(path.read_text(encoding="utf-8")).items():
            running.setdefault(nodeid, set()).update(kernels)
        after = sum(len(v) for v in running.values())
        print(f"{path.name}: +{after - before} edge(s), {after} total")

    print(f"wrote {args.out}: {len(merged)} tests, {sum(len(v) for v in merged.values())} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
