"""Measure ``D(m)``: which tests each mutation actually breaks.

``D(m)`` is the ground truth the safety violation rate is computed against, and it is
**measured, never inferred from the kernel map** — the map is the thing being judged, so
deriving the denominator from it would make the result circular.

For each mutation: apply it, run the whole suite with no selection, record the failing
node ids, revert. The baseline is measured first, in the same job, and subtracted:

* a test already failing before any mutation appears in every ``D(m)`` and would make
  every mutation look detected;
* so a non-empty baseline invalidates the whole file, and this tool says so rather than
  quietly subtracting and carrying on.

    python tools/ground_truth.py --out ground-truth.json
    python tools/ground_truth.py --out part1.json --chunk 1/2   # split across jobs

Chunking exists because the GPU suite is ~230 s and the job cap is 30 minutes: nine
mutations plus a baseline do not fit in one job on the device.

The tool refuses to start on a dirty tree. A stray edit would be applied to every
mutation's run and land in every ``D(m)``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mutate  # noqa: E402

FAILED_LINE = re.compile(r"^FAILED (\S+)")


def _run_suite(root: Path) -> tuple[list[str], float]:
    """Run the whole suite, returning the failing node ids and the wall time.

    No marker and no keyword expression: this is the unselected reference run, and it
    has to collect everything the campaign could possibly select.
    """
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "-rf"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    failed = [m.group(1) for line in proc.stdout.splitlines() if (m := FAILED_LINE.match(line))]
    return sorted(failed), round(elapsed, 2)


def _tree_is_clean(root: Path) -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and not proc.stdout.strip()


def _selected(chunk: str | None) -> list[mutate.Mutation]:
    if not chunk:
        return list(mutate.CORPUS)
    index, total = (int(part) for part in chunk.split("/"))
    if not 1 <= index <= total:
        raise SystemExit(f"--chunk {chunk}: index must be between 1 and {total}")
    return [m for i, m in enumerate(mutate.CORPUS) if i % total == index - 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--chunk", help="i/n — measure every n-th mutation, offset i")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="only for a chunk whose baseline was measured by another job",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    arm = mutate.detect_arm(root)

    if not _tree_is_clean(root):
        raise SystemExit(
            "the working tree has uncommitted changes; D(m) measured on a dirty tree "
            "attributes those edits to every mutation"
        )

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()

    # Recorded, and checked again by the report. Every mutation here is to device code:
    # on the NumPy backend the gpu tests all skip, so D(m) comes back empty for all of
    # them and the suite looks catastrophically inadequate. A measurement taken without
    # a device is not a weaker measurement, it is a meaningless one.
    backend = subprocess.run(
        [
            sys.executable,
            "-c",
            "from motion_pipeline.processing.gpu_processor import GPUProcessor;"
            "print(GPUProcessor().backend)",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if backend != "pycuda":
        print(f"::warning::backend is {backend!r}: D(m) will be empty for every kernel mutation")

    result: dict[str, object] = {"arm": arm, "commit": head, "backend": backend, "mutations": {}}

    if not args.skip_baseline:
        print("--- baseline (no mutation)", flush=True)
        failed, seconds = _run_suite(root)
        result["baseline"] = {"failed": failed, "duration_s": seconds}
        print(f"    {len(failed)} failing, {seconds}s", flush=True)
        if failed:
            print(
                "::error::the baseline is not green; every D(m) would inherit these "
                f"failures: {', '.join(failed)}",
                flush=True,
            )

    mutations: dict[str, object] = {}
    for mutation in _selected(args.chunk):
        print(f"--- {mutation.id}", flush=True)
        mutate.apply(mutation, root, arm)
        try:
            failed, seconds = _run_suite(root)
        finally:
            # Always, including on an interrupted run: a mutation left in the tree
            # would be attributed to whatever runs next.
            mutate.revert(mutation, root, arm)
        mutations[mutation.id] = {
            "kind": mutation.kind,
            "behaviour_changing": mutation.behaviour_changing,
            "summary": mutation.summary,
            "failed": failed,
            "duration_s": seconds,
        }
        print(f"    {len(failed)} failing, {seconds}s", flush=True)

    result["mutations"] = mutations
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}: {len(mutations)} mutation(s) on arm {arm!r}")

    if not _tree_is_clean(root):
        raise SystemExit("the tree is dirty after the run: a mutation failed to revert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
