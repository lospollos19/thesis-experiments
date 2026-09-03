"""The step 04 campaign: run one condition over a chunk of the mutation corpus.

Three conditions, and the whole study is the contrast between them:

| | Arm | Selector | Expected |
|---|---|---|---|
| A | `variant/inline` | testmon alone | safe, near-total over-selection |
| B | `variant/extern` | testmon alone | unsafe, empty selection |
| C | `variant/extern` | testmon ∪ kernel-aware | the contribution |

For each mutation this records what was selected and how long the device was occupied.
Whether the selection was *safe* is decided later, against `D(m)` from R2 — this tool
deliberately does not know `D(m)`. A campaign that could see the answer while choosing
what to run would be worth nothing.

    python tools/campaign.py --condition C --chunk 1/2 --out campaign-C-1.json

## What is timed, and why it is timed that way

Occupancy is measured as **selection + execution**, both on the device:

* *selection* — `pytest --testmon --collect-only`, which is what computing the answer
  costs, plus the kernel layer for condition C. Deciding not to run tests is not free
  and belongs in the figure.
* *execution* — `pytest <selected node ids>` for the ids that came out.

Every condition is executed the same way, by node id, so the three are comparable.
The consequence, stated rather than hidden: this excludes testmon's own coverage
tracing during execution, which the real workflow would pay in A and B. It is a
constant offset against those two arms, and it flatters them, not condition C.

## The database

testmon's database is built once per job, on the **unmutated** tree, and snapshotted.
Every mutation restores that snapshot first: a selection run writes to the database, so
without the restore each mutation would be selecting against the state the previous one
left behind. The snapshot is checkpointed out of WAL first, for the reason step 05
measured — the content sits in ``-wal`` until then, and copying the main file alone
ships an empty database that silently re-runs everything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kernel_select  # noqa: E402
import mutate  # noqa: E402

CONDITIONS = {"A": "inline", "B": "extern", "C": "extern"}
DB = ".testmondata"


def _pytest(root: Path, *args: str) -> tuple[str, float]:
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return proc.stdout, round(time.monotonic() - started, 2)


def _nodeids(text: str) -> list[str]:
    return sorted(
        {line.strip() for line in text.splitlines() if "::" in line and " " not in line.strip()}
    )


def _collect(root: Path, *args: str) -> list[str]:
    out, _ = _pytest(root, "tests/", "-q", "--collect-only", *args)
    return _nodeids(out)


def _snapshot_db(root: Path, into: Path) -> None:
    subprocess.run(
        [sys.executable, "tools/checkpoint_testmon_db.py", DB], cwd=root, capture_output=True
    )
    into.mkdir(parents=True, exist_ok=True)
    for path in root.glob(f"{DB}*"):
        shutil.copy2(path, into / path.name)


def _restore_db(root: Path, snapshot: Path) -> None:
    for path in root.glob(f"{DB}*"):
        path.unlink()
    for path in snapshot.glob(f"{DB}*"):
        shutil.copy2(path, root / path.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--map", type=Path, help="kernel_deps.json, required for condition C")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--chunk")
    parser.add_argument(
        "--only",
        help="comma-separated mutation ids, for re-measuring a subset after a fix",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run each selected set this many times, recording every timing",
    )
    parser.add_argument(
        "--skip-unselected",
        action="store_true",
        help="do not measure the reference; another job measures it",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="measure only the unselected suite, --repeats times, and stop",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    arm = mutate.detect_arm(root)
    if arm != CONDITIONS[args.condition]:
        raise SystemExit(
            f"condition {args.condition} runs on {CONDITIONS[args.condition]!r} but this "
            f"tree is {arm!r}; the checkout is on the wrong branch"
        )
    if args.condition == "C" and not args.map:
        raise SystemExit("condition C needs --map: it is the kernel-aware layer's only input")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()

    # The manifest is the unmutated baseline, by definition. The *collection* is not:
    # it is re-taken after each mutation, below.
    collected = _collect(root)
    gpu_collected = _collect(root, "-m", "gpu")
    manifest = kernel_select.build_manifest(head)
    kernel_map = json.loads(args.map.read_text(encoding="utf-8")) if args.map else {}

    print(f"condition {args.condition} on {arm} at {head[:7]}", flush=True)
    print(f"collected {len(collected)} ({len(gpu_collected)} gpu)", flush=True)

    # Reference: the unselected suite, the denominator of every saving below. Its own
    # variance matters as much as the selections' — a ratio is no more precise than the
    # quantity it divides by — so it is repeated too, and every timing is kept.
    unselected_all: list[float] = []
    if not args.skip_unselected:
        for i in range(args.repeats):
            _, seconds = _pytest(root, "tests/", "-q", "--tb=no")
            unselected_all.append(seconds)
            print(f"unselected suite [{i + 1}/{args.repeats}]: {seconds}s", flush=True)
    unselected_seconds = (
        sum(unselected_all) / len(unselected_all) if unselected_all else float("nan")
    )

    if args.reference_only:
        args.out.write_text(
            json.dumps(
                {
                    "condition": args.condition,
                    "arm": arm,
                    "commit": head,
                    "reference_only": True,
                    "unselected_seconds": unselected_seconds,
                    "unselected_all": unselected_all,
                    "mutations": {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}: reference only, {len(unselected_all)} timing(s)")
        return 0

    for path in root.glob(f"{DB}*"):
        path.unlink()
    _, db_seconds = _pytest(root, "tests/", "-q", "--tb=no", "--testmon")
    snapshot = root / ".campaign-db"
    _snapshot_db(root, snapshot)
    print(f"testmon database built in {db_seconds}s", flush=True)

    results: dict[str, object] = {}
    if args.only:
        wanted = [m.strip() for m in args.only.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in mutate.BY_ID]
        if unknown:
            raise SystemExit(f"unknown mutation id(s): {', '.join(unknown)}")
        selection = [mutate.BY_ID[m] for m in wanted]
    elif args.chunk:
        selection = _chunk(args.chunk)
    else:
        selection = list(mutate.CORPUS)

    for mutation in selection:
        print(f"--- {mutation.id}", flush=True)
        _restore_db(root, snapshot)
        mutate.apply(mutation, root, arm)
        try:
            # Collected *after* the mutation, on purpose. A mutation can rename a
            # parametrised id — halving MAX_RADIUS turns ...[33] into ...[17] — so a
            # selection drawn from the pre-mutation collection names a test that no
            # longer exists, and pytest then aborts without running anything. Measured:
            # run 31958999896 reported 188 tests selected and 1.48 s of occupancy.
            # Re-collecting is also what a real workflow does: you change the code, then
            # you select from what is there.
            collected = _collect(root)
            gpu_collected = _collect(root, "-m", "gpu")
            kernel_free = set(_collect(root, "-m", "gpu and no_kernel_launch"))
            out, select_seconds = _pytest(root, "tests/", "-q", "--collect-only", "--testmon")
            testmon_selection = set(_nodeids(out))
            if args.condition == "C":
                selected, reasons = kernel_select.select(
                    manifest, kernel_map, collected, gpu_collected, kernel_free, testmon_selection
                )
            else:
                selected, reasons = sorted(testmon_selection & set(collected)), {"rule": "testmon"}

            run_all: list[float] = []
            for _ in range(args.repeats):
                if selected:
                    _, seconds = _pytest(root, "-q", "--tb=no", *selected)
                else:
                    seconds = 0.0
                run_all.append(seconds)
            run_seconds = sum(run_all) / len(run_all)
        finally:
            mutate.revert(mutation, root, arm)

        results[mutation.id] = {
            "selected": selected,
            "n_selected": len(selected),
            "n_selected_gpu": len(set(selected) & set(gpu_collected)),
            "n_testmon": len(testmon_selection),
            "select_seconds": select_seconds,
            "run_seconds": round(run_seconds, 2),
            "run_seconds_all": run_all,
            "repeats": args.repeats,
            "occupancy_seconds": round(select_seconds + run_seconds, 2),
            "reasons": reasons,
        }
        print(
            f"    selected {len(selected)} "
            f"({results[mutation.id]['n_selected_gpu']} gpu), "  # type: ignore[index]
            f"occupancy {results[mutation.id]['occupancy_seconds']}s",  # type: ignore[index]
            flush=True,
        )

    shutil.rmtree(snapshot, ignore_errors=True)
    for path in root.glob(f"{DB}*"):
        path.unlink()

    args.out.write_text(
        json.dumps(
            {
                "condition": args.condition,
                "arm": arm,
                "commit": head,
                "collected": len(collected),
                "gpu_collected": len(gpu_collected),
                "unselected_seconds": unselected_seconds,
                "unselected_all": unselected_all,
                "db_build_seconds": db_seconds,
                "mutations": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    return 0


def _chunk(spec: str) -> list[mutate.Mutation]:
    index, total = (int(part) for part in spec.split("/"))
    return [m for i, m in enumerate(mutate.CORPUS) if i % total == index - 1]


if __name__ == "__main__":
    raise SystemExit(main())
