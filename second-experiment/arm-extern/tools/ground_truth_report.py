"""Aggregate the per-arm ``D(m)`` measurements and rule on their validity.

Takes the JSON files produced by ``tools/ground_truth.py`` — one or more per arm, since
the measurement is chunked across jobs — and answers three questions before any
violation rate is computed from them:

1. **Is each baseline green?** A test failing before any mutation appears in every
   ``D(m)`` and makes every mutation look detected. Red.
2. **Is ``D(m)`` the same on both arms?** The same semantic change must break the same
   tests whichever storage holds it. If it does not, the two expressions are not the
   same change and the corpus entry is wrong — the arms would not be comparable. Red.
3. **Which mutations belong in the denominator?** A behaviour-changing mutation with
   ``D(m) = {}`` is invisible to the suite. That is a statement about test adequacy, not
   about any selector, so it is excluded and reported separately rather than counted as
   a mutation the selector handled. The equivalent mutant is expected to be empty and is
   excluded by construction; if it is *not* empty it was misclassified. Red.

    python tools/ground_truth_report.py inline=a.json,b.json extern=c.json \\
        --out verdict-ground-truth.md --denominator denominator.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_arm(spec: str) -> tuple[str, dict]:
    """``arm=file.json,file2.json`` -> merged measurement for that arm."""
    arm, _, files = spec.partition("=")
    if not files:
        raise SystemExit(f"{spec!r}: expected arm=file.json[,file2.json]")
    merged: dict = {"arm": arm, "commit": None, "baseline": None, "backend": None, "mutations": {}}
    for name in files.split(","):
        data = json.loads(Path(name).read_text(encoding="utf-8"))
        if data["arm"] != arm:
            raise SystemExit(f"{name}: contains arm {data['arm']!r}, expected {arm!r}")
        merged["backend"] = data.get("backend")
        if merged["commit"] and data["commit"] != merged["commit"]:
            raise SystemExit(
                f"{name}: measured at {data['commit'][:7]}, but another chunk of the "
                f"same arm was measured at {merged['commit'][:7]} — the chunks are not "
                "the same content state and cannot be merged"
            )
        merged["commit"] = data["commit"]
        if data.get("baseline") is not None:
            merged["baseline"] = data["baseline"]
        merged["mutations"].update(data["mutations"])
    return arm, merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", help="arm=file.json[,file2.json]")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--denominator", type=Path, help="write the usable mutation ids here")
    args = parser.parse_args(argv)

    arms = dict(_load_arm(spec) for spec in args.arms)
    lines = ["## Ground truth — D(m)", ""]
    red = False

    # 0. the measurement has to have run on a device at all. Every mutation is to device
    # code, so on the NumPy backend every gpu test skips and D(m) is empty for all of
    # them — a result that looks like a uniformly inadequate suite and is an artefact.
    for arm, data in arms.items():
        if data.get("backend") != "pycuda":
            lines.append(
                f"- {arm}: **RED** — measured on backend {data.get('backend')!r}, not "
                "`pycuda`. Every kernel mutation is invisible without a device; this "
                "file says nothing about the suite."
            )
            red = True

    # 1. baselines
    for arm, data in arms.items():
        baseline = data.get("baseline")
        if baseline is None:
            lines.append(f"- {arm}: **RED** — no baseline was measured for this arm")
            red = True
        elif baseline["failed"]:
            lines.append(
                f"- {arm}: **RED** — baseline not green, {len(baseline['failed'])} failing; "
                "every D(m) inherits them"
            )
            lines += [f"  - `{n}`" for n in baseline["failed"]]
            red = True
        else:
            lines.append(f"- {arm}: baseline green at `{data['commit'][:7]}`")

    # 2. comparability, and 3. the denominator
    ids = sorted({mid for data in arms.values() for mid in data["mutations"]})
    denominator: list[str] = []
    undetected: list[str] = []

    lines += [
        "",
        "| Mutation | changes behaviour | tests broken, per arm | in denominator |",
        "|---|---|---|---|",
    ]
    for mid in ids:
        entries = {arm: data["mutations"].get(mid) for arm, data in arms.items()}
        missing = [arm for arm, entry in entries.items() if entry is None]
        if missing:
            lines.append(f"| `{mid}` | — | **missing on {', '.join(missing)}** | no |")
            red = True
            continue

        sets = {arm: set(entry["failed"]) for arm, entry in entries.items()}
        sizes = " / ".join(f"{arm}: {len(s)}" for arm, s in sets.items())
        changing = next(iter(entries.values()))["behaviour_changing"]

        distinct = {frozenset(s) for s in sets.values()}
        if len(distinct) > 1:
            lines.append(f"| `{mid}` | {changing} | {sizes} — **differ** | no |")
            red = True
            continue

        detected = bool(next(iter(distinct)))
        if not changing:
            # The equivalent control: expected to break nothing.
            verdict = "no (equivalent control)" if not detected else "**MISCLASSIFIED**"
            red = red or detected
        elif detected:
            denominator.append(mid)
            verdict = "yes"
        else:
            undetected.append(mid)
            verdict = "no (undetectable by the suite)"
        lines.append(f"| `{mid}` | {changing} | {sizes} | {verdict} |")

    lines += [
        "",
        f"- mutations in the violation-rate denominator: **{len(denominator)}**",
        f"- behaviour-changing but undetectable by the suite: **{len(undetected)}**",
    ]
    if undetected:
        lines += [
            "",
            "Undetectable mutations are excluded from the denominator. They measure the",
            "suite's adequacy, not any selector's safety: no selection could have caught",
            "them, so counting them either way would misattribute the result.",
        ]
        lines += [f"  - `{m}`" for m in undetected]

    lines.append("")
    lines.append(f"- verdict: **{'RED' if red else 'GREEN'}**")

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
    if args.denominator:
        args.denominator.write_text(
            json.dumps({"denominator": denominator, "undetected": undetected}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if red else 0


if __name__ == "__main__":
    raise SystemExit(main())
