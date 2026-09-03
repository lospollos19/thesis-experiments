"""Cross the campaign's selections with R2's ground truth: the two dependent variables.

    python tools/campaign_report.py --ground-truth gt/*.json \\
        --campaign A=a1.json,a2.json B=b.json C=c1.json,c2.json --out verdict.md

For each condition and each mutation in the denominator:

* **safety** — a violation is ``D(m) != {}`` and ``D(m) & S(m) == {}``: the change was
  detectable and the selector ran nothing that would have caught it. The violation rate
  is over the denominator R2 established, which excludes the equivalent control and any
  mutation the suite cannot see. Those exclusions are about the *suite*, and charging
  them to a selector would be measuring the wrong thing.
* **occupancy** — selection plus execution, against the unselected suite as reference.

The two are reported together and never separately. A condition that selects nothing has
a perfect saving and is worthless; one that selects everything is perfectly safe and
saves nothing. Only the pair says anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_ground_truth(paths: list[Path]) -> tuple[dict[str, set[str]], list[str]]:
    """Merge R2's per-arm chunks into D(m), and derive the denominator."""
    per_arm: dict[str, dict[str, set[str]]] = {}
    changing: dict[str, bool] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for mid, entry in data["mutations"].items():
            per_arm.setdefault(mid, {})[data["arm"]] = set(entry["failed"])
            changing[mid] = entry["behaviour_changing"]

    dm: dict[str, set[str]] = {}
    for mid, arms in per_arm.items():
        distinct = {frozenset(s) for s in arms.values()}
        if len(distinct) > 1:
            raise SystemExit(
                f"{mid}: D(m) differs between arms, so the two expressions are not the "
                "same semantic change and no violation rate over them is comparable"
            )
        dm[mid] = set(next(iter(distinct)))
    denominator = sorted(mid for mid, d in dm.items() if changing[mid] and d)
    return dm, denominator


def _load_campaign(spec: str) -> tuple[str, dict]:
    name, _, files = spec.partition("=")
    merged: dict = {"mutations": {}}
    for path in files.split(","):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["condition"] != name:
            raise SystemExit(f"{path}: holds condition {data['condition']!r}, expected {name!r}")
        merged.update({k: v for k, v in data.items() if k != "mutations"})
        merged["mutations"].update(data["mutations"])
    return name, merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, nargs="+", required=True)
    parser.add_argument("--campaign", nargs="+", required=True, help="X=file[,file]")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    dm, denominator = _load_ground_truth(args.ground_truth)
    conditions = dict(_load_campaign(spec) for spec in args.campaign)

    lines = [
        "## Step 04 — the two dependent variables",
        "",
        f"Denominator: **{len(denominator)}** behaviour-changing mutations the suite detects.",
        "",
        "| Condition | arm | selector | violations | violation rate | mean occupancy | saving |",
        "|---|---|---|---|---|---|---|",
    ]
    detail: dict[str, tuple[list[str], list[str]]] = {}

    for name in sorted(conditions):
        data = conditions[name]
        reference = data["unselected_seconds"]
        violations, occupancies, missing = [], [], []
        for mid in denominator:
            entry = data["mutations"].get(mid)
            if entry is None:
                # Unmeasured is not the same as unsafe. Counting it as a violation would
                # inflate the rate with the campaign's own gaps; it shrinks the base and
                # is reported, so a partial campaign cannot be mistaken for a full one.
                missing.append(mid)
                continue
            occupancies.append(entry["occupancy_seconds"])
            if not (dm[mid] & set(entry["selected"])):
                violations.append(mid)
        measured = len(denominator) - len(missing)
        mean = sum(occupancies) / len(occupancies) if occupancies else 0.0
        rate = len(violations) / measured if measured else 0.0
        selector = "testmon alone" if name in {"A", "B"} else "testmon ∪ kernel-aware"
        note = f" (of {len(denominator)}; {len(missing)} not measured)" if missing else ""
        lines.append(
            f"| **{name}** | `{data['arm']}` | {selector} | {len(violations)}/{measured}{note} "
            f"| **{100 * rate:.1f} %** | {mean:.1f} s | **{100 * (1 - mean / reference):.1f} %** "
            f"(ref {reference:.1f} s) |"
        )
        detail[name] = (violations, missing)

    lines += ["", "### Per-mutation selection sizes", "", "| Mutation | |D(m)| |"]
    header = "|---|---|"
    for name in sorted(conditions):
        lines[-1] += f" {name} selected |"
        header += "---|"
    lines.append(header)
    for mid in denominator:
        row = f"| `{mid}` | {len(dm[mid])} |"
        for name in sorted(conditions):
            entry = conditions[name]["mutations"].get(mid)
            if entry is None:
                row += " — |"
            else:
                caught = "" if dm[mid] & set(entry["selected"]) else " ⚠"
                row += f" {entry['n_selected']}{caught} |"
        lines.append(row)

    for name, (violations, missing) in sorted(detail.items()):
        if violations:
            lines += ["", f"**Condition {name} violated on:**"]
            lines += [f"- `{v}`" for v in violations]
        if missing:
            lines += ["", f"**Condition {name} has no measurement for:**"]
            lines += [f"- `{v}`" for v in missing]

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
    # A violation is the finding for B and a failure for C, so the exit status does not
    # decide it. The report is read, not thresholded.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
