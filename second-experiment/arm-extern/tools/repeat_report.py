"""Mean, spread and saving with propagated uncertainty, from a repeated campaign.

A saving is a ratio of two measured quantities, so it is no more precise than either.
Reporting ``12.1 %`` from one timing of the selection and one of the reference states a
precision neither has. This turns the repeated timings into an interval.

    python tools/repeat_report.py campaign-C-*.json --out verdict-repeats.md

Uses the sample standard deviation and Student's t for the 95 % interval, since n is
small — with five repeats the normal approximation understates the interval by about
25 %. The saving's uncertainty is propagated from both terms:

    saving = 1 - s/r      →      σ_saving = (s/r) · √((σ_s/s)² + (σ_r/r)²)

The reference is measured in its own job and shared by every selection, which is why it
carries its own row: a drift in the machine between the reference job and a selection
job is a systematic error this design can show but not remove.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# Two-sided 95 %, degrees of freedom n-1.
T95 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _stats(values: list[float]) -> tuple[float, float, float]:
    """Mean, sample standard deviation, and half-width of the 95 % interval."""
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = statistics.stdev(values)
    return mean, sd, T95.get(n - 1, 1.96) * sd / math.sqrt(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    reference: list[float] = []
    selections: dict[str, tuple[int, list[float]]] = {}
    for path in args.files:
        data = json.loads(path.read_text(encoding="utf-8"))
        reference.extend(data.get("unselected_all") or [])
        for mid, entry in data["mutations"].items():
            timings = entry.get("run_seconds_all") or [entry["run_seconds"]]
            selections[mid] = (entry["n_selected"], [t + entry["select_seconds"] for t in timings])

    if not reference:
        raise SystemExit("no unselected reference in these files; the saving has no denominator")

    r_mean, r_sd, r_ci = _stats(reference)
    lines = [
        "## Repeated campaign — occupancy with intervals",
        "",
        f"Unselected reference: **{r_mean:.1f} s** ± {r_ci:.1f} (95 %), "
        f"sd {r_sd:.2f} s over n = {len(reference)}, "
        f"relative spread **{100 * r_sd / r_mean:.2f} %**",
        "",
        "| Selection | tests | occupancy | 95 % interval | saving |",
        "|---|---|---|---|---|",
    ]
    for mid, (n_sel, timings) in sorted(selections.items(), key=lambda kv: kv[1][0]):
        mean, sd, ci = _stats(timings)
        saving = 1 - mean / r_mean
        # Propagated from both terms; a ratio inherits the uncertainty of its divisor.
        if not math.isnan(sd) and mean and r_mean:
            rel = math.sqrt((sd / mean) ** 2 + (r_sd / r_mean) ** 2)
            sigma = (mean / r_mean) * rel
            band = f"± {100 * sigma:.1f} pt"
        else:
            band = "—"
        lines.append(
            f"| `{mid}` | {n_sel} | {mean:.1f} s | ± {ci:.1f} s (n = {len(timings)}) "
            f"| **{100 * saving:.1f} %** {band} |"
        )

    lines += [
        "",
        "The saving's interval is propagated from the selection *and* the reference: a "
        "ratio is no more precise than the quantity it divides by. A saving whose "
        "interval covers zero is not a measured saving.",
    ]
    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
