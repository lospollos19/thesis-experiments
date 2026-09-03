"""The mutation corpus for RQ2, and the tool that applies it.

A *mutation* here is a change to **device-code semantics**, defined once and expressed
in whichever storage the current arm uses. That is the whole point: the two violation
rates only compare if the same semantic change is applied to the inline string and to
the ``.cu`` file. Two separately written edits would compare two experiments.

Comparability is obtained by construction rather than by discipline. The CUDA-C text is
byte-identical in both arms — only the file holding it differs — so one ``find``/
``replace`` pair drives both, and this tool resolves the target file from the tree it is
run in. The one exception is the tile geometry, which is a Python constant on the inline
arm and a ``#define`` on the external arm; those mutations carry an ``inline_*`` form,
and that asymmetry is not an inconvenience but the object of study.

Every application asserts the pattern occurs **exactly once**. A pattern matching zero
or two places is not a controlled edit, and a corpus that has drifted from the sources
must fail loudly rather than mutate something adjacent.

    python tools/mutate.py list
    python tools/mutate.py apply absdiff-drop-abs
    python tools/mutate.py revert absdiff-drop-abs
    python tools/mutate.py verify          # every mutation applies and reverts cleanly

## What the corpus is for, and the distinction that decides the numbers

The safety violation rate is the share of *behaviour-changing* edits for which the
selector did not re-run a test that would have caught the change. Computing it needs
two sets per mutation ``m``:

* ``D(m)`` — the tests that actually fail under ``m``, measured by running the **whole**
  suite with no selection. Ground truth, never assumed from the map.
* ``S(m)`` — the tests the selector picks.

A violation is ``D(m) != {}`` and ``D(m) & S(m) == {}``. When ``D(m)`` is empty the
mutation is invisible to the suite: that is a statement about test adequacy, not about
the selector, and it must be excluded from the violation denominator and reported
separately. Conflating the two would credit or blame the selector for the suite's blind
spots.

``D(m)`` is also the arms' comparability check. The same semantic mutation must produce
the same ``D(m)`` on both arms; if it does not, the two expressions are not the same
change and the corpus entry is wrong.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROCESSING = Path("src/motion_pipeline/processing")
KERNELS_PY = PROCESSING / "kernels.py"
CUDA_DIR = PROCESSING / "cuda"


@dataclass(frozen=True)
class Mutation:
    """One semantic change to device code, expressed for both storage modes."""

    id: str
    kernel: str | None
    kind: str
    behaviour_changing: bool
    summary: str
    find: str
    replace: str
    #: Only for values that are a Python constant inline and a #define externally.
    inline_find: str | None = None
    inline_replace: str | None = None

    def patterns(self, arm: str) -> tuple[str, str]:
        if arm == "inline" and self.inline_find is not None:
            assert self.inline_replace is not None
            return self.inline_find, self.inline_replace
        return self.find, self.replace


CORPUS: tuple[Mutation, ...] = (
    # ---------------------------------------------------------------- grayscale
    Mutation(
        id="grayscale-swap-bgr-weights",
        kernel="grayscale_bgr_u8",
        kind="operator",
        behaviour_changing=True,
        summary="swap the blue and red luminance weights (a channel-order bug)",
        find=(
            "dst[y * width + x] = 0.114f * (float)src[px + 0]\n"
            "                       + 0.587f * (float)src[px + 1]\n"
            "                       + 0.299f * (float)src[px + 2];"
        ),
        replace=(
            "dst[y * width + x] = 0.299f * (float)src[px + 0]\n"
            "                       + 0.587f * (float)src[px + 1]\n"
            "                       + 0.114f * (float)src[px + 2];"
        ),
    ),
    Mutation(
        id="grayscale-perturb-weight",
        kernel="grayscale_bgr_u8",
        kind="numeric",
        behaviour_changing=True,
        summary="perturb the green luminance weight by 0.002",
        find="+ 0.587f * (float)src[px + 1]",
        replace="+ 0.585f * (float)src[px + 1]",
    ),
    # ------------------------------------------------------------- convolutions
    Mutation(
        id="conv1d-h-drop-last-tap",
        kernel="conv1d_horizontal",
        kind="off-by-one",
        behaviour_changing=True,
        summary="drop the last tap of the horizontal accumulation loop",
        # The loop bound alone appears in both convolutions; the accumulation line
        # pins it to the horizontal one.
        find=(
            "for (int k = 0; k < 2 * radius + 1; ++k) {\n"
            "        acc += kern[k] * tile[threadIdx.y][threadIdx.x + k];"
        ),
        replace=(
            "for (int k = 0; k < 2 * radius; ++k) {\n"
            "        acc += kern[k] * tile[threadIdx.y][threadIdx.x + k];"
        ),
    ),
    Mutation(
        id="conv1d-h-border-wrap",
        kernel="conv1d_horizontal",
        kind="boundary",
        behaviour_changing=True,
        summary="horizontal border rule: clamp to edge becomes wrap-around",
        find="gx = min(max(gx, 0), width - 1);",
        replace="gx = (gx % width + width) % width;",
    ),
    Mutation(
        id="conv1d-v-border-wrap",
        kernel="conv1d_vertical",
        kind="boundary",
        behaviour_changing=True,
        summary="vertical border rule: clamp to edge becomes wrap-around",
        find="gy = min(max(gy, 0), height - 1);",
        replace="gy = (gy % height + height) % height;",
    ),
    # ---------------------------------------------------------------- absdiff
    Mutation(
        id="absdiff-drop-abs",
        kernel="absdiff_f32",
        kind="operator",
        behaviour_changing=True,
        summary="signed difference instead of absolute difference",
        find="dst[i] = fabsf(a[i] - b[i]);",
        replace="dst[i] = a[i] - b[i];",
    ),
    # -------------------------------------------------------------- threshold
    Mutation(
        id="threshold-boundary",
        kernel="threshold_u8",
        kind="boundary",
        behaviour_changing=True,
        summary="threshold comparison becomes inclusive (> becomes >=)",
        find="dst[i] = (src[i] > value) ? 255 : 0;",
        replace="dst[i] = (src[i] >= value) ? 255 : 0;",
    ),
    # --------------------------------------------------------- tile geometry
    # The constant case, and the reason a launch-site tracer is not sufficient on its
    # own: no kernel launch changes, but MAX_KERNEL_SIZE — a documented part of the
    # device contract — does.
    Mutation(
        id="geometry-halve-max-radius",
        kernel=None,
        kind="constant",
        behaviour_changing=True,
        summary="halve MAX_RADIUS: the largest supported blur drops from 33 to 17 taps",
        find="#define MAX_RADIUS 16",
        replace="#define MAX_RADIUS 8",
        inline_find="MAX_RADIUS = 16",
        inline_replace="MAX_RADIUS = 8",
    ),
    # Deliberately behaviour-preserving: block geometry changes, results do not. Kept
    # in the corpus as a control — an equivalent mutant that no test can detect, so
    # neither selecting nor skipping is a safety statement. Excluded from the
    # violation denominator; see the module docstring.
    Mutation(
        id="geometry-narrow-tile-width",
        kernel=None,
        kind="equivalent",
        behaviour_changing=False,
        summary="halve TILE_W: different block geometry, identical results",
        find="#define TILE_W 32",
        replace="#define TILE_W 16",
        inline_find="TILE_W = 32",
        inline_replace="TILE_W = 16",
    ),
)

BY_ID = {m.id: m for m in CORPUS}


def detect_arm(root: Path) -> str:
    """Infer the arm from the tree rather than taking it as a flag.

    A flag can be passed wrongly and would silently mutate nothing on one arm; the
    presence of the ``.cu`` directory cannot.
    """
    return "extern" if (root / CUDA_DIR).is_dir() else "inline"


def target_path(mutation: Mutation, root: Path, arm: str) -> Path:
    if arm == "inline":
        return root / KERNELS_PY
    if mutation.kernel is None:
        return root / CUDA_DIR / "common.cuh"
    return root / CUDA_DIR / f"{mutation.kernel}.cu"


def _substitute(path: Path, find: str, replace: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(find)
    if count != 1:
        raise SystemExit(
            f"{path}: pattern occurs {count} time(s), expected exactly 1.\n"
            f"  pattern: {find!r}\n"
            "The corpus has drifted from the sources; fix the corpus rather than "
            "loosening the match."
        )
    path.write_text(source.replace(find, replace), encoding="utf-8")


def apply(mutation: Mutation, root: Path, arm: str) -> Path:
    find, replace = mutation.patterns(arm)
    path = target_path(mutation, root, arm)
    _substitute(path, find, replace)
    return path


def revert(mutation: Mutation, root: Path, arm: str) -> Path:
    find, replace = mutation.patterns(arm)
    path = target_path(mutation, root, arm)
    _substitute(path, replace, find)
    return path


def _verify(root: Path, arm: str) -> int:
    """Apply and revert every mutation, checking the file returns byte-identical."""
    failures = 0
    for mutation in CORPUS:
        path = target_path(mutation, root, arm)
        before = path.read_text(encoding="utf-8")
        try:
            apply(mutation, root, arm)
        except SystemExit as exc:
            print(f"FAIL  {mutation.id}: {exc}")
            failures += 1
            continue
        mutated = path.read_text(encoding="utf-8")
        revert(mutation, root, arm)
        after = path.read_text(encoding="utf-8")
        if mutated == before:
            print(f"FAIL  {mutation.id}: applying it changed nothing")
            failures += 1
        elif after != before:
            print(f"FAIL  {mutation.id}: revert did not restore {path}")
            path.write_text(before, encoding="utf-8")
            failures += 1
        else:
            print(f"ok    {mutation.id}  ({path.name})")
    changing = sum(1 for m in CORPUS if m.behaviour_changing)
    print(
        f"\n{len(CORPUS)} mutation(s) on arm {arm!r}: "
        f"{changing} behaviour-changing, {len(CORPUS) - changing} equivalent control(s)"
    )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "apply", "revert", "verify"])
    parser.add_argument("mutation_id", nargs="?")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    root = args.root.resolve()
    arm = detect_arm(root)

    if args.action == "list":
        print(f"arm: {arm}\n")
        for m in CORPUS:
            flag = " " if m.behaviour_changing else "="
            target = target_path(m, root, arm).name
            print(f"{flag} {m.id:30} {m.kind:11} {target:20} {m.summary}")
        print("\n'=' marks an equivalent mutant: device code changes, behaviour does not.")
        return 0

    if args.action == "verify":
        return 1 if _verify(root, arm) else 0

    if not args.mutation_id:
        parser.error(f"{args.action} needs a mutation id (see `list`)")
    mutation = BY_ID.get(args.mutation_id)
    if mutation is None:
        parser.error(f"unknown mutation {args.mutation_id!r}; known: {', '.join(BY_ID)}")

    path = apply(mutation, root, arm) if args.action == "apply" else revert(mutation, root, arm)
    print(f"{args.action}ed {mutation.id} in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
