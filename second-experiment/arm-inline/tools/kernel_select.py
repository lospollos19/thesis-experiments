"""The kernel-aware selector: condition C.

testmon selects over the Python import and coverage graph. A ``.cu`` file is not a
Python module and never executes in the interpreter, so that graph has no edge to it —
which is the failure this study measures. This tool supplies the missing edges. It does
**not** replace testmon: the selection is the union of the two, because Python-side
dependencies are still real and still testmon's job.

    python tools/kernel_select.py manifest --out manifest.json
    python tools/kernel_select.py select --manifest manifest.json \\
        --map kernel_deps.json --collected collected.txt \\
        --testmon-selection testmon.txt --out selected.txt

``manifest`` runs on the **unmutated** tree, next to the tracing that produces the map,
and records what each kernel's source hashed to. ``select`` runs afterwards, compares,
and emits the node ids to run. Both read the sources through ``get_kernel_source``, the
same single indirection the processor uses, so one implementation covers both arms
without ever branching on which one it is in.

## The rules, and why each is the safe direction

* **R1 — an edge to a changed kernel selects the test.** The map, used as intended.
* **R2 — a collected node id absent from the map is selected.** Absence is not evidence
  of independence. It is also not hypothetical: parametrised ids embed values, so
  ``test_gaussian_blur_matches_numpy[33]`` becomes ``[17]`` when ``MAX_RADIUS`` is
  halved and the id in the map ceases to exist. The natural implementation — look the
  test up, skip it if it has no edge to a changed kernel — deselects it exactly when it
  matters.
* **R3 — any kernel change selects every test declared ``no_kernel_launch``.** Those
  tests launch nothing and are absent from the map by design, but they depend on all
  five kernels *compiling*. A launch-site tracer cannot see that edge, so it is declared
  rather than traced.
* **R4 — a change to the shared geometry selects everything.** ``common.cuh`` reaches
  the launch configuration, the halo guard and ``MAX_KERNEL_SIZE``, which is part of the
  documented contract. No test in this suite can be shown independent of it, so the
  honest answer is the whole suite. Reported as over-selection rather than hidden.

R2 and R3 are the second edge type that R1's question 3 turned from a hypothesis into a
requirement; R4 is the constant dependency it named.

A manifest is refused when it was built from different content than the map it is used
with — the same discipline as testmon's database, for the same reason: a stale baseline
silently selects against a state that no longer exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motion_pipeline.processing import kernels  # noqa: E402

GEOMETRY = "__geometry__"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(commit: str | None) -> dict[str, object]:
    """Hash every kernel source plus the shared geometry, on the current tree.

    The reload is not optional. On ``variant/extern`` the loader binds
    ``_COMMON_SOURCE`` and the tile constants at **import** time, so a process that
    imported the module and then edited ``common.cuh`` — which is exactly what the
    campaign does — hashes a header it read minutes ago. The ``.cu`` files are re-read
    per call and were fine; the shared header was not, and R4 never fired.

    That is the same defect as testmon's, in the tool written to cover it: a value bound
    at import is invisible to anything that does not import again. Measured, in run
    `31955528366`: ``geometry_changed: false`` for a mutation that edits nothing else.
    """
    importlib.reload(kernels)
    sources = {name: _digest(kernels.get_kernel_source(name)) for name in kernels.KERNEL_NAMES}
    geometry = _digest(f"{kernels.TILE_W},{kernels.TILE_H},{kernels.MAX_RADIUS}")
    return {"commit": commit, "kernels": sources, GEOMETRY: geometry}


def changed_against(manifest: dict) -> tuple[set[str], bool]:
    """Return the kernels whose source moved, and whether the geometry moved."""
    current = build_manifest(None)
    changed = {
        name
        for name, digest in current["kernels"].items()  # type: ignore[union-attr]
        if manifest["kernels"].get(name) != digest
    }
    # A kernel present now and absent from the manifest counts as changed above; one
    # present in the manifest and gone now is a structural change, not a content one.
    changed |= set(manifest["kernels"]) - set(current["kernels"])  # type: ignore[arg-type]
    return changed, manifest[GEOMETRY] != current[GEOMETRY]


def select(
    manifest: dict,
    kernel_map: dict[str, list[str]],
    collected: list[str],
    gpu_collected: list[str],
    kernel_free: set[str],
    testmon_selection: set[str],
) -> tuple[list[str], dict[str, object]]:
    """Apply the four rules, then union with whatever testmon already selected.

    Two universes, and the distinction is load-bearing. The map only ever describes
    ``gpu``-marked tests, so R1, R2 and R3 range over ``gpu_collected``: a CPU test is
    absent from the map because it launches nothing, not because it was missed, and
    treating that absence as a dependency would select the entire CPU suite on every
    kernel edit. R4 ranges over everything, because the tile geometry is the one piece
    of device state reachable from Python without a launch — ``MAX_KERNEL_SIZE`` is a
    module attribute, and a CPU test asserting on it changes behaviour when the header
    moves. On ``variant/extern`` nothing Python-side changes, so testmon cannot cover
    that test and this rule is the only thing that does.
    """
    changed, geometry_changed = changed_against(manifest)
    reasons: dict[str, object] = {
        "changed_kernels": sorted(changed),
        "geometry_changed": geometry_changed,
    }

    if geometry_changed:  # R4 — the whole suite, gpu-marked or not
        selected = set(collected)
        reasons["rule"] = "R4 — shared geometry changed, the whole suite is selected"
    else:
        # R2 covers ids the map does not explain, within the map's own universe. A
        # declared no_kernel_launch test is absent by design and is explained: it belongs
        # to R3 and is selected when a kernel changes, not merely because it has no edge.
        # Without this exclusion the selector picks those three tests even when nothing
        # at all has changed.
        selected = {n for n in gpu_collected if n not in kernel_map and n not in kernel_free}
        for nodeid, used in kernel_map.items():
            if changed & set(used):  # R1
                selected.add(nodeid)
        if changed:  # R3
            selected |= kernel_free
        reasons["rule"] = "R1/R2/R3"

    reasons["kernel_selected"] = len(selected)
    # Union, not intersection: testmon's edges are real edges over Python code, and this
    # tool knows nothing about them.
    selected |= testmon_selection & set(collected)
    reasons["testmon_selected"] = len(testmon_selection)
    reasons["total_selected"] = len(selected)
    reasons["collected"] = len(collected)
    return sorted(selected), reasons


def _read_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    build = sub.add_parser("manifest", help="hash the kernel sources of the current tree")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--commit", help="the content state this manifest describes")

    run = sub.add_parser("select", help="emit the node ids to run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--map", type=Path, required=True)
    run.add_argument("--collected", type=Path, required=True, help="every collected node id")
    run.add_argument(
        "--gpu-collected",
        type=Path,
        help="the gpu-marked subset, the map's universe; defaults to --collected",
    )
    run.add_argument("--kernel-free", type=Path)
    run.add_argument("--testmon-selection", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--map-commit", help="reject the manifest if it names another commit")

    args = parser.parse_args(argv)

    if args.action == "manifest":
        manifest = build_manifest(args.commit)
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}: {len(manifest['kernels'])} kernels, commit {args.commit}")
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.map_commit and manifest.get("commit") not in (None, args.map_commit):
        raise SystemExit(
            f"the manifest was built at {manifest['commit']} but the map at "
            f"{args.map_commit}; a baseline from another content state selects against "
            "a tree that no longer exists"
        )

    kernel_map = json.loads(args.map.read_text(encoding="utf-8"))
    collected = _read_ids(args.collected)
    if not collected:
        raise SystemExit("nothing was collected; refusing to report an empty selection")

    gpu_collected = _read_ids(args.gpu_collected) if args.gpu_collected else collected
    unknown = set(gpu_collected) - set(collected)
    if unknown:
        raise SystemExit(
            f"{len(unknown)} gpu node id(s) are not in the full collection; the two "
            "lists come from different content states"
        )

    selected, reasons = select(
        manifest,
        kernel_map,
        collected,
        gpu_collected,
        set(_read_ids(args.kernel_free)),
        set(_read_ids(args.testmon_selection)),
    )
    args.out.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(json.dumps(reasons, indent=2))
    print(f"wrote {args.out}: {len(selected)} of {len(collected)} tests selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
