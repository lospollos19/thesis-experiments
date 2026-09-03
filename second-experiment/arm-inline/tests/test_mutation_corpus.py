"""The mutation corpus is instrumentation, so it is tested like instrumentation.

``tools/mutate.py`` defines the semantic edits whose detection produces the safety
violation rate. If a kernel is reworked and a pattern stops matching, the campaign
would silently mutate nothing and report a perfect violation rate — the failure mode
that flatters the result. These tests fail on corpus drift instead.

Everything runs on sources materialised from **git HEAD**, not from the working tree,
and in a temporary directory. Both halves of that matter:

* editing the real sources would leave the tree dirty on failure, and the arms are
  compared at identical content;
* during a measurement campaign the working tree *is* mutated, and a check that read
  it would fail on every mutation. That failure would land in ``D(m)`` — the set of
  tests a mutation breaks — inflating the violation-rate denominator with a test that
  says nothing about device behaviour, and destroying the equivalent-mutant control by
  making it detectable. Reading HEAD makes this check independent of the campaign it
  is checking.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import mutate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARM = mutate.detect_arm(REPO_ROOT)


def _pristine_tree(mutation: mutate.Mutation, tmp_path: Path) -> Path:
    """Materialise this mutation's target file at its HEAD content, under ``tmp_path``.

    ``git show HEAD:<path>`` rather than a copy: a mutation is an uncommitted working-tree
    edit, so HEAD is the unmutated content by construction.
    """
    relative = mutate.target_path(mutation, Path("."), ARM)
    blob = subprocess.run(
        ["git", "show", f"HEAD:./{relative.as_posix()}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if blob.returncode != 0:
        pytest.fail(f"{mutation.id}: cannot read {relative} from HEAD: {blob.stderr.strip()}")
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(blob.stdout, encoding="utf-8")
    return tmp_path


def test_ids_are_unique():
    ids = [m.id for m in mutate.CORPUS]
    assert len(ids) == len(set(ids))


def test_the_corpus_covers_every_kernel():
    """A kernel no mutation touches contributes nothing to the violation rate."""
    from motion_pipeline.processing import kernels

    covered = {m.kernel for m in mutate.CORPUS if m.kernel is not None}
    assert covered == set(kernels.KERNEL_NAMES)


def test_the_corpus_keeps_an_equivalent_control():
    """At least one mutation must change device code without changing behaviour.

    It is the control that tells a selector's silence apart from a selector's miss:
    for an equivalent mutant no test can fail, so neither selecting nor skipping is a
    safety statement, and it is excluded from the violation denominator.
    """
    assert any(not m.behaviour_changing for m in mutate.CORPUS)
    assert any(m.behaviour_changing for m in mutate.CORPUS)


@pytest.mark.parametrize("mutation", mutate.CORPUS, ids=lambda m: m.id)
def test_each_mutation_applies_once_and_reverts_exactly(mutation, tmp_path: Path):
    tree = _pristine_tree(mutation, tmp_path)
    path = mutate.target_path(mutation, tree, ARM)

    before = path.read_text(encoding="utf-8")
    find, _ = mutation.patterns(ARM)
    assert before.count(find) == 1, (
        f"{mutation.id}: pattern occurs {before.count(find)} time(s) in {path.name}, "
        "expected exactly 1 — the corpus has drifted from the sources"
    )

    mutate.apply(mutation, tree, ARM)
    mutated = path.read_text(encoding="utf-8")
    assert mutated != before, f"{mutation.id} applied but changed nothing"

    mutate.revert(mutation, tree, ARM)
    assert path.read_text(encoding="utf-8") == before, f"{mutation.id} did not revert cleanly"


def test_detect_arm_matches_this_checkout():
    """The arm is inferred from the tree, never passed as a flag that can be wrong."""
    from motion_pipeline.processing import kernels

    expected = "extern" if hasattr(kernels, "KERNEL_DIR") else "inline"
    assert expected == ARM
