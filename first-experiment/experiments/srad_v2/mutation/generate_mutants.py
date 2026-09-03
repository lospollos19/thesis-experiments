#!/usr/bin/env python3
"""
Source-level mutant generator for srad_v2 (RQ1, 14 MUTGPU operators).

Mull cannot parse CUDA C; MUTGPU (Zhu & Zaidman) uses a Pyparsing grammar. srad's
mutation points are syntactically local, so we apply targeted textual rules -- one
mutant per mutation point -- and emit a manifest. Pure text: GENERATION needs no
CUDA/GPU and runs anywhere. Operator names follow the MUTGPU paper.

Each mutant = a full copy of the affected source file with exactly one change,
written to  mutation/mutants/<id>/<basename>  plus an entry in mutants.json.
The campaign runner overlays that one file onto a fresh srad_v2 tree and builds it
(Job B = CuPBoP/SIL, Job C = nvcc/HIL). Determinism guard is disabled during the
campaign (mutants are expected to diverge).
"""
from __future__ import annotations
import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent      # experiments/srad_v2/mutation
SRAD = HERE.parent                          # experiments/srad_v2
OUT = HERE / "mutants"

KERNEL = "srad_kernel.cu"
HOST = "srad.cu"


@dataclass
class Mutation:
    id: str
    operator: str
    category: str          # "GPU" | "conventional"
    file: str
    line: int
    original: str          # the matched text
    mutated: str           # the replacement text
    race_dependent: bool   # True => predicted SIL/HIL gap (sync_removal, atom_removal)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _apply(text: str, start: int, end: int, repl: str) -> str:
    return text[:start] + repl + text[end:]


# --- Operator rules ---------------------------------------------------------
# Each rule is (operator, category, race_dependent, generator-fn).
# A generator-fn takes the file text and yields (start, end, replacement, orig, mut).

def op_sync_removal(text):
    # Remove a __syncthreads() call -> the classic race-inducing mutant (H1).
    for m in re.finditer(r"__syncthreads\s*\(\s*\)\s*;", text):
        yield m.start(), m.end(), "; /* MUT sync_removal */", m.group(0), "; /* removed */"


def op_share_removal(text):
    # Drop the __shared__ qualifier -> threads lose the shared tile.
    for m in re.finditer(r"__shared__\s+", text):
        yield m.start(), m.end(), "/* MUT share_removal */ ", "__shared__", "(none)"


def op_gpu_index_replacement(text):
    # threadIdx <-> blockIdx, one occurrence at a time.
    for m in re.finditer(r"\bthreadIdx\b", text):
        yield m.start(), m.end(), "blockIdx", "threadIdx", "blockIdx"
    for m in re.finditer(r"\bblockIdx\b", text):
        yield m.start(), m.end(), "threadIdx", "blockIdx", "threadIdx"


def op_gpu_index_increment(text):
    for m in re.finditer(r"\b(threadIdx|blockIdx)\.([xyz])\b", text):
        yield m.start(), m.end(), f"({m.group(0)} + 1)", m.group(0), f"({m.group(0)} + 1)"


def op_gpu_index_decrement(text):
    for m in re.finditer(r"\b(threadIdx|blockIdx)\.([xyz])\b", text):
        yield m.start(), m.end(), f"({m.group(0)} - 1)", m.group(0), f"({m.group(0)} - 1)"


def op_alloc_swap(text):
    # Swap grid/block in the kernel launch config <<<grid, block>>>.
    for m in re.finditer(r"<<<\s*([^,<>]+?)\s*,\s*([^<>]+?)\s*>>>", text):
        grid, block = m.group(1), m.group(2)
        repl = f"<<<{block}, {grid}>>>"
        yield m.start(), m.end(), repl, m.group(0), repl


def _alloc_pm(text, delta, opname):
    # Increment/decrement a dim3 launch-config constructor arg (execution config).
    for m in re.finditer(r"dim3\s+(\w+)\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)", text):
        var, a, b = m.group(1), m.group(2), m.group(3)
        repl = f"dim3 {var}({a} {delta} 1, {b})"
        yield m.start(), m.end(), repl, m.group(0), repl


def op_alloc_increment(text):
    yield from _alloc_pm(text, "+", "alloc_increment")


def op_alloc_decrement(text):
    yield from _alloc_pm(text, "-", "alloc_decrement")


def op_atom_removal(text):
    # srad has no atomics -> yields nothing (recorded as an inapplicable operator, H4).
    for m in re.finditer(r"\batomic\w+\s*\(", text):
        yield m.start(), m.end(), "/* MUT atom_removal */(", m.group(0), "(non-atomic)"


# Conventional operators -- scoped so they hit real expressions/predicates only,
# never pointer declarations, parameter lists, includes, or address arithmetic
# (index/offset math is already covered by the gpu_index_* operators).
_ADDR_HINT = re.compile(r"index|cols\s*\*")

def _scoped(text, predicate):
    """Yield (line_start_pos, line_text) for lines matching predicate."""
    pos = 0
    for line in text.splitlines(keepends=True):
        if predicate(line):
            yield pos, line
        pos += len(line)

_ASSIGN = re.compile(r"(?<![=<>!])=(?!=)")   # a lone '=', not ==, <=, >=, !=

def _is_assignment_line(line):
    # An executable assignment carrying float math: real '=' (not a comparison),
    # not a directive, not address computation. Excludes `float * J_cuda,` (no '=')
    # and `by == gridDim.y - 1` (comparison, not assignment).
    s = line.lstrip()
    return (_ASSIGN.search(line) is not None
            and not s.startswith("#")
            and not _ADDR_HINT.search(line))

def _is_condition_line(line):
    # Relational/logical operators only make sense inside a branch/loop head.
    # This excludes `#include <...>` and pointer/shift noise.
    s = line.lstrip()
    return not s.startswith("#") and re.search(r"\b(if|while|for)\b", line) is not None


_MATH_SWAP = {"+": "-", "-": "+", "*": "/", "/": "*"}

def op_math_replacement(text):
    # Replace a spaced binary arithmetic operator on a physics assignment line,
    # but NOT inside [...] (that is address arithmetic -> gpu_index territory).
    for base, line in _scoped(text, _is_assignment_line):
        for m in re.finditer(r" ([+\-*/]) ", line):
            if line.count("[", 0, m.start(1)) > line.count("]", 0, m.start(1)):
                continue  # operator sits inside an array subscript
            op = m.group(1)
            s = base + m.start(1)
            yield s, s + 1, _MATH_SWAP[op], op, _MATH_SWAP[op]


_CBR = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}

def op_conditional_boundary_replacement(text):
    # <-> <= etc. (PIT CBR), condition lines only. Longest-match first.
    for base, line in _scoped(text, _is_condition_line):
        for m in re.finditer(r"(<=|>=|<|>)", line):
            op = m.group(1)
            s = base + m.start()
            yield s, s + len(op), _CBR[op], op, _CBR[op]


_NEG = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}

def op_negate_conditional_replacement(text):
    for base, line in _scoped(text, _is_condition_line):
        for m in re.finditer(r"(==|!=|<=|>=|<|>)", line):
            op = m.group(1)
            s = base + m.start()
            yield s, s + len(op), _NEG[op], op, _NEG[op]


_LOGIC = {"&&": "||", "||": "&&"}

def op_logical_replacement(text):
    for base, line in _scoped(text, _is_condition_line):
        for m in re.finditer(r"(&&|\|\|)", line):
            op = m.group(1)
            s = base + m.start()
            yield s, s + len(op), _LOGIC[op], op, _LOGIC[op]


def op_increment_replacement(text):
    for m in re.finditer(r"(\+\+|--)", text):
        op = m.group(1)
        repl = "--" if op == "++" else "++"
        yield m.start(), m.end(), repl, op, repl


# Operator registry: (name, category, race_dependent, fn, target_file)
OPERATORS = [
    ("sync_removal",                    "GPU",          True,  op_sync_removal,                    KERNEL),
    ("share_removal",                   "GPU",          False, op_share_removal,                   KERNEL),
    ("gpu_index_replacement",           "GPU",          False, op_gpu_index_replacement,           KERNEL),
    ("gpu_index_increment",             "GPU",          False, op_gpu_index_increment,             KERNEL),
    ("gpu_index_decrement",             "GPU",          False, op_gpu_index_decrement,             KERNEL),
    ("alloc_swap",                      "GPU",          False, op_alloc_swap,                      HOST),
    ("alloc_increment",                 "GPU",          False, op_alloc_increment,                 HOST),
    ("alloc_decrement",                 "GPU",          False, op_alloc_decrement,                 HOST),
    ("atom_removal",                    "GPU",          True,  op_atom_removal,                    KERNEL),
    ("math_replacement",                "conventional", False, op_math_replacement,                KERNEL),
    ("conditional_boundary_replacement","conventional", False, op_conditional_boundary_replacement,KERNEL),
    ("negate_conditional_replacement",  "conventional", False, op_negate_conditional_replacement,  KERNEL),
    ("logical_replacement",             "conventional", False, op_logical_replacement,             KERNEL),
    ("increment_replacement",           "conventional", False, op_increment_replacement,           KERNEL),
]


def generate():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    sources = {KERNEL: (SRAD / KERNEL).read_text(), HOST: (SRAD / HOST).read_text()}
    manifest: list[dict] = []
    counts: dict[str, int] = {}

    for name, cat, race, fn, target in OPERATORS:
        text = sources[target]
        n = 0
        for start, end, repl, orig, mut in fn(text):
            n += 1
            mid = f"{name}_{n:03d}"
            mutated_text = _apply(text, start, end, repl)
            mdir = OUT / mid
            mdir.mkdir()
            (mdir / target).write_text(mutated_text)
            manifest.append(asdict(Mutation(
                id=mid, operator=name, category=cat, file=target,
                line=_line_of(text, start), original=orig, mutated=mut,
                race_dependent=race,
            )))
        counts[name] = n

    (OUT / "mutants.json").write_text(json.dumps(manifest, indent=2))

    total = len(manifest)
    print(f"Generated {total} mutants into {OUT.relative_to(SRAD.parent.parent)}\n")
    print(f"{'operator':34} {'cat':12} {'count':>5}  race")
    print("-" * 62)
    for name, cat, race, _, _ in OPERATORS:
        print(f"{name:34} {cat:12} {counts[name]:>5}  {'YES' if race else ''}")
    print("-" * 62)
    print(f"{'TOTAL':34} {'':12} {total:>5}")
    return manifest


if __name__ == "__main__":
    generate()
