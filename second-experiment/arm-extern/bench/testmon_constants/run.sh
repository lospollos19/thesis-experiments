#!/usr/bin/env bash
# Where does testmon's dependency tracking stop?
#
# For each binding form in demo/config.py: build the database on a clean tree, change
# that one binding, and ask whether testmon would re-run **the test that depends on it**.
# Repeated, because the answer to a weaker question — "what does it select?" — is not
# stable.
#
# No device, no CUDA, a few seconds.
#
#   bench/testmon_constants/run.sh          # REPEATS=5 by default
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$(cd "$HERE/../.." && poetry env info --executable 2>/dev/null)}"
REPEATS="${REPEATS:-5}"
if [ ! -x "$PY" ]; then
  echo "no interpreter: set PY=/path/to/python (needs pytest and pytest-testmon)" >&2
  exit 1
fi
cd "$HERE" || exit 1

clean() { rm -rf .testmondata* demo/__pycache__ tests/__pycache__ .pytest_cache; }

check() {
  local label="$1" edit="$2" dependent="$3" hits=0
  for _ in $(seq 1 "$REPEATS"); do
    clean
    PYTHONPATH="$HERE" "$PY" -m pytest -q --testmon >/dev/null 2>&1
    cp demo/config.py demo/config.py.bak
    # BSD and GNU sed disagree on -i; write through a temp file instead.
    sed "$edit" demo/config.py > demo/config.new && mv demo/config.new demo/config.py
    if PYTHONPATH="$HERE" "$PY" -m pytest -q --collect-only --testmon 2>&1 \
       | grep -q "$dependent"; then
      hits=$((hits + 1))
    fi
    mv demo/config.py.bak demo/config.py
  done
  printf '%-26s %-48s %s/%s\n' "$label" "$dependent" "$hits" "$REPEATS"
}

printf '%-26s %-48s %s\n' "change" "the test that depends on it" "re-run"
printf '%-26s %-48s %s\n' "--------------------------" \
  "------------------------------------------------" "------"

check "SIZE = 10 -> 11"  's/^SIZE = 10/SIZE = 11/' \
  "test_reads_constant_as_attribute"
check "LIMIT = 5 -> 6"   's/^LIMIT = 5/LIMIT = 6/' \
  "test_reads_import_time_derived_value"
check "NAMES + element"  's/^NAMES = \["a", "b"\]/NAMES = ["a","b","c"]/' \
  "test_reads_mutable_structure"
check "FACTOR = 3 -> 4"  's/^FACTOR = 3/FACTOR = 4/' \
  "test_calls_function_with_default_argument"
check "body of limit_plus" 's/    return LIMIT + n/    return LIMIT + n + 0/' \
  "test_calls_function_reading_constant_at_runtime"
clean
rm -f demo/config.py.bak

cat <<'NOTE'

A test is linked to a file if and only if it executed a line of that file during its own
run. Reading a value does not count; calling a function does. Module-level bindings run
once at import, before any test, so a test that only reads one has no edge to the module
and is never re-run — including when the value it asserts on is the one that changed.

Secondary observation, not part of the claim: *which* non-dependent test gets dragged in
alongside varies between runs. That is noise in the over-selection direction. The column
above does not move.
NOTE
