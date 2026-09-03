#!/usr/bin/env bash
# Drive bench/compile_cost.py: 2 configs x 2 cache states x REPS repetitions, one fresh
# process each. Writes JSONL to $OUT and never runs two configurations in one process --
# compilation is a start-up cost and the second one in a process would be measured warm.
#
#   bench/run_compile_cost.sh [output.jsonl]
#
# Environment:
#   REPS             repetitions per (config, cache state); default 5
#   BENCH_CACHE_DIR  PyCUDA cache directory the script controls; default under RUNNER_TEMP
#   RUNNER           command prefix for the interpreter; default "poetry run python"

set -uo pipefail   # deliberately not -e: one failed repetition must not lose the rest

OUT="${1:-compile-cost.jsonl}"
REPS="${REPS:-5}"
BENCH_CACHE_DIR="${BENCH_CACHE_DIR:-${RUNNER_TEMP:-/tmp}/rq2-compile-cache}"
RUNNER="${RUNNER:-poetry run python}"
SCRIPT="bench/compile_cost.py"

export BENCH_CACHE_DIR
: > "$OUT"

run() {  # run <config> <cache> <rep> [--warmup]
  local config="$1" cache="$2" rep="$3" extra="${4:-}"
  echo "--- config $config, cache $cache, rep $rep ${extra}"
  # shellcheck disable=SC2086 - RUNNER and extra are intentionally word-split
  if ! $RUNNER "$SCRIPT" --config "$config" --cache "$cache" --rep "$rep" \
       --cache-dir "$BENCH_CACHE_DIR" --out "$OUT" $extra >/dev/null; then
    echo "::warning::config $config, cache $cache, rep $rep failed — no record written"
  fi
}

# shellcheck disable=SC2086
$RUNNER "$SCRIPT" --probe --cache-dir "$BENCH_CACHE_DIR" --out "$OUT" >/dev/null || {
  echo "::error::probe failed — PyCUDA or nvcc unusable, measuring would be meaningless"
  exit 1
}

# Cold: nvcc actually runs. Upper bound, and the state a first run on a fresh machine sees.
# A and B interleave inside a repetition so thermal drift hits both configurations alike.
for rep in $(seq 1 "$REPS"); do
  for config in A B; do
    rm -rf "$BENCH_CACHE_DIR"
    run "$config" cold "$rep"
  done
done

# Hot: the cache is populated once per configuration and then kept. This is the regime a
# measurement campaign actually runs in, since git clean -xdff never touches this directory.
rm -rf "$BENCH_CACHE_DIR"
run A hot 0 --warmup
run B hot 0 --warmup
for rep in $(seq 1 "$REPS"); do
  for config in A B; do
    run "$config" hot "$rep"
  done
done

echo "wrote $OUT ($(wc -l < "$OUT") records)"
