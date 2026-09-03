#!/usr/bin/env bash
# racecheck diagnostic (RQ1, HIL only) -- RUN THIS ON THE AGX ORIN (real GPU).
#
# Separates World B (real but benign shared-memory race, oracle-blind) from
# World C (removed barrier was redundant -> equivalent mutant) for the 9
# sync_removal mutants, which the campaign's diff oracle (tau=1e-05) cannot tell
# apart: gap=0, SURVIVED on both SIL and HIL.
#
# compute-sanitizer --tool racecheck flags shared-memory RAW/WAR/WAW hazards
# between two __syncthreads(), whether or not the race corrupts the output.
#
# Method (mirrors the campaign, one variable changed = -lineinfo instead of -G):
#   - control FIRST (pristine kernel): if it already reports hazards, srad has a
#     latent race and per-mutant attribution is confounded -> flagged in bold.
#   - one target at a time, sequentially (shared iGPU, P3: never parallel).
#   - reduced niter=1: a hazard shows on the first kernel launch (no propagation
#     needed) and racecheck slows execution 10-100x.
#   - after each run: check exit code + that srad_hil.out is non-empty. A missing
#     output means the kernel did NOT run -- never mistake "no hazard" for that.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"
MUT="$HERE/mutants"
LOGDIR="$HERE/logs"
ARCH="${SM_ARCH:-sm_87}"          # AGX Orin = Ampere, cc 8.7
ARGS=(512 512 0 127 0 127 0.5 1)  # niter=1 (hazard detection needs no propagation)
SANITIZER="${SANITIZER:-compute-sanitizer}"

export PATH="/usr/local/cuda/bin:$PATH"
mkdir -p "$LOGDIR"

echo "=== tool / toolchain ==="
command -v "$SANITIZER" || { echo "FATAL: $SANITIZER not found (need CUDA 13.x)"; exit 1; }
"$SANITIZER" --version | head -3
nvcc --version | tail -2
echo "arch=$ARCH  args=${ARGS[*]}  flags=-lineinfo"
echo

# Build one target ($1=name, $2=kernel-to-overlay or empty for pristine control),
# run once bare (proves the kernel executes), then under racecheck.
# Appends one CSV row to $LOGDIR/summary.csv: name,build,exec,hazards,summary_line
run_one() {
  local name="$1" kernel="${2:-}"
  local wd; wd="$(mktemp -d)"
  cp "$SRC/srad.cu" "$SRC/srad.h" "$SRC/srad_host.h" "$wd/"
  if [ -n "$kernel" ]; then cp "$kernel" "$wd/srad_kernel.cu"; else cp "$SRC/srad_kernel.cu" "$wd/"; fi

  # -lineinfo: keep optimizations + scheduling (a full -G run reorders threads and
  # can mask the very race we hunt) while getting source-line attribution.
  if ! ( cd "$wd" && nvcc -lineinfo -arch="$ARCH" srad.cu -o srad ) >"$LOGDIR/$name.build.log" 2>&1; then
    echo "$name  BUILD-FAIL"; echo "$name,FAIL,,,build-fail" >>"$LOGDIR/summary.csv"; rm -rf "$wd"; return
  fi

  # Bare run: confirms the kernel actually executes (non-empty output matrix).
  ( cd "$wd" && ./srad "${ARGS[@]}" ) >"$LOGDIR/$name.run.log" 2>&1
  local rc=$?
  local out; out=$(awk '/Printing Output:/{f=1;next} /Computation Done/{f=0} f' "$LOGDIR/$name.run.log" | wc -l)
  local exec="ok"
  if [ "$rc" -ne 0 ] || [ "$out" -eq 0 ]; then exec="KERNEL-DID-NOT-RUN(rc=$rc,rows=$out)"; fi

  # racecheck run.
  local rclog="$LOGDIR/$name.rc.log"
  ( cd "$wd" && "$SANITIZER" --tool racecheck --log-file "$rclog" ./srad "${ARGS[@]}" ) >"$LOGDIR/$name.rc.stdout" 2>&1
  local sumline hazards
  sumline=$(grep -i "RACECHECK SUMMARY" "$rclog" 2>/dev/null | tail -1)
  hazards=$(printf '%s' "$sumline" | grep -oE '[0-9]+ hazard' | grep -oE '[0-9]+' | head -1)
  [ -z "$hazards" ] && hazards="?"

  echo "$name  exec=$exec  hazards=$hazards  | $sumline"
  echo "$name,ok,$exec,$hazards,\"$sumline\"" >>"$LOGDIR/summary.csv"
  rm -rf "$wd"
}

echo "name,build,exec,hazards,summary_line" > "$LOGDIR/summary.csv"

echo "=== CONTROL (pristine kernel) -- must be hazard-free, else latent race ==="
run_one baseline_control ""
echo

echo "=== 9 sync_removal mutants (one at a time) ==="
for n in 001 002 003 004 005 006 007 008 009; do
  run_one "sync_removal_$n" "$MUT/sync_removal_$n/srad_kernel.cu"
done

echo
echo "=== done. raw logs in $LOGDIR/ ; machine summary: $LOGDIR/summary.csv ==="
