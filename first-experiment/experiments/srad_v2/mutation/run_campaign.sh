#!/usr/bin/env bash
# Mutation campaign runner (RQ1). One environment per invocation, loops over all
# mutants -> a single CI job builds+runs the whole set (cheap on Actions minutes).
#
#   run_campaign.sh sil    # inside the CuPBoP SIL image (Job B toolchain)
#   run_campaign.sh hil    # on the Orin self-hosted runner (nvcc)
#
# Per mutant: assemble a fresh srad_v2 tree, overlay the ONE mutated file, build +
# run on the calibration input, extract the matrix, verdict against the committed
# SIL baseline at tolerance TAU. Emits results_<mode>.json for analyze_campaign.py.
#
# NOTE: the determinism guard is intentionally NOT applied here -- mutants are
# expected to diverge; divergence is the signal, not an error.
set -uo pipefail   # deliberately no -e: per-mutant failures are data, not aborts

MODE="${1:?usage: run_campaign.sh <sil|hil>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRAD="$(dirname "$HERE")"
MUT="$HERE/mutants"
BASELINE=""   # set at runtime to the pristine SAME-PLATFORM baseline (see below)
TAU="${TAU:-1e-05}"
NITER="${NITER:-2}"   # bump (e.g. 100) to let race/propagation faults manifest
ARGS=(512 512 0 127 0 127 0.5 "$NITER")
PER_MUTANT_TIMEOUT="${PER_MUTANT_TIMEOUT:-180}"
RESULTS="$HERE/results_${MODE}.jsonl"   # JSONL: one verdict per line, resumable

BASE_FILES=(srad.cu srad_kernel.cu srad.h srad_host.h translate_srad.sh run_hil.sh)

command -v timeout >/dev/null || timeout() { shift; "$@"; }   # macOS fallback (no-op limit)

# (Re)generate mutants so the run matches the committed generator exactly.
python3 "$HERE/generate_mutants.py" >/dev/null

# verdict vs the pristine same-platform baseline: KILLED_OUTPUT|SURVIVED|DIM_ERR + max diff.
verdict() {  # $1 = candidate file
python3 - "$BASELINE" "$1" "$TAU" <<'PY'
import sys
base, cand, tau = sys.argv[1], sys.argv[2], float(sys.argv[3])
try:
    A=[l.split() for l in open(base).read().splitlines()]
    B=[l.split() for l in open(cand).read().splitlines()]
except Exception:
    print("DIM_ERR 0"); sys.exit()
if len(A)!=len(B) or any(len(a)!=len(b) for a,b in zip(A,B)):
    print("DIM_ERR 0"); sys.exit()
mx=0.0
for a,b in zip(A,B):
    for x,y in zip(a,b):
        d=abs(float(x)-float(y))
        if d>mx: mx=d
print(("KILLED_OUTPUT" if mx>tau else "SURVIVED"), mx)
PY
}

# Build + run srad in $1 (mode-specific). Optional $2 = a mutated file to overlay.
# Sets globals RC and OUTFILE.
build_run() {
  local wd="$1" overlay="${2:-}"
  mkdir -p "$wd"
  for f in "${BASE_FILES[@]}"; do cp "$SRAD/$f" "$wd/" 2>/dev/null; done
  [ -n "$overlay" ] && cp "$overlay" "$wd/$(basename "$overlay")"
  if [ "$MODE" = sil ]; then
    ( cd "$wd" && timeout "$PER_MUTANT_TIMEOUT" bash translate_srad.sh srad.cu "${ARGS[@]}" ) >"$wd/log" 2>&1
    RC=$?; OUTFILE="$wd/srad_sil.out"
  else
    ( cd "$wd" && export PATH="/usr/local/cuda/bin:$PATH" && timeout "$PER_MUTANT_TIMEOUT" bash run_hil.sh srad.cu "${ARGS[@]}" ) >"$wd/log" 2>&1
    RC=$?; OUTFILE="$wd/srad_hil.out"
  fi
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# --- pristine SAME-PLATFORM baseline (unmutated srad on THIS environment) --------
# Each side scores its mutants against ITS OWN correct-code output, which cancels
# the SIL/HIL platform noise floor. Scoring HIL mutants against the SIL baseline
# injects that 1e-05 floor and false-kills equivalent mutants (the bug this fixes).
echo "Building pristine $MODE baseline (unmutated srad)..."
build_run "$work/_baseline"
if [ $RC -ne 0 ] || [ ! -s "$OUTFILE" ]; then
  echo "FATAL: pristine $MODE build failed -- cannot run campaign"; tail -20 "$work/_baseline/log"; exit 2
fi
BASELINE="$work/${MODE}_baseline.out"; cp "$OUTFILE" "$BASELINE"
echo "baseline ready: $(wc -l < "$BASELINE") rows"

# --- GPU / toolchain health gate ------------------------------------------------
# Correct code must roughly agree across platforms (noise floor ~1e-05). A GROSS
# divergence from the committed reference means the kernels are not really running
# (e.g. a wedged Orin returning undiffused J=exp(I), cudaErrorDevicesUnavailable
# swallowed by srad) -> abort now instead of emitting a whole campaign of false
# SURVIVED verdicts. Loose threshold (0.05 >> 1e-05): trips only on gross failure,
# never on a real SIL/HIL difference. This is a sanity gate, NOT the verdict oracle.
REF="$SRAD/baseline/srad_sil_baseline.out"
if [ "$NITER" = 2 ] && [ -f "$REF" ]; then   # committed reference is niter=2 only
  HD=$(python3 - "$REF" "$BASELINE" <<'PY'
import sys
A=[l.split() for l in open(sys.argv[1]).read().splitlines()]
B=[l.split() for l in open(sys.argv[2]).read().splitlines()]
mx=0.0
if len(A)==len(B) and all(len(a)==len(b) for a,b in zip(A,B)):
    for a,b in zip(A,B):
        for x,y in zip(a,b):
            d=abs(float(x)-float(y))
            if d>mx: mx=d
else:
    mx=float("inf")
print(mx)
PY
)
  echo "health gate: pristine $MODE vs committed reference, max diff = $HD"
  if ! python3 -c "import sys;sys.exit(0 if float('$HD')<=0.05 else 1)"; then
    echo "FATAL: pristine $MODE output diverges grossly from the reference ($HD > 0.05)."
    echo "       Kernels are likely NOT executing (wedged GPU / broken toolchain)."
    echo "       Fix the environment before running the campaign -- aborting."
    exit 3
  fi
fi

# Mutant order: OOB-prone ADDRESS operators LAST. An out-of-bounds mutation can
# wedge a shared GPU (P6); running the safe operators first guarantees they are all
# measured even if a later address mutant wedges the device.
OOB_OPS="gpu_index_replacement gpu_index_increment gpu_index_decrement alloc_swap alloc_increment alloc_decrement"
mapfile -t IDS < <(OOB="$OOB_OPS" python3 - "$MUT/mutants.json" <<'PY'
import json, os, sys
oob = set(os.environ["OOB"].split())
ms = json.load(open(sys.argv[1]))
safe = [m["id"] for m in ms if m["operator"] not in oob]
last = [m["id"] for m in ms if m["operator"] in oob]
for i in safe + last: print(i)
PY
)

# Resume (RESUME=1, after a reboot): skip mutants already in the JSONL, append the rest.
declare -A DONE=()
if [ "${RESUME:-0}" = 1 ] && [ -f "$RESULTS" ]; then
  while IFS= read -r d; do DONE["$d"]=1; done < <(python3 -c "import json;[print(json.loads(l)['id']) for l in open('$RESULTS') if l.strip()]")
  echo "resume: $RESULTS already has ${#DONE[@]} results -- skipping those"
else
  : > "$RESULTS"   # fresh run
fi

total=0; killed=0; survived=0; N=${#IDS[@]}
for mid in "${IDS[@]}"; do
  total=$((total+1))
  if [ -n "${DONE[$mid]:-}" ]; then printf '[%3d/%3d] %-34s (already done)\n' "$total" "$N" "$mid"; continue; fi
  meta=$(python3 -c "import json;m=[x for x in json.load(open('$MUT/mutants.json')) if x['id']=='$mid'][0];print(m['operator'],m['file'],str(m['race_dependent']).lower())")
  op=$(echo "$meta" | cut -d' ' -f1); mfile=$(echo "$meta" | cut -d' ' -f2); race=$(echo "$meta" | cut -d' ' -f3)

  wd="$work/$mid"
  build_run "$wd" "$MUT/$mid/$mfile"

  status="" ; maxd="0"
  if [ $RC -ne 0 ] || [ ! -s "$OUTFILE" ]; then
    status="KILLED_ERROR"   # build fail / crash / timeout / CUDA error => detected here
  else
    read status maxd < <(verdict "$OUTFILE")
  fi

  # Wedge detection (HIL): a wedged shared GPU makes every subsequent mutant fail
  # with devices-unavailable. Abort now -- results so far are valid; reboot the Orin
  # and re-run with RESUME=1 to continue past this point.
  if [ "$MODE" = hil ] && grep -qiE "busy or unavailable|are unavailable|no CUDA-capable device" "$wd/log" 2>/dev/null; then
    rm -rf "$wd"
    echo "=== GPU WEDGED at $mid (cudaErrorDevicesUnavailable) ==="
    echo "Results so far ($((total-1)) mutants) are valid in $RESULTS."
    echo "Reboot the Orin, then re-run with:  RESUME=1 NITER=$NITER bash run_campaign.sh hil"
    exit 4
  fi

  printf '{"id":"%s","operator":"%s","file":"%s","race_dependent":%s,"mode":"%s","verdict":"%s","max_diff":%s}\n' \
    "$mid" "$op" "$mfile" "$race" "$MODE" "$status" "$maxd" >> "$RESULTS"
  case "$status" in KILLED_*|DIM_ERR) killed=$((killed+1));; SURVIVED) survived=$((survived+1));; esac
  rm -rf "$wd"
  printf '[%3d/%3d] %-34s %-14s (max=%s)\n' "$total" "$N" "$mid" "$status" "$maxd"
done

echo "=== $MODE campaign done: processed $total, $killed killed, $survived survived ==="
echo "results -> $RESULTS"
