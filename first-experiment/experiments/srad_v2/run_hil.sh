#!/usr/bin/env bash
# HIL reference run (native nvcc on the AGX Orin) -- mirror of translate_srad.sh
# so the output is directly comparable to the SIL baseline. RUN THIS ON THE ORIN.
#
# Identical to SIL except the compiler: same source (post-refactor srad.cu with
# #define OUTPUT), same args, same srand(7), SAME extraction block. Only nvcc/GPU
# differs -> the diff vs the baseline is the SIL/HIL noise floor, nothing else.
#
# Usage: bash run_hil.sh [source.cu] [program args...]
#   defaults to the calibration args used by Job B (srad-sil.yaml).
set -euo pipefail

SRC="${1:-srad.cu}"; shift || true
if [ "$#" -eq 0 ]; then
  set -- 512 512 0 127 0 127 0.5 2   # MUST match srad-sil.yaml
fi
BASE="$(basename "$SRC" .cu)"

# AGX Orin = Ampere, compute capability 8.7. Override with SM_ARCH if needed.
ARCH="${SM_ARCH:-sm_87}"

echo "=== nvcc compile (arch=$ARCH) ==="
nvcc -arch="$ARCH" "$SRC" -o "$BASE"

echo "=== running on GPU (HIL), args: $* ==="
./"$BASE" "$@" | tee srad_hil.full.log

# SAME extraction as translate_srad.sh -> comparable dumps.
awk '/Printing Output:/{f=1;next} /Computation Done/{f=0} f' srad_hil.full.log > srad_hil.out
test -s srad_hil.out || { echo "ERROR: srad_hil.out empty -- is #define OUTPUT enabled?"; exit 1; }

echo "=== srad_hil.out ready ($(wc -l < srad_hil.out) rows) ==="
echo "Next: copy srad_hil.out back and run compare_outputs.sh vs the SIL baseline."
