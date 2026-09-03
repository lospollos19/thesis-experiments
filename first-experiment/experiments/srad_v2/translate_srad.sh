#!/usr/bin/env bash
# CuPBoP translation chain for a single-translation-unit CUDA program.
# Mirrors the proven vecadd flow; srad_v2 is single-TU (srad.cu #includes srad_kernel.cu).
# Usage: translate_srad.sh <source.cu> [program args...]
set -euo pipefail

SRC="${1:-srad.cu}"; shift || true
BASE="$(basename "$SRC" .cu)"

# Compile CUDA source to bitcode. Final link fails without a GPU (expected) -> || true.
# -save-temps drops the host/kernel .bc we actually need.
clang++ -std=c++11 "$SRC" -I. -I"$CuPBoP_PATH" --cuda-path="$CUDA_PATH" \
  --cuda-gpu-arch=sm_50 -L"$CUDA_PATH/lib64" \
  -lcudart_static -ldl -lrt -pthread -save-temps -v || true

# Host triple varies across clang builds, so match the .bc by pattern (not a hard-coded name).
KERNEL_BC=$(ls "$BASE"-*nvptx64*sm_50.bc)
HOST_BC=$(ls "$BASE"-host-*.bc)
echo "kernel=$KERNEL_BC  host=$HOST_BC"

# CuPBoP IR-to-IR translation (this is where srad's __shared__/__syncthreads stress CuPBoP)
"$CuPBoP_PATH/build/compilation/kernelTranslator" "$KERNEL_BC" kernel.bc
"$CuPBoP_PATH/build/compilation/hostTranslator"   "$HOST_BC"   host.bc

llc --relocation-model=pic --filetype=obj kernel.bc
llc --relocation-model=pic --filetype=obj host.bc

g++ -o "$BASE" -fPIC -no-pie \
  -I"$CuPBoP_PATH/runtime/threadPool/include" \
  -L"$CuPBoP_PATH/build/runtime" -L"$CuPBoP_PATH/build/runtime/threadPool" \
  host.o kernel.o -lc -lCPUruntime -lthreadPool -lpthread

echo "=== translation + link OK, running on CPU (CuPBoP) ==="
# Full run log (stdout mixes status messages + the OUTPUT matrix dump).
./"$BASE" "$@" | tee srad_sil.full.log

# Extract only the result matrix J into srad_sil.out for SIL/HIL comparison.
# srad.cu prints "Printing Output:" then the matrix, then "Computation Done".
# Same extraction must be applied to the HIL run to produce a comparable srad_hil.out.
awk '/Printing Output:/{f=1;next} /Computation Done/{f=0} f' srad_sil.full.log > srad_sil.out
test -s srad_sil.out || { echo "::error::srad_sil.out empty -- is #define OUTPUT enabled?"; exit 1; }
echo "=== srad_v2 ran to completion under CuPBoP, matrix dumped to srad_sil.out ($(wc -l < srad_sil.out) rows) ==="