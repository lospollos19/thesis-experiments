#!/usr/bin/env bash
# Element-wise comparison of two srad matrix dumps -> noise-floor metrics.
# Both files must be the same dimensions (same args/seed).
#
# Usage: bash compare_outputs.sh <fileA> <fileB> [threshold]
#   threshold optional; with 0 (default) it just reports max abs diff + RMSE,
#   which IS the noise floor when A=SIL baseline and B=HIL reference (correct code).
#   Re-run with a chosen tolerance to see how many cells a mutant would move past it.
set -euo pipefail

A="${1:?fileA required}"
B="${2:?fileB required}"
THR="${3:-0}"

awk -v thr="$THR" '
FNR==NR { for(i=1;i<=NF;i++) a[FNR,i]=$i; acols[FNR]=NF; arows=FNR; next }
{
  if (NF != acols[FNR]) { printf "DIM MISMATCH row %d: %d vs %d cols\n", FNR, acols[FNR], NF > "/dev/stderr"; err=1 }
  for (i=1;i<=NF;i++) {
    d = $i - a[FNR,i]; if (d < 0) d = -d;
    n++; sumsq += d*d;
    if (d > maxd) { maxd = d; mr = FNR; mc = i }
    if (thr > 0 && d > thr) { over++; if (!ff) { ff=1; fr=FNR; fc=i; fd=d } }
  }
  brows = FNR
}
END {
  if (err)   { print "ERROR: dimension mismatch (files not comparable)" > "/dev/stderr"; exit 2 }
  if (arows != brows) { printf "ERROR: row count %d vs %d\n", arows, brows > "/dev/stderr"; exit 2 }
  if (n == 0) { print "no data" > "/dev/stderr"; exit 2 }
  printf "cells compared : %d  (%d x %d)\n", n, arows, acols[1]
  printf "max abs diff   : %.8g   (row %d col %d)\n", maxd, mr, mc
  printf "RMSE           : %.8g\n", sqrt(sumsq/n)
  if (thr > 0) {
    printf "threshold      : %.8g\n", thr
    printf "cells > thr    : %d  (%.4f%%)\n", over+0, (over+0)*100.0/n
    if (ff) printf "first > thr    : row %d col %d diff %.8g\n", fr, fc, fd
    else    print  "first > thr    : none (all within tolerance)"
  }
}' "$A" "$B"
