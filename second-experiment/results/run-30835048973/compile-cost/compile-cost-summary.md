# Compilation cost — one translation unit against five

`compile_s` = `SourceModule` construction + `get_function` for all five kernels.
CUDA context creation is excluded (reported separately below): it is paid either way
and is not what the split changes.

## compile_s

| Config | Cache | n | Median (s) | Min | Max |
|---|---|---|---|---|---|
| A — 1 unit | cold | 5 | 0.623 | 0.619 | 0.623 |
| A — 1 unit | hot | 5 | 0.006 | 0.006 | 0.006 |
| B — 5 units | cold | 5 | 2.649 | 2.641 | 2.653 |
| B — 5 units | hot | 5 | 0.007 | 0.007 | 0.007 |

## Ratio, the number the decision turns on

- **cold**: B / A = **4.26×** (2.649 s against 0.623 s, medians).
- **hot**: B / A = **1.13×** (0.007 s against 0.006 s, medians).

## Per-kernel breakdown, config B

Five costs each close to the single-unit cost means a fixed per-`nvcc` overhead and a
split that costs about 5×. Costs well below it mean the price tracks compiled volume.

| Kernel | Cache | n | Median build (s) | Min | Max | Source bytes |
|---|---|---|---|---|---|---|
| `grayscale_bgr_u8` | cold | 5 | 0.518 | 0.515 | 0.519 | 551 |
| `grayscale_bgr_u8` | hot | 5 | 0.006 | 0.006 | 0.006 | 551 |
| `conv1d_horizontal` | cold | 5 | 0.555 | 0.554 | 0.559 | 1032 |
| `conv1d_horizontal` | hot | 5 | 0.000 | 0.000 | 0.000 | 1032 |
| `conv1d_vertical` | cold | 5 | 0.554 | 0.552 | 0.556 | 951 |
| `conv1d_vertical` | hot | 5 | 0.000 | 0.000 | 0.000 | 951 |
| `absdiff_f32` | cold | 5 | 0.511 | 0.510 | 0.514 | 401 |
| `absdiff_f32` | hot | 5 | 0.000 | 0.000 | 0.000 | 401 |
| `threshold_u8` | cold | 5 | 0.510 | 0.506 | 0.512 | 413 |
| `threshold_u8` | hot | 5 | 0.000 | 0.000 | 0.000 | 413 |

## CUDA context creation (excluded from the figures above)

| Config | Cache | n | Median (s) | Min | Max |
|---|---|---|---|---|---|
| A — 1 unit | cold | 5 | 0.063 | 0.061 | 0.064 |
| A — 1 unit | hot | 5 | 0.061 | 0.060 | 0.061 |
| B — 5 units | cold | 5 | 0.063 | 0.061 | 0.064 |
| B — 5 units | hot | 5 | 0.060 | 0.060 | 0.061 |

## Environment

- nvcc: `Build cuda_13.2.r13.2/compiler.37668154_0` at `/usr/local/cuda/bin/nvcc`
- PyCUDA: `2026.1`, Python `3.12.3`
- cache directory used: `/home/vub/Documents/moussa/gpu-motion-runner/_work/_temp/rq2-compile-cache`
- PyCUDA default cache directory (not used, reported for context): `/home/vub/.cache/pycuda/compiler-cache-v1`
- single-unit source: 3294 bytes

## Runs discarded

- config A, hot, rep 0: warmup run, populates the cache for the hot block
- config B, hot, rep 0: warmup run, populates the cache for the hot block

