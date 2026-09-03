## Q2 — test to kernel map

- tests in the map: **110**
- edges: **319**
- tests depending on a strict subset: **76**
- verdict: **NON-TRIVIAL**

| Kernels per test | Tests |
|---|---|
| 0 | 0 |
| 1 | 33 |
| 2 | 28 |
| 3 | 0 |
| 4 | 15 |
| 5 | 34 |

| Kernel | Tests depending on it |
|---|---|
| `absdiff_f32` | 56 |
| `conv1d_horizontal` | 77 |
| `conv1d_vertical` | 77 |
| `grayscale_bgr_u8` | 41 |
| `threshold_u8` | 68 |

## Q3 — completeness

- gpu tests collected: **113**
- declared `no_kernel_launch`: **3**
- absent and not declared: **0**
- present but with no kernel: **0**
- verdict: **COMPLETE**

The declared tests are absent by design: they assert on the backend, exit
before a launch, or raise on a guard. They are not dependency-free — they
carry compilation and constant dependencies that a launch-site tracer
cannot observe, and a kernel-aware selector needs a second edge type for
them. Absence from this map is not a licence to deselect them.

### Suite duration after the split into five translation units

113 passed, 52 deselected in 228.00s
113 passed, 52 deselected in 229.74s
113 passed, 52 deselected in 230.51s

Reference before the split: ~234 s.
