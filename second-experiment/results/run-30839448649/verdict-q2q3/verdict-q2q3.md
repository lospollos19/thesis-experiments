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
- absent from the map: **3**
- present but with no kernel: **0**
- verdict: **INCOMPLETE**
  - absent: `tests/test_kernels.py::test_grayscale_passthrough_2d_matches_numpy`
  - absent: `tests/test_kernels.py::test_backend_is_pycuda_on_device`
  - absent: `tests/test_kernels.py::test_blur_rejects_kernel_larger_than_halo`

### Suite duration after the split into five translation units

113 passed, 52 deselected in 227.58s
113 passed, 52 deselected in 231.01s
113 passed, 52 deselected in 229.34s

Reference before the split: ~234 s.
