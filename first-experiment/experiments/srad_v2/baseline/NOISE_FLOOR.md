# SIL/HIL noise floor

Measured on unmutated code, same input and seed on both platforms, to set the
mutation campaign's tolerance.

- SIL: `srad_sil_baseline.out` (CuPBoP on x86, CUDA 11.7)
- HIL: `srad_hil.out` (Orin, nvcc 13.2, `-arch=sm_87`)
- Input: `512 512 0 127 0 127 0.5 2`, `srand(7)`
- Compared element by element with `compare_outputs.sh`

| Metric | Value |
|---|---|
| Cells compared | 262144 (512x512) |
| Max absolute difference | 1e-05 |
| RMSE | 1.34e-06 |

The max difference equals the resolution of the output format (`printf("%.5f")`),
so the two platforms agree to the last printed digit and the floor is set by the
format, not by the translation.

**Tolerance: tau = 1e-05.** A mutant is killed when `max abs diff > tau`. Below
that it survives. A fault whose effect stays under 1e-05 is invisible on either
platform: that is a limit of the oracle, not of the SIL stage.
