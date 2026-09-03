#ifndef SRAD_HOST_H
#define SRAD_HOST_H

// Pure host-side (no CUDA) helpers extracted from srad.cu so they can be unit
// tested natively (Job A: g++ + GoogleTest, no GPU, no CuPBoP). srad.cu includes
// this header and calls these functions -> single source of truth, the tests
// exercise the real code path, not a copy.
//
// HEADER-ONLY (inline) on purpose: srad_v2 must stay a single translation unit
// for the CuPBoP flow (srad_v1 was rejected for being multi-TU). A separate .cpp
// would reintroduce multi-TU linking and break Job B. The tests just #include it.

#include <stdlib.h>

// Fill I (rows*cols) with a reproducible pseudo-random field in [0,1].
// Uses srand(7) internally, so the output is deterministic regardless of prior
// RNG state -- this fixed seed is what makes srad_sil.out reproducible.
inline void random_matrix(float *I, int rows, int cols) {
    srand(7);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            I[i * cols + j] = rand() / (float)RAND_MAX;
        }
    }
}

// Speckle-ROI statistics over J[r1..r2][c1..c2] (row-major, stride = cols).
// Returns q0sqr; optionally writes meanROI / varROI. Same operations, same order,
// same float type as the original inline block in runTest() -> bit-identical
// result (keeps srad_sil.out deterministic).
inline float compute_q0sqr(const float *J, int cols,
                           int r1, int r2, int c1, int c2, int size_R,
                           float *meanROI_out = 0, float *varROI_out = 0) {
    float sum = 0, sum2 = 0, tmp;
    for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
            tmp   = J[i * cols + j];
            sum  += tmp;
            sum2 += tmp * tmp;
        }
    }
    float meanROI = sum / size_R;
    float varROI  = (sum2 / size_R) - meanROI * meanROI;
    if (meanROI_out) *meanROI_out = meanROI;
    if (varROI_out)  *varROI_out  = varROI;
    return varROI / (meanROI * meanROI);
}

#endif // SRAD_HOST_H
