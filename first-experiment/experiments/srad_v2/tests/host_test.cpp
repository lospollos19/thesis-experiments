// Job A: native unit tests (g++ + GoogleTest, no GPU, no CuPBoP) for the pure
// host helpers in srad_host.h. This is the "runs anywhere" tier of the pipeline;
// the GPU/kernel tier goes through CuPBoP separately (Job B).
#include "srad_host.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

// ---- random_matrix ---------------------------------------------------------

// srand(7) is applied inside, so output must not depend on prior RNG state.
TEST(RandomMatrix, DeterministicRegardlessOfPriorState) {
    const int rows = 8, cols = 8;
    std::vector<float> a(rows * cols), b(rows * cols);

    random_matrix(a.data(), rows, cols);
    // Perturb global RNG state, then refill: must be byte-identical.
    srand(12345);
    (void)rand();
    random_matrix(b.data(), rows, cols);

    EXPECT_EQ(a, b);
}

TEST(RandomMatrix, ValuesInUnitRange) {
    const int rows = 16, cols = 16;
    std::vector<float> m(rows * cols);
    random_matrix(m.data(), rows, cols);
    for (float v : m) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
}

// ---- compute_q0sqr ---------------------------------------------------------

// Uniform ROI => variance 0 => q0sqr 0, mean = the constant.
TEST(ComputeQ0sqr, UniformRoiHasZeroCoefficient) {
    const int cols = 4;
    std::vector<float> J(4 * cols, 2.5f);   // 4x4, all 2.5
    float mean = -1, var = -1;
    // ROI = whole matrix: rows 0..3, cols 0..3, size_R = 16
    float q = compute_q0sqr(J.data(), cols, 0, 3, 0, 3, 16, &mean, &var);
    EXPECT_FLOAT_EQ(mean, 2.5f);
    EXPECT_NEAR(var, 0.0f, 1e-5f);
    EXPECT_NEAR(q, 0.0f, 1e-5f);
}

// Known values, oracle computed in double independently of the implementation.
TEST(ComputeQ0sqr, MatchesHandComputedStats) {
    const int cols = 2;
    // 2x2 ROI = {1,2,3,4}
    std::vector<float> J = {1.f, 2.f, 3.f, 4.f};
    const int size_R = 4;
    float mean = 0, var = 0;
    float q = compute_q0sqr(J.data(), cols, 0, 1, 0, 1, size_R, &mean, &var);

    double sum = 1 + 2 + 3 + 4;                 // 10
    double sum2 = 1 + 4 + 9 + 16;               // 30
    double m = sum / size_R;                    // 2.5
    double v = (sum2 / size_R) - m * m;         // 7.5 - 6.25 = 1.25
    double expected_q = v / (m * m);            // 1.25 / 6.25 = 0.2

    EXPECT_NEAR(mean, m, 1e-5);
    EXPECT_NEAR(var, v, 1e-5);
    EXPECT_NEAR(q, expected_q, 1e-5);
}

// ROI restricted to a sub-window: only the selected cells count.
TEST(ComputeQ0sqr, RespectsRoiWindow) {
    const int cols = 3;
    // 3x3, distinct values
    std::vector<float> J = {1,2,3, 4,5,6, 7,8,9};
    // ROI = center column-ish: rows 1..1, cols 1..1 -> single cell {5}
    float mean = 0, var = 0;
    float q = compute_q0sqr(J.data(), cols, 1, 1, 1, 1, 1, &mean, &var);
    EXPECT_FLOAT_EQ(mean, 5.0f);
    EXPECT_NEAR(var, 0.0f, 1e-5f);
    EXPECT_NEAR(q, 0.0f, 1e-5f);
}
