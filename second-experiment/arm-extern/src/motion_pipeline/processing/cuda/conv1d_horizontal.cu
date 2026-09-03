// Horizontal pass of a separable convolution, border pixels clamped (edge padding).
//
// Note on __syncthreads(): threads outside the image must still reach the barrier,
// so the guards clamp coordinates for the load phase and only skip the final store.
__global__ void conv1d_horizontal(const float *src, float *dst, const float *kern,
                                  int radius, int width, int height)
{
    __shared__ float tile[TILE_H][TILE_W + 2 * MAX_RADIUS];

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    int yc = min(y, height - 1);

    // Cooperatively load TILE_W pixels plus a `radius` halo on each side.
    for (int i = threadIdx.x; i < TILE_W + 2 * radius; i += TILE_W) {
        int gx = blockIdx.x * TILE_W + i - radius;
        gx = min(max(gx, 0), width - 1);
        tile[threadIdx.y][i] = src[yc * width + gx];
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < 2 * radius + 1; ++k) {
        acc += kern[k] * tile[threadIdx.y][threadIdx.x + k];
    }
    dst[y * width + x] = acc;
}
