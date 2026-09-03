// Vertical pass of a separable convolution, border pixels clamped (edge padding).
__global__ void conv1d_vertical(const float *src, float *dst, const float *kern,
                                int radius, int width, int height)
{
    __shared__ float tile[TILE_H + 2 * MAX_RADIUS][TILE_W];

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;
    int xc = min(x, width - 1);

    for (int i = threadIdx.y; i < TILE_H + 2 * radius; i += TILE_H) {
        int gy = blockIdx.y * TILE_H + i - radius;
        gy = min(max(gy, 0), height - 1);
        tile[i][threadIdx.x] = src[gy * width + xc];
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < 2 * radius + 1; ++k) {
        acc += kern[k] * tile[threadIdx.y + k][threadIdx.x];
    }
    dst[y * width + x] = acc;
}
