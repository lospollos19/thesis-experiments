// BGR uint8 -> float32 luminance, one thread per output pixel.
__global__ void grayscale_bgr_u8(const unsigned char *src, float *dst,
                                 int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int px = (y * width + x) * 3;
    dst[y * width + x] = 0.114f * (float)src[px + 0]
                       + 0.587f * (float)src[px + 1]
                       + 0.299f * (float)src[px + 2];
}
