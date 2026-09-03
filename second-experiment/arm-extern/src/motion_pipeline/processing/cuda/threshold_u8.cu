// Binary mask: 255 where src > value, else 0.
__global__ void threshold_u8(const float *src, unsigned char *dst, float value,
                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int i = y * width + x;
    dst[i] = (src[i] > value) ? 255 : 0;
}
