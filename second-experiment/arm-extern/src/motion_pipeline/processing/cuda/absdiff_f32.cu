// |a - b| on float32 images, one thread per pixel.
__global__ void absdiff_f32(const float *a, const float *b, float *dst,
                            int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int i = y * width + x;
    dst[i] = fabsf(a[i] - b[i]);
}
