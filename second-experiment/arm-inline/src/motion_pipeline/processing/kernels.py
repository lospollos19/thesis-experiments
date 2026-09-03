"""CUDA-C sources for the frame primitives, kept apart from the orchestration code.

Every kernel used by :mod:`motion_pipeline.processing.gpu_processor` lives here as an
inline string and is reached through the single indirection :func:`get_kernel_source`.
That indirection is the intended switch point: an alternative storage mode (external
``.cu`` files) can be plugged in behind it without touching any caller.

Layout conventions shared by all kernels:

* images are row-major and contiguous, indexed as ``y * width + x``;
* colour input is 8-bit BGR interleaved (3 bytes per pixel), matching OpenCV;
* intermediate results are ``float32``, binary masks are ``uint8`` (0 or 255);
* out-of-image reads are clamped to the border (``edge`` padding), which is what the
  NumPy reference path does via ``np.pad(..., mode="edge")``.
"""

from __future__ import annotations

# Tile geometry shared by the two separable convolution kernels. MAX_RADIUS caps the
# halo the shared-memory tile can hold, i.e. the largest supported blur kernel is
# 2 * MAX_RADIUS + 1 taps.
TILE_W = 32
TILE_H = 8
MAX_RADIUS = 16
MAX_KERNEL_SIZE = 2 * MAX_RADIUS + 1

_COMMON_DEFINES = f"""
#define TILE_W {TILE_W}
#define TILE_H {TILE_H}
#define MAX_RADIUS {MAX_RADIUS}
"""

_GRAYSCALE_BGR_U8 = """
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
"""

# Note on __syncthreads(): threads outside the image must still reach the barrier, so
# the guards clamp coordinates for the load phase and only skip the final store.
_CONV1D_HORIZONTAL = """
// Horizontal pass of a separable convolution, border pixels clamped (edge padding).
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
"""

_CONV1D_VERTICAL = """
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
"""

_ABSDIFF_F32 = """
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
"""

_THRESHOLD_U8 = """
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
"""

# name -> CUDA-C source. The convolution passes share the tile-geometry defines.
_SOURCES: dict[str, str] = {
    "grayscale_bgr_u8": _GRAYSCALE_BGR_U8,
    "conv1d_horizontal": _COMMON_DEFINES + _CONV1D_HORIZONTAL,
    "conv1d_vertical": _COMMON_DEFINES + _CONV1D_VERTICAL,
    "absdiff_f32": _ABSDIFF_F32,
    "threshold_u8": _THRESHOLD_U8,
}

KERNEL_NAMES: tuple[str, ...] = tuple(_SOURCES)


def get_kernel_source(name: str) -> str:
    """Return the CUDA-C source of kernel ``name``.

    Single point of access to the kernel sources: callers never read the storage
    directly, so the storage mode can be changed here alone.
    """
    try:
        return _SOURCES[name]
    except KeyError:
        known = ", ".join(KERNEL_NAMES)
        raise KeyError(f"unknown kernel {name!r}; known kernels: {known}") from None


def get_module_source() -> str:
    """Concatenate every kernel into a single translation unit.

    PyCUDA compiles one ``SourceModule`` per string, so the processor builds all
    kernels at once rather than paying nvcc's start-up cost five times.
    """
    return _COMMON_DEFINES + "\n".join(
        (_GRAYSCALE_BGR_U8, _CONV1D_HORIZONTAL, _CONV1D_VERTICAL, _ABSDIFF_F32, _THRESHOLD_U8)
    )
