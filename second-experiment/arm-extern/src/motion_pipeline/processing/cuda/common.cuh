// Tile geometry shared by the two separable convolution kernels.
//
// This header is the single source of truth for these three values: the Python
// loader parses them out of this file rather than declaring its own copies, so a
// change here reaches both the compiled kernels and the launch configuration.
//
// MAX_RADIUS caps the halo the shared-memory tile can hold, i.e. the largest
// supported blur kernel is 2 * MAX_RADIUS + 1 taps.

#define TILE_W 32
#define TILE_H 8
#define MAX_RADIUS 16
