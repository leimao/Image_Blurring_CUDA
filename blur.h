#ifndef BLUR_H
#define BLUR_H

#include <cuda_runtime.h>

void rgbaBlurDevice(const uchar4 * const d_rgbaImage, uchar4 * const d_blurRGBAImage, size_t numRows, size_t numCols, const float * const filter, size_t filterHeight, size_t filterWidth);

#endif