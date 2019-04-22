#include "utils.h"
#include <cuda_runtime.h>
#include <string>
#include <cstdio>
#include <cassert>
#include <cmath>

/*
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) 
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}
*/

// Copy data from host to device
template <typename T> 
void copyMemHost2Device(T * & deviceMem, T * & hostMem, size_t numUnits)
{
    checkCuda(cudaMemcpy(deviceMem, hostMem, sizeof(T) * numUnits, cudaMemcpyHostToDevice));
} 

template <typename T> 
void copyMemDevice2Host(T * & hostMem, T * & deviceMem, size_t numUnits)
{
    checkCuda(cudaMemcpy(hostMem, deviceMem, sizeof(T) * numUnits, cudaMemcpyDeviceToHost));
} 

// Generate kernel/filter for Gaussian blurring
float * generateGaussianBlurKernel(const int kernelWidth = 9, const float kernelSigma = 2.0)
{
    // Kernel width has to be odd number
    assert(kernelWidth % 2 == 1);
    float * kernel = new float[kernelWidth * kernelWidth];
    float kernelSum = 0.0;
    for (int r = -kernelWidth/2; r <= kernelWidth/2; r ++) 
    {
        for (int c = -kernelWidth/2; c <= kernelWidth/2; c ++) 
        {
            float filterValue = expf( -(float)(c * c + r * r) / (2.f * kernelSigma * kernelSigma));
            kernel[(r + kernelWidth/2) * kernelWidth + c + kernelWidth/2] = filterValue;
            kernelSum += filterValue;
        }
    }

    float normalizationFactor = 1.0 / kernelSum;

    for (int r = -kernelWidth/2; r <= kernelWidth/2; r ++) 
    {
        for (int c = -kernelWidth/2; c <= kernelWidth/2; c ++) 
        {
            kernel[(r + kernelWidth/2) * kernelWidth + c + kernelWidth/2] *= normalizationFactor;
        }
    }
    return kernel;
}

