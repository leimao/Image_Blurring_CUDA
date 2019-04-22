#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cuda_runtime.h>

//cudaError_t checkCuda(cudaError_t result);

#define checkCuda(val) check( (val), #val, __FILE__, __LINE__ )
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) 
{
    if (err != cudaSuccess) 
    {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

template <typename T> 
void copyMemHost2Device(T * & deviceMem, T * & hostMem, size_t numUnits);

template <typename T> 
void copyMemDevice2Host(T * & hostMem, T * & deviceMem, size_t numUnits);

float * generateGaussianBlurKernel(const int kernelWidth, const float kernelSigma);

#endif