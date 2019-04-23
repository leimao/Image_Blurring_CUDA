#include "timer.h"
#include <cuda_runtime.h>

GpuTimer::GpuTimer()
{
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

GpuTimer::~GpuTimer()
{
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void GpuTimer::start()
{
    cudaEventRecord(startEvent, 0);
}

void GpuTimer::stop()
{
    cudaEventRecord(stopEvent, 0);
}

float GpuTimer::elapsed()
{
    float elapsed;
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
    return elapsed;
}

