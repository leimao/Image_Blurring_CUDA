#ifndef GPU_TIMER_H
#define GPU_TIMER_H

#include <cuda_runtime.h>

class GpuTimer
{
    public:
        GpuTimer();
        ~GpuTimer();
        void start();
        void stop();
        float elapsed();

    private:
        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;
};

#endif