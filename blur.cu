#include "utils.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

// Separate RGBA images to individual R, G, B channels
__global__
void separateChannelsKernel(const uchar4 * const rgbaImage, int numRows, int numCols, unsigned char * const redChannel, unsigned char * const greenChannel, unsigned char * const blueChannel)
{
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    uchar4 pixel;
    unsigned char R;
    unsigned char G;
    unsigned char B;

    for (int i = idxY; i < numRows; i += strideY)
    {
        for (int j = idxX; j < numCols; j += strideX)
        {
            int idx = i * numCols + j;

            pixel = rgbaImage[idx];
            R = pixel.x;
            G = pixel.y;
            B = pixel.z;

            redChannel[idx] = R;
            greenChannel[idx] = G;
            blueChannel[idx] = B;
        }
    }
}

void separateChannelsDevice(const uchar4 * const rgbaImage, int numRows, int numCols, unsigned char * const redChannel, unsigned char * const greenChannel, unsigned char * const blueChannel)
{
    const int blockSizeX = 16;
    const int blockSizeY = 16;
    const int gridSizeX = (numRows + blockSizeX - 1) / blockSizeX;
    const int gridSizeY = (numCols + blockSizeY - 1) / blockSizeY;

    const dim3 blockSize(blockSizeX, blockSizeY, 1);
    const dim3 gridSize(gridSizeX, gridSizeY, 1);

    separateChannelsKernel<<<gridSize, blockSize>>>(rgbaImage, numRows, numCols, redChannel, greenChannel, blueChannel);

    cudaDeviceSynchronize();

    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl; 
    }
    checkCuda(cudaDeviceSynchronize()); 
}


// Combine individual R, G, B channels to RGBA images
__global__
void recombineChannelsKernel(const unsigned char * const redChannel, const unsigned char * const greenChannel, const unsigned char * const blueChannel, uchar4 * const rgbaImage, int numRows, int numCols)
{
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    uchar4 * pixel;
    unsigned char R;
    unsigned char G;
    unsigned char B;

    for (int i = idxY; i < numRows; i += strideY)
    {
        for (int j = idxX; j < numCols; j += strideX)
        {
            int idx = i * numCols + j;

            pixel = rgbaImage + idx;
            R = redChannel[idx];
            G = greenChannel[idx];
            B = blueChannel[idx];

            // There is also a library function make_uchar4
            pixel->x = R;
            pixel->y = G;
            pixel->z = B;
            // Alpha was set to 255 for no transparency
            pixel->w = 255;
        }
    }
}

void recombineChannelsDevice(const unsigned char * const redChannel, const unsigned char * const greenChannel, const unsigned char * const blueChannel, uchar4 * const rgbaImage, int numRows, int numCols)
{
    const int blockSizeX = 16;
    const int blockSizeY = 16;
    const int gridSizeX = (numRows + blockSizeX - 1) / blockSizeX;
    const int gridSizeY = (numCols + blockSizeY - 1) / blockSizeY;

    const dim3 blockSize(blockSizeX, blockSizeY, 1);
    const dim3 gridSize(gridSizeX, gridSizeY, 1);

    recombineChannelsKernel<<<gridSize, blockSize>>>(redChannel, greenChannel, blueChannel, rgbaImage, numRows, numCols);

    cudaDeviceSynchronize();

    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl; 
    }
    checkCuda(cudaDeviceSynchronize()); 
}

__global__
void filterChannelBlurKernel(const unsigned char * const inputChannel, unsigned char * const outputChannel, int numRows, int numCols, const float * const filter, const int filterHeight, const int filterWidth)
{
    // We assume filterHeight and filterWidth are odd numbers.

    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    // We may also save all the element in the receptionField temporarily
    // char * receptionField = new char[filterHeight * filterWidth];

    // Calculate weight sum which might not necessarily be 1
    float weightSum = 0;
    for (int k = 0; k < filterHeight * filterHeight; k ++)
    {
        weightSum += filter[k];
    }
    assert(weightSum != 0);

    for (int i = idxY; i < numRows; i += strideY)
    {
        for (int j = idxX; j < numCols; j += strideX)
        {
            int iCurrent, jCurrent;
            float pixelWeightedSum = 0;
            for (int h = 0; h < filterHeight; h ++)
            {
                iCurrent = i - filterHeight / 2 + h;
                for (int w = 0; w < filterWidth; w ++)
                {
                    jCurrent = j - filterWidth / 2 + w;

                    int k = h * filterHeight + w;
                    // Make sure we do not go outside the channel or image
                    if (iCurrent < 0 || iCurrent >= numRows || jCurrent < 0 || jCurrent >= numCols)
                    {
                        // We do zero paddings
                        pixelWeightedSum += 0 * filter[k];
                    }
                    else
                    {
                        pixelWeightedSum += inputChannel[iCurrent * numCols + jCurrent] * filter[k];
                    }
                }
            }

            float pixelWeightedAvg = pixelWeightedSum / weightSum;
            outputChannel[i * numCols + j] = (char) pixelWeightedAvg;
        }
    }

}

void channelBlurDevice(const unsigned char * const inputChannel, unsigned char * const outputChannel, size_t numRows, size_t numCols, const float * const filter, const size_t filterHeight, const size_t filterWidth)
{
    const int blockSizeX = 16;
    const int blockSizeY = 16;
    const int gridSizeX = (numRows + blockSizeX - 1) / blockSizeX;
    const int gridSizeY = (numCols + blockSizeY - 1) / blockSizeY;

    const dim3 blockSize(blockSizeX, blockSizeY, 1);
    const dim3 gridSize(gridSizeX, gridSizeY, 1);

    filterChannelBlurKernel<<<gridSize, blockSize>>>(inputChannel, outputChannel, numRows, numCols, filter, filterHeight, filterWidth);

    checkCuda(cudaDeviceSynchronize()); 

    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl; 
    }
    checkCuda(cudaDeviceSynchronize()); 
}

void rgbaBlurDevice(const uchar4 * const d_rgbaImage, uchar4 * const d_blurRGBAImage, size_t numRows, size_t numCols, const float * const filter, size_t filterHeight, size_t filterWidth)
{

    size_t numPixels = numRows * numCols;

    // Pointers to RGB channels
    unsigned char * redChannel, * greenChannel, * blueChannel;
    unsigned char * redBlurChannel, * greenBlurChannel, * blueBlurChannel;

    float * d_filter;

    // Preprocess
    // Allocate memories on device
    checkCuda(cudaMalloc(&redChannel, sizeof(unsigned char) * numPixels));
    checkCuda(cudaMalloc(&blueChannel, sizeof(unsigned char) * numPixels));
    checkCuda(cudaMalloc(&greenChannel, sizeof(unsigned char) * numPixels));

    checkCuda(cudaMalloc(&redBlurChannel, sizeof(unsigned char) * numPixels));
    checkCuda(cudaMalloc(&blueBlurChannel, sizeof(unsigned char) * numPixels));
    checkCuda(cudaMalloc(&greenBlurChannel, sizeof(unsigned char) * numPixels));

    checkCuda(cudaMalloc(&d_filter, sizeof(float) * filterHeight * filterWidth));
    checkCuda(cudaMemcpy(d_filter, filter, sizeof(float) * filterHeight * filterWidth, cudaMemcpyHostToDevice));

    // Separate RGB channels from RGBA image
    separateChannelsDevice(d_rgbaImage, numRows, numCols, redChannel, greenChannel, blueChannel);


    // Blur channels
    channelBlurDevice(redChannel, redBlurChannel, numRows, numCols, d_filter, filterHeight, filterWidth);
    channelBlurDevice(greenChannel, greenBlurChannel, numRows, numCols, d_filter, filterHeight, filterWidth);
    channelBlurDevice(blueChannel, blueBlurChannel, numRows, numCols, d_filter, filterHeight, filterWidth);

    // Combine channels to RGBA image
    recombineChannelsDevice(redBlurChannel, greenBlurChannel, blueBlurChannel, d_blurRGBAImage, numRows, numCols);

    checkCuda(cudaFree(redChannel));
    checkCuda(cudaFree(greenChannel));
    checkCuda(cudaFree(blueChannel));

    checkCuda(cudaFree(redBlurChannel));
    checkCuda(cudaFree(greenBlurChannel));
    checkCuda(cudaFree(blueBlurChannel));

    checkCuda(cudaFree(d_filter));
}

