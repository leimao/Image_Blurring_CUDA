#include <string>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "utils.h"
#include "blur.h"

int main(int argc, char ** argv)
{
    std::string input_filepath;
    std::string output_filepath;

    size_t kernelWidth;
    float kernelSigma;

    output_filepath = "image_blur.jpg";
    kernelWidth = 51;
    kernelSigma = 10.0;

    switch (argc)
    {
        case 2:
        {
            input_filepath = std::string(argv[1]);
            break;
        }
        case 3: 
        {
            input_filepath = std::string(argv[1]);
            output_filepath = std::string(argv[2]);
            break;
        }
        case 4: 
        {
            input_filepath = std::string(argv[1]);
            output_filepath = std::string(argv[2]);
            kernelWidth = std::stoi(argv[3]);
            if (kernelWidth % 2 == 0)
            {
                std::cout << "Gaussian kernel width has to be odd value." << std::endl;
                return -1;
            }
            break;
        }
        case 5: 
        {
            input_filepath = std::string(argv[1]);
            output_filepath = std::string(argv[2]);
            kernelWidth = std::stoi(argv[3]);
            if (kernelWidth % 2 == 0)
            {
                std::cout << "Gaussian kernel width has to be odd value." << std::endl;
                return -1;
            }
            kernelSigma = std::stof(argv[4]);
            break;
        }
        default:
        {
            std::cout << "Too many arguments." << std::endl;
        }

    }

    float * gaussianFilter = generateGaussianBlurKernel(kernelWidth, kernelSigma);

    cv::Mat bgrImage;
    cv::Mat rgbaImage;
    cv::Mat blurRGBAImage;
    cv::Mat blurBGRImage;

    // Read BGR image
    bgrImage = cv::imread(input_filepath, cv::IMREAD_COLOR);
    // Convert BGR to RGBA image
    cv::cvtColor(bgrImage, rgbaImage, cv::COLOR_BGR2RGBA);

    size_t numRows = rgbaImage.rows;
    size_t numCols = rgbaImage.cols;
    size_t numPixels = numRows * numCols;

    uchar4 * h_blurBGRImage;
    // Pointers to RGBA images
    uchar4 * h_rgbaImage, * d_rgbaImage;
    // Pointers to blur images
    // h_blurImage: pointer to blurImage on host
    uchar4 * h_blurRGBAImage, * d_blurRGBAImage;

    // Allocate memory for final output on host
    // CV_8UC4 8-bit image with 4 channel
    blurRGBAImage.create(numRows, numCols, CV_8UC4);

    // Get pointers to data on host
    h_blurRGBAImage = blurRGBAImage.ptr<uchar4>(0);
    h_blurBGRImage = blurBGRImage.ptr<uchar4>(0);
    h_rgbaImage = rgbaImage.ptr<uchar4>(0);

    // Preprocess
    // Allocate memories on device
    checkCuda(cudaMalloc(&d_blurRGBAImage, sizeof(uchar4) * numPixels));
    checkCuda(cudaMalloc(&d_rgbaImage, sizeof(uchar4) * numPixels));
    
    checkCuda(cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
    //copyMemDevice(h_rgbaImage, d_rgbaImage, numPixels);

    // Blur RGBA on device
    rgbaBlurDevice(d_rgbaImage, d_blurRGBAImage, numRows, numCols, gaussianFilter, kernelWidth, kernelWidth);

    // Copy data from device to host
    checkCuda(cudaMemcpy(h_blurRGBAImage, d_blurRGBAImage, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

    // Convert BGR to RGBA image
    cv::cvtColor(blurRGBAImage, blurBGRImage, cv::COLOR_RGBA2BGR);

    // Write output to file
    cv::imwrite(output_filepath, blurBGRImage);

    // Free memory
    free(gaussianFilter);
    checkCuda(cudaFree(d_rgbaImage));
    checkCuda(cudaFree(d_blurRGBAImage));

    return 0;
}