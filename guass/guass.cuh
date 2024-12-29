//
// Created by chenk on 2024/12/29.
//

#ifndef GUASS_CUH
#define GUASS_CUH
#include <opencv2/opencv.hpp>
cv::Mat guassCPU(const cv::Mat &src, const cv::Size &guass_size, double sigmaX, double sigmaY);
cv::Size createKernelSizeBysigma(int imageType, cv::Size ksize, double sigma1, double sigma2);
void guassGPU(unsigned char* input, unsigned char* output, size_t numRows, size_t numCols, int kernelRowSize, int kernelColSize, float* kernelRow, float* kernelCol);
__global__ void GaussianKernel1d_row(unsigned char* src, unsigned char* dst, int height, int width, int filterWidth, float* filter);
__global__ void GaussianKernel1d_col(unsigned char* src, unsigned char* dst, int height, int width, int filterWidth, float* filter);
#endif //GUASS_CUH
