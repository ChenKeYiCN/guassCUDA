//
// Created by chenk on 2024/12/29.
//
#include "guass.cuh"
#include <cstdio>

cv::Mat guassCPU(const cv::Mat &src, const cv::Size &guass_size, double sigmaX, double sigmaY)
{
    cv::Mat blur;
    GaussianBlur(src, blur, guass_size, sigmaX, sigmaY);
    return blur;
}

cv::Size createKernelSizeBysigma(int imageType, cv::Size ksize, double sigma1, double sigma2)
{
    cv::Size output_size = ksize;
    int depth = CV_MAT_DEPTH(imageType);
    if( sigma2 <= 0 )
        sigma2 = sigma1;
    // automatic detection of kernel size from sigma
    if( ksize.width <= 0 && sigma1 > 0 )
        output_size.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    if( ksize.height <= 0 && sigma2 > 0 )
        output_size.height = cvRound(sigma2*(depth == CV_8U ? 3 : 4)*2 + 1)|1;

    CV_Assert( output_size.width  > 0 && output_size.width  % 2 == 1 &&
               output_size.height > 0 && output_size.height % 2 == 1 );
    return output_size;
}

void guassGPU(unsigned char *input, unsigned char *output, size_t numRows, size_t numCols, int kernelRowSize,
    int kernelColSize, float *kernelRow, float *kernelCol)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    GaussianKernel1d_row<<<blocksPerGrid, threadsPerBlock>>>(input, output, numRows, numCols, kernelRowSize, kernelRow);
    GaussianKernel1d_col<<<blocksPerGrid, threadsPerBlock>>>(output, input, numRows, numCols, kernelColSize, kernelCol);
}

__global__ void GaussianKernel1d_row(unsigned char *src, unsigned char *dst, int height, int width, int filterWidth, float *filter)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y; //二维块的当前行
    int x = blockDim.x * blockIdx.x + threadIdx.x; //二维块的当前列
    int ind = y * width + x;
    if (y >= height || x >= width)
    {
        return;
    }
    float color = 0.0f;
    int padding = filterWidth / 2;

    for (int i = 0; i < filterWidth; i++)
    {
        float k = filter[i];
        // printf("wtf i=%d, width=%d, k=%f\n", i, filterWidth, k);
        int origin_x = x + i - padding;
        if(origin_x < 0)
            origin_x = std::abs(origin_x) - 1;
        else if(origin_x >= width)
            origin_x = 2 * width - origin_x - 1;

        float s = (float)src[y * width + origin_x];
        color += k * s;
    }
    dst[ind] = min(max((int)color, 0), 255);
}

__global__ void GaussianKernel1d_col(unsigned char *src, unsigned char *dst, int height, int width, int filterWidth, float *filter)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y; //二维块的当前行
    int x = blockDim.x * blockIdx.x + threadIdx.x; //二维块的当前列
    int ind = y * width + x;
    if (y >= height || x >= width)
    {
        return;
    }
    float color = 0.0f;
    int padding = filterWidth / 2;

    for (int i = 0; i < filterWidth; i++)
    {
        float k = filter[i];
        int origin_y = y + i - padding;
        if(origin_y < 0)
            origin_y = std::abs(origin_y) - 1;
        else if(origin_y >= height)
            origin_y = 2 * height - origin_y - 1;

        float s = (float)src[origin_y * width + x];
        color += k * s;
    }
    // dst[x * width + y] = min(max((int)color, 0), 255);
    dst[ind] = min(max((int)color, 0), 255);
    //    dst[ind] = color;
}

