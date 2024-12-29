//
// Created by chenk on 2024/12/29.
//

#ifndef UTILS_CUH
#define UTILS_CUH
#include <string>
#include <opencv2/opencv.hpp>

cv::Mat readImage(const std::string &fileName, bool read_gray=true, bool show=true);
void showImage(const cv::Mat &img, std::string windowName="image");
#endif //UTILS_CUH
