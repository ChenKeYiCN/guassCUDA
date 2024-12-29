//
// Created by chenk on 2024/12/29.
//
#include "utils.cuh"

cv::Mat readImage(const std::string &fileName, bool read_gray, bool show)
{
    int flag = read_gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    cv::Mat image = cv::imread(fileName, flag);
    if(show)
    {
        showImage(image);
    }
    return image;
}

void showImage(const cv::Mat &img, std::string windowName)
{
    cv::imshow("windowName", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
