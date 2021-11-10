#pragma once

#include <opencv2/highgui.hpp>

const int max_theta = 360;
cv::Mat buildHough(cv::Mat sobel);