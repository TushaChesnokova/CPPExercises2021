#include "hough.h"

#include <libutils/rasserts.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

cv::Mat buildHough(cv::Mat sobel) {
    rassert(sobel.type() == CV_32FC1, 865465465132132);
    int width = sobel.cols;
    int height = sobel.rows;
    int max_r = (int)sqrt(width*width+height*height);

    cv::Mat accumulator(max_r, max_theta, CV_32FC1, cv::Scalar_(0));

    for (int y0 = 0; y0 < height; ++y0) {
        for (int x0 = 0; x0 < width; ++x0) {
            float strength = sobel.at<float>(y0, x0);
            for (int theta0 = 0; theta0 < max_theta; ++theta0) {
                float theta0rad = M_PI/180.0*theta0;
                float r0 = x0*cos(theta0rad)+y0*sin(theta0rad);
                if((int)r0 < 0 || (int)r0 >= max_r)
                    continue;
                accumulator.at<float>((int)r0, theta0) += strength;
            }
        }
    }
    return accumulator;
}