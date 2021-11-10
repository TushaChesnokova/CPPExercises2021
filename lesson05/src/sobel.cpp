#include "sobel.h"

#include <libutils/rasserts.h>
#include <iostream>


cv::Mat convertBGRToGray(cv::Mat img) {
    int height = img.rows;
    int width = img.cols;
    cv::Mat grayscaleImg(height, width, CV_32FC1);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            cv::Vec3b color = img.at<cv::Vec3b>(j, i);
            float grayIntensity = (color[0]+color[1]+color[2])/3.0;
            grayscaleImg.at<float>(j, i) = grayIntensity;
        }
    }
    return grayscaleImg;
}


cv::Mat sobelDXY(cv::Mat img) {
    int height = img.rows;
    int width = img.cols;
    cv::Mat dxyImg(height, width, CV_32FC2);
    rassert(img.type() == CV_32FC1, 23781792319049);
    int dxSobelKoef[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1},
    };

    int dySobelKoef[3][3] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1},
    };

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            float dxSum = 0.0f;
            float dySum = 0.0f;
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if(j+dj >= height || j+dj < 0 || i+di >= width || i+di < 0)
                        continue;
                    float intensity = img.at<float>(j + dj, i + di);
                    dxSum += dxSobelKoef[1 + dj][1 + di] * intensity;
                    dySum += dySobelKoef[1 + dj][1 + di] * intensity;
                }
            }

            dxyImg.at<cv::Vec2f>(j, i) = cv::Vec2f(dxSum/4.0, dySum/4.0);
        }
    }

    return dxyImg;
}

cv::Mat convertDXYToDX(cv::Mat img) {
    rassert(img.type() == CV_32FC2,
            238129037129092);
    int width = img.cols;
    int height = img.rows;
    cv::Mat dxImg(height, width, CV_32FC1);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            cv::Vec2f dxy = img.at<cv::Vec2f>(j, i);
            dxImg.at<float>(j, i) = abs(dxy[0]);
        }
    }
    return dxImg;
}

cv::Mat convertDXYToDY(cv::Mat img) {
    rassert(img.type() == CV_32FC2,
            238129037129092);
    int width = img.cols;
    int height = img.rows;
    cv::Mat dyImg(height, width, CV_32FC1);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            cv::Vec2f dxy = img.at<cv::Vec2f>(j, i);
            dyImg.at<float>(j, i) = abs(dxy[1]);
        }
    }
    return dyImg;
}

cv::Mat convertDXYToGradientLength(cv::Mat img) {
    rassert(img.type() == CV_32FC2,
            238129037129092);
    int width = img.cols;
    int height = img.rows;
    cv::Mat dxyImg(height, width, CV_32FC1);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            cv::Vec2f dxy = img.at<cv::Vec2f>(j, i);
            dxyImg.at<float>(j, i) = sqrt(pow(dxy[0],2)+pow(dxy[1],2));
        }
    }
    return dxyImg;
}