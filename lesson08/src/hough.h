#pragma once

#include <opencv2/highgui.hpp>
#include <set>

const int max_theta = 360;
cv::Mat buildHough(cv::Mat sobel);

class PolarLineExtremum {
public:
    double theta;
    double r;
    double votes;

    PolarLineExtremum(double theta, double r, double votes)
    {
        this->theta = theta;
        this->r = r;
        this->votes = votes;
    }
};
bool operator<(const PolarLineExtremum &a, const PolarLineExtremum &b);

std::set<PolarLineExtremum> findLocalExtremums(cv::Mat houghSpace, int radius = 3);

std::vector<PolarLineExtremum> filterStrongLines(std::set<PolarLineExtremum> allLines, double thresholdFromWinner);