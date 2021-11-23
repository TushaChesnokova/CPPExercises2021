#include "hough.h"

#include <libutils/rasserts.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <set>

bool operator<(const PolarLineExtremum &a, const PolarLineExtremum &b) {return a.votes > b.votes;}

cv::Mat buildHough(cv::Mat sobel) {
    rassert(sobel.type() == CV_32FC1, 237128273918006);
    int width = sobel.cols;
    int height = sobel.rows;
    int max_r = (int)sqrt(width*width+height*height);

    cv::Mat accumulator(max_r, max_theta, CV_32FC1, cv::Scalar_(0));

    for (int y0 = 0; y0 < height; ++y0) {
        for (int x0 = 0; x0 < width; ++x0) {
            float strength = sobel.at<float>(y0, x0);
            for (int theta0 = 0; theta0 < max_theta; ++theta0) {
                float theta0rad = M_PI/180.0*theta0;
                float theta1rad = M_PI/180.0*(theta0+1);
                int r0 = x0*cos(theta0rad)+y0*sin(theta0rad);
                int r1 = x0*cos(theta1rad)+y0*sin(theta1rad);
                if((int)r0 < 0 || (int)r0 >= max_r)
                    continue;

                int minR = std::max(0,std::min(r0,r1));
                int maxR = std::min(max_r-1,std::max(r0,r1));
                for(int i = minR; i <= maxR; i++){
                    accumulator.at<float>(i, (theta0+1)%max_theta) += strength/2.0;
                    accumulator.at<float>(i, theta0) += strength/2.0;

                }
            }
        }
    }
    return accumulator;
}

std::set<PolarLineExtremum> findLocalExtremums(cv::Mat houghSpace, int radius){
    rassert(houghSpace.type() == CV_32FC1, 237128273918006);
    std::set<PolarLineExtremum> res;
    for(int r = 0; r < houghSpace.rows; r++){
        for(int theta = 0; theta < houghSpace.cols; theta++){
            bool isMax = true;
            for(int dr = -radius; dr <= radius; dr++){
                for(int dtheta = -radius; dtheta <= radius; dtheta++){
                    if(dr + r < 0 || dr + r >= houghSpace.rows || dtheta + theta < 0 || dtheta + theta >= houghSpace.cols)
                        continue;
                    isMax = isMax && (houghSpace.at<float>(r, theta) >= houghSpace.at<float>(r + dr, theta + dtheta));
                }
            }
            if(isMax) {
                res.insert(PolarLineExtremum(theta, r,houghSpace.at<float>(r, theta)));
            }
        }
    }
    return res;
}

std::vector<PolarLineExtremum> filterStrongLines(std::set<PolarLineExtremum> allLines, double thresholdFromWinner){
    std::vector<PolarLineExtremum> res;
    if(allLines.empty())
        return res;
    double maxVal = allLines.begin()->votes;
    res.emplace_back(*allLines.begin());
    for (auto it = allLines.begin(); it++, it != allLines.end() ;) {
        if(it->votes >= thresholdFromWinner*maxVal){
            res.emplace_back(*it);
        } else break;
    }
    return res;
}