#pragma once

#include <opencv2/highgui.hpp>
#include <libutils/rasserts.h>
#include <opencv2/imgproc.hpp>

#include <vector>

std::vector<std::pair<cv::Point, cv::Mat>> splitSymbols(cv::Mat img2);

std::vector<std::vector<std::pair<cv::Point, cv::Mat>>> sort(std::vector<std::pair<cv::Point, cv::Mat>> symbols);