#include "parseSymbols.h"

std::vector<cv::Mat> splitSymbols(cv::Mat img)
{
    cv::Mat img2 = img.clone();
    std::vector<cv::Mat> symbols;
    cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);

    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(img, img, 0xff, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 10);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    cv::adaptiveThreshold(img2, img2, 0xff, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 20);
    cv::dilate(img2, img2, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));
    std::vector<std::vector<cv::Point>> contoursPoints2;
    cv::findContours(img2, contoursPoints2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (const auto& points : contoursPoints2) {
        cv::Rect box = cv::boundingRect(points);
        symbols.emplace_back(img, box);
    }
    return symbols;
}