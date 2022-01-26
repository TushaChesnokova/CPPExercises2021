#include <iostream>
#include "parseSymbols.h"

std::vector<std::pair<cv::Point, cv::Mat>> splitSymbols(cv::Mat img)
{
    cv::Mat img2 = img.clone();
    std::vector<std::pair<cv::Point, cv::Mat>> symbols;
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
        symbols.emplace_back(std::make_pair(box.tl(),cv::Mat(img, box)));
    }
    return symbols;
}

bool compY (const std::pair<cv::Point, cv::Mat> & a, const std::pair<cv::Point, cv::Mat> & b) {
    return a.first.y < b.first.y;
}

bool compX (const std::pair<cv::Point, cv::Mat> & a, const std::pair<cv::Point, cv::Mat> & b) {
    return a.first.x < b.first.x;
}

std::vector<std::vector<std::pair<cv::Point, cv::Mat>>> sort(std::vector<std::pair<cv::Point, cv::Mat>> symbols){
    std::vector<std::vector<std::pair<cv::Point, cv::Mat>>> res;
    std::sort(symbols.begin(), symbols.end(), compY);
    auto a = 0;
    auto b = 1;
    double midH = 0;
    for(const auto & i : symbols){
        midH += i.second.rows;
    }
    midH /= symbols.size();
    midH *= 1.5;
    while(b < symbols.size()){
        if((symbols[b].first.y - symbols[b - 1].first.y) >= midH){
            std::vector<std::pair<cv::Point, cv::Mat>> X;
            for(;a < b; a++){
                X.push_back(symbols[a]);
            }
            sort(X.begin(), X.end(), compX);
            res.push_back(X);
        }
        b++;
    }
    std::vector<std::pair<cv::Point, cv::Mat>> X;
    for(;a < b; a++){
        X.push_back(symbols[a]);
    }
    sort(X.begin(), X.end(), compX);
    res.push_back(X);
    return res;
}