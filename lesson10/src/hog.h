#pragma once

#include <opencv2/highgui.hpp>

#include <vector>

// Здесь мы будем хранить параметры нашего алгоритма:
#define NBINS 8 // число корзин (т.е. количество разных направлений угла которые мы рассматриваем)


typedef std::vector<double> HoG;

HoG buildHoG(cv::Mat grad_x, cv::Mat grad_y);

HoG buildHoG(cv::Mat img);

std::ostream& operator << (std::ostream& os, const HoG& hog);

double distance(HoG a, HoG b);

