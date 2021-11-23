#include "hough.h"

#include <libutils/rasserts.h>

#include <opencv2/imgproc.hpp>

#define _USE_MATH_DEFINES

#include <math.h>
#include <set>

//double toRadians(double degrees)
//{
//    const double PI = 3.14159265358979323846264338327950288;
//    return degrees * PI / 180.0;
//}
//
//double estimateR(double x0, double y0, double theta0radians)
//{
//    double r0 = x0 * cos(theta0radians) + y0 * sin(theta0radians);
//    return r0;
//}
//
bool operator<(const PolarLineExtremum &a, const PolarLineExtremum &b) { return a.votes > b.votes; }

cv::Mat buildHough(cv::Mat sobel) {
    rassert(sobel.type() == CV_32FC1, 237128273918006);
    int width = sobel.cols;
    int height = sobel.rows;
    int max_r = (int) sqrt(width * width + height * height);

    cv::Mat accumulator(max_r, max_theta, CV_32FC1, cv::Scalar_(0));

    for (int y0 = 0; y0 < height; ++y0) {
        for (int x0 = 0; x0 < width; ++x0) {
            float strength = sobel.at<float>(y0, x0);
            for (int theta0 = 0; theta0 < max_theta; ++theta0) {
                float theta0rad = M_PI / 180.0 * theta0;
                float theta1rad = M_PI / 180.0 * (theta0 + 1);
                int r0 = x0 * cos(theta0rad) + y0 * sin(theta0rad);
                int r1 = x0 * cos(theta1rad) + y0 * sin(theta1rad);
                if ((int) r0 < 0 || (int) r0 >= max_r)
                    continue;

                int minR = std::max(0, std::min(r0, r1));
                int maxR = std::min(max_r - 1, std::max(r0, r1));
                for (int i = minR; i <= maxR; i++) {
                    accumulator.at<float>(i, (theta0 + 1) % max_theta) += strength / 2.0;
                    accumulator.at<float>(i, theta0) += strength / 2.0;

                }
            }
        }
    }
    return accumulator;
    // TODO скопируйте сюда свою реализацию построения пространства Хафа из прошлого задания - lesson08
}


std::set<PolarLineExtremum> findLocalExtremums(cv::Mat houghSpace, int radius) {
    rassert(houghSpace.type() == CV_32FC1, 237128273918006);
    std::set<PolarLineExtremum> res;
    for (int r = 0; r < houghSpace.rows; r++) {
        for (int theta = 0; theta < houghSpace.cols; theta++) {
            bool isMax = true;
            for (int dr = -radius; dr <= radius; dr++) {
                for (int dtheta = -radius; dtheta <= radius; dtheta++) {
                    if (dr + r < 0 || dr + r >= houghSpace.rows || dtheta + theta < 0 ||
                        dtheta + theta >= houghSpace.cols)
                        continue;
                    isMax = isMax && (houghSpace.at<float>(r, theta) >= houghSpace.at<float>(r + dr, theta + dtheta));
                }
            }
            if (isMax) {
                res.insert(PolarLineExtremum(theta, r, houghSpace.at<float>(r, theta)));
            }
        }
    }
    return res;
    // TODO скопируйте сюда свою реализацию извлечения экстремумов из прошлого задания - lesson08
}

std::set<PolarLineExtremum> filterStrongLines(std::set<PolarLineExtremum> allLines, double thresholdFromWinner) {
    std::set<PolarLineExtremum> res;
    if (allLines.empty())
        return res;
    double maxVal = allLines.begin()->votes;
    res.insert(*allLines.begin());
    for (auto it = allLines.begin(); it++, it != allLines.end();) {
        if (it->votes >= thresholdFromWinner * maxVal) {
            res.insert(*it);
        } else break;
    }
    return res;
    // TODO скопируйте сюда свою реализацию фильтрации сильных прямых из прошлого задания - lesson08
}

cv::Mat drawCirclesOnExtremumsInHoughSpace(cv::Mat houghSpace, std::set<PolarLineExtremum> lines, int radius) {
    // TODO Доделайте эту функцию - пусть она скопирует картинку с пространством Хафа и отметит на ней красным кружком указанного радиуса (radius) места где были обнаружены экстремумы (на базе списка прямых)

    // делаем копию картинки с пространством Хафа (чтобы не портить само пространство Хафа)
    cv::Mat houghSpaceWithCrosses = houghSpace.clone();

    // проверяем что пространство состоит из 32-битных вещественных чисел (т.е. картинка одноканальная)
    rassert(houghSpaceWithCrosses.type() == CV_32FC1, 347823472890137);

    // но мы хотим рисовать КРАСНЫЙ кружочек вокруг найденных экстремумов, а значит нам не подходит черно-белая картинка
    // поэтому ее надо преобразовать в обычную цветную BGR картинку
    cv::cvtColor(houghSpaceWithCrosses, houghSpaceWithCrosses, cv::COLOR_GRAY2BGR);
    // проверяем что теперь все хорошо и картинка трехканальная (но при этом каждый цвет - 32-битное вещественное число)
    rassert(houghSpaceWithCrosses.type() == CV_32FC3, 347823472890148);

    for (auto line: lines) {

        // Пример как рисовать кружок в какой-то точке (закомментируйте его):
        cv::Point point(line.theta, line.r);
        cv::Scalar color(0, 0, 255); // BGR, т.е. красный цвет
        cv::circle(houghSpaceWithCrosses, point, radius, color);

        // TODO отметьте в пространстве Хафа красным кружком радиуса radius экстремум соответствующий прямой line
    }

    return houghSpaceWithCrosses;
}

cv::Point PolarLineExtremum::intersect(PolarLineExtremum that) {
    // Одна прямая - наш текущий объект (this) у которого был вызван этот метод, у этой прямой такие параметры:
    double theta0 = this->theta;
    double r0 = this->r;
    double a1 = cos(M_PI / 180.0 * theta0);
    double b1 = sin(M_PI / 180.0 * theta0);
    double c1 = -r0;

    // Другая прямая - другой объект (that) который был передан в этот метод как аргумент, у этой прямой такие параметры:
    double theta1 = that.theta;
    double r1 = that.r;
    double a2 = cos(M_PI / 180.0 * theta1);
    double b2 = sin(M_PI / 180.0 * theta1);
    double c2 = -r1;

    // TODO реализуйте поиск пересечения этих двух прямых, напоминаю что формула прямой описана тут - https://www.polarnick.com/blogs/239/2021/school239_11_2021_2022/2021/11/02/lesson8-hough-transform.html
    // после этого загуглите как искать пересечение двух прямых, пример запроса: "intersect two 2d lines"
    // и не забудьте что cos/sin принимают радианы (используйте toRadians)
    if ((a1 * b2 - a2 * b1) != 0) {
        int x = -(c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1);
        int y = -(a1 * c2 - a2 * c1) / (a1 * b2 - a2 * b1);
        return cv::Point(x, y);
    } else {
        return cv::Point(-999, -999);
    }
}

// TODO Реализуйте эту функцию - пусть она скопирует картинку и отметит на ней прямые в соответствии со списком прямых
cv::Mat drawLinesOnImage(cv::Mat img, std::set<PolarLineExtremum> lines) {
    // делаем копию картинки (чтобы при рисовании не менять саму оригинальную картинку)
    cv::Mat imgWithLines = img.clone();

    // проверяем что картинка черно-белая (мы ведь ее такой сделали ради оператора Собеля) и 8-битная
    rassert(imgWithLines.type() == CV_8UC1, 45728934700167);

    // но мы хотим рисовать КРАСНЫЕ прямые, а значит нам не подходит черно-белая картинка
    // поэтому ее надо преобразовать в обычную цветную BGR картинку с 8 битами в каждом пикселе
    cv::cvtColor(imgWithLines, imgWithLines, cv::COLOR_GRAY2BGR);
    rassert(imgWithLines.type() == CV_8UC3, 45728934700172);

    // выпишем размер картинки
    int width = imgWithLines.cols;
    int height = imgWithLines.rows;

    for (auto line: lines) {

        // нам надо найти точки на краях картинки
        cv::Point pointA;
        cv::Point pointB;

        // TODO создайте четыре прямых соответствующих краям картинки (на бумажке нарисуйте картинку и подумайте какие theta/r должны быть у прямых?):
        // напоминаю - чтобы посмотреть какие аргументы требует функция (или в данном случае конструктор объекта) - нужно:
        // 1) раскомментировать эти четыре строки ниже
        // 2) поставить каретку (указатель где вы вводите новые символы) внутри скобок функции (или конструктора, т.е. там где были три вопроса: ???)
        // 3) нажать Ctrl+P чтобы показать список параметров (P=Parameters)
        PolarLineExtremum leftImageBorder(90, 0, 0);
        PolarLineExtremum bottomImageBorder(0, height, 0);
        PolarLineExtremum rightImageBorder(0, width, 0);
        PolarLineExtremum topImageBorder(90, 0, 0);

        // TODO воспользуйтесь недавно созданной функций поиска пересечения прямых чтобы найти точки пересечения краев картинки:
        pointA = line.intersect(leftImageBorder);
        pointB = line.intersect(rightImageBorder);
        if ((pointA.x == -999) && (pointA.y == -999)) {
            // TODO а в каких случаях нужно использовать пересечение с верхним и нижним краем картинки?
            pointA = line.intersect(bottomImageBorder);
            pointB = line.intersect(topImageBorder);
        }

        // TODO переделайте так чтобы цвет для каждой прямой был случайным (чтобы легче было различать близко расположенные прямые)
        cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
        cv::line(imgWithLines, pointA, pointB, color);
    }

    return imgWithLines;
}
