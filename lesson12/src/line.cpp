#include "line.h"

#include <libutils/rasserts.h>

#include <opencv2/imgproc.hpp>

#include <random>

double Line::getYFromX(double x)
{
    rassert(b != 0.0, 2734832748932790061); // случай вертикальной прямой не рассматривается для простоты
    double y = (-c-a*x)/b;
    return y;
}

std::vector<cv::Point2f> Line::generatePoints(int n,
                                              double fromX, double toX,
                                              double gaussianNoiseSigma)
{
    std::vector<cv::Point2f> points;

    // пусть зерно случайности порождающее последовательность координат будет однозначно опредляться по числу точек
    unsigned int randomSeed = n;
    std::mt19937 randomGenerator(randomSeed); // это генератор случайных чисел (см. https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine )

    for (int i = 0; i < n; ++i) {
        // это правило генерации случайных чисел - указание какие мы хотим координаты x - равномерно распределенные в диапазоне от fromX  до toX
        std::uniform_real_distribution<> xDistribution(fromX, toX);

        double x = xDistribution(randomGenerator);

        // найдем идеальную координату y для данной координаты x:
        double idealY = getYFromX(x);

        // указание какую мы хотим координату y - распределенную около idealY в соответствии с распределением Гаусса (т.н. нормальное распределение)
        std::normal_distribution<> yDistribution(idealY, gaussianNoiseSigma);
        double y = yDistribution(randomGenerator);

        points.emplace_back(x, y);
    }

    return points;
}

// эта функция рисует на картинке указанные точки
// при этом если картинка пустая - эта функция должна увеличить картинку до размера в который впишутся все точки
void plotPoints(cv::Mat &img, std::vector<cv::Point2f> points, double scale, const cv::Scalar& color)
{
    rassert(!points.empty(), 347238947320012);

    if (img.empty()) {
        // если картинка пустая - нужно увеличить картинку до размера в который впишутся все точки
        float maxX = 0.0f;
        float maxY = 0.0f;
        for (auto & point : points) {
            maxX = std::max(maxX, point.x);
            maxY = std::max(maxY, point.y);
        }
        cv::Scalar black(0, 0, 0);
        // увеличим на 10% размер картинки чтобы точки были не совсем на краю
        maxX *= 1.1;
        maxY *= 1.1;
        // создаем картинку нужного размера (и сразу указываем что она заполнена черным)
        int nrows = (int) (maxY * scale);
        int ncols = (int) (maxX * scale);
        img = cv::Mat(nrows, ncols, CV_8UC3, black);

        // рисуем текст для указания координат в углах картинки
        cv::Scalar white(255, 255, 255);
        float textHeight = cv::getTextSize("0;0", cv::FONT_HERSHEY_DUPLEX, 1.0, 1, nullptr).height;
        cv::putText(img, "0;0", cv::Point(0, textHeight), cv::FONT_HERSHEY_DUPLEX, 1.0, white);

        cv::putText(img, "0;" + std::to_string(maxY), cv::Point(0, nrows - 5), cv::FONT_HERSHEY_DUPLEX, 1.0, white);

        std::string textTopRight = std::to_string(maxX) + ";0";
        float textWidth = cv::getTextSize(textTopRight, cv::FONT_HERSHEY_DUPLEX, 1.0, 1, nullptr).width;
        cv::putText(img, textTopRight, cv::Point(ncols-textWidth, textHeight), cv::FONT_HERSHEY_DUPLEX, 1.0, white);
    } else {
        rassert(img.type() == CV_8UC3, 34237849200017);
    }

    for (auto & point : points) {
        cv::circle(img, point * scale, 5, color, 2);
    }
}

// метод прямой позволяющий нарисовать ее на картинке (т.е. на простом графике)
void Line::plot(cv::Mat &img, double scale, const cv::Scalar& color)
{
    rassert(!img.empty(), 3478342937820055);
    rassert(img.type() == CV_8UC3, 34237849200055);

    cv::line(img, cv::Point(0, getYFromX(0)*scale), cv::Point(img.cols, getYFromX(img.cols/scale)*scale), color);
}

Line fitLineFromTwoPoints(const cv::Point2f& a, const cv::Point2f& b)
{
    //rassert(a.x != b.x, 23892813901800104); // для упрощения можно считать что у нас не бывает вертикальной прямой
    return Line(b.y-a.y, a.x-b.x, b.x*a.y-b.y*a.x);
}

double Line::getDistanceSqr(const cv::Point2f& p) const{
    return pow(a*p.x+b*p.y+c,2)/(a*a+b*b);
}

Line fitLineFromNPoints(const std::vector<cv::Point2f>& points)
{
    double minDis = DBL_MAX;
    Line *l = nullptr;
    for(const auto& p : points)
        for(const auto& p2 : points){
            if(p == p2)
                continue;
            Line l2 = fitLineFromTwoPoints(p,p2);
            double dis = 0;
            for(const auto& p3 : points)
                dis+=l2.getDistanceSqr(p3);
            if(minDis > dis)
                l = &l2;
        }
    return *l;
}

Line fitLineFromNNoisyPoints(const std::vector<cv::Point2f>& points)
{
    int rad =15;
    int ro;
    int bestVotes = 0;
    int votes;
    Line bestLine = fitLineFromTwoPoints(points[0], points[1]) ;
    Line randLine(0,0,0);
    for (int i = 0; i< 1000; i++){
       cv::Point a = points[rand()%points.size()];
       cv::Point b = points[rand()%points.size()];
       while(a==b) {
           a = points[rand()%points.size()];
       }
        randLine = fitLineFromTwoPoints(a, b);
       for (int j=0; j< points.size(); j++){
           ro = std::abs(randLine.a*points[j].x+randLine.b*points[j].y+randLine.c)/std::sqrt(std::pow(randLine.a,2)+std::pow(randLine.b,2));
           if(ro<=rad);
           votes++;
       }
       if (votes>bestVotes) {
           bestLine = randLine;
           bestVotes=votes;
       }
    }
    // TODO 06 БОНУС - реализуйте построение прямой по многим точкам включающим нерелевантные (такое чтобы прямая как можно лучше учитывала НАИБОЛЬШЕЕ число точек)
    return bestLine;
}

std::vector<cv::Point2f> generateRandomPoints(int n,
                                              double fromX, double toX,
                                              double fromY, double toY)
{
    std::vector<cv::Point2f> points;

    // пусть зерно случайности порождающее последовательность координат будет однозначно опредляться по числу точек
    unsigned int randomSeed = n;
    std::mt19937 randomGenerator(randomSeed); // это генератор случайных чисел (см. https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine )

    for (int i = 0; i < n; ++i) {
        // это правило генерации случайных чисел - указание какие мы хотим координаты x - равномерно распределенные в диапазоне от fromX  до toX
        std::uniform_real_distribution<> xDistribution(fromX, toX);
        std::uniform_real_distribution<> yDistribution(fromY, toY);

        double x = xDistribution(randomGenerator);
        double y = yDistribution(randomGenerator);

        points.push_back(cv::Point2f(x, y));
    }

    return points;
}

std::ostream& operator << (std::ostream& os, const Line& line)
{
    os << line.a << "*x + " << line.b << "*y + " << line.c << " = 0";
    return os;
}