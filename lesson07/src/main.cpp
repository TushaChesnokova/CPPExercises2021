#include <filesystem>
#include <iostream>
#include <libutils/rasserts.h>
#include <math.h>

#include "hough.h"
#include "../../lesson05/src/sobel.h"

#include <opencv2/imgproc.hpp>


void test(std::string name) {
    std::string full_path = "lesson05/data/" + name + ".jpg";
    cv::Mat img = cv::imread(full_path);
    rassert(!img.empty(), 238982391080010);
    cv::Mat dxy = sobelDXY(convertBGRToGray(img));
    cv::Mat grad_x = convertDXYToDX(dxy), grad_y = convertDXYToDY(dxy);

    cv::imwrite("lesson07/resultsData/" + name + "_1_sobel_x.png", grad_x);
    cv::imwrite("lesson07/resultsData/" + name + "_2_sobel_y.png", grad_y);
    cv::imwrite("lesson07/resultsData/" + name + "_3_sobel_x.png", grad_x);
    cv::imwrite("lesson07/resultsData/" + name + "_4_sobel_y.png", grad_y);

    cv::Mat len = convertDXYToGradientLength(dxy);
    cv::imwrite("lesson07/resultsData/" + name + "_5_sobel_strength.png", len);

    cv::Mat hough = buildHough(len);
    cv::imwrite("lesson07/resultsData/" + name + "_6_hough.png", hough);
    float max_accumulated = 0.0f;
    for (int i = 0; i < hough.rows; i++) {
        for (int j = 0; j < hough.cols; j++) {
            max_accumulated = std::max(max_accumulated, hough.at<float>(i,j));
        }
    }
    hough*=0xff/max_accumulated;

    cv::imwrite("lesson07/resultsData/" + name + "_7_hough_normalized.png", hough);
}


int main() {
    try {
        // TODO посмотрите на результат (аккумулятор-пространство Хафа) на всех этих картинках (раскомментируя их одну за другой)
        // TODO подумайте и напишите здесь оветы на вопросы:
        test("line01");
        // 1) Какие координаты примерно должны бы быть у самой яркой точки в картинке line01_7_hough_normalized.png?
        // ответ:

//        test("line02");
//        // 2) Какие координаты примерно должны бы быть у самой яркой точки в картинке line02_7_hough_normalized.png?
//        // ответ:
//
//        test("line11");
//        // 3) Чем должно бы принципиально отличаться пространство Хафа относительно случая line01?
//        // ответ:
//
//        test("line12");
//        // 4) Зная правильный ответ из предыдущего случая line11 - как найти правильнйы ответ для line12?
//        // ответ:
//
//        test("line21_water_horizont");
//        // 5) Сколько должно бы быть ярких точек?
//        // ответ:
//
//        test("multiline1_paper_on_table");
//        // 6) Сколько должно бы быть ярких точек? Сколько вы насчитали в пространстве Хафа?
//        // ответ:
//
//        test("multiline2_paper_on_table");
//        // 7) Сколько должно бы быть ярких точек? Сколько вы насчитали в пространстве Хафа? Есть ли интересные наблюдения относительно предыдущего случая?
//        // ответ:
//
        test("valve");
//        // 8) Какие-нибудь мысли?

        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception! " << e.what() << std::endl;
        return 1;
    }
}
