#include <filesystem>
#include <iostream>
#include <libutils/rasserts.h>

#include "parseSymbols.h"
#include "hog.h"

#include <opencv2/imgproc.hpp>

#define NSAMPLES_PER_LETTER 5
#define LETTER_DIR_PATH std::string("lesson11/generatedData/letters")


cv::Scalar randColor() {
    return cv::Scalar(128 + rand() % 128, 128 + rand() % 128, 128 + rand() % 128); // можно было бы брать по модулю 255, но так цвета будут светлее и контрастнее
}


cv::Mat drawContours(int rows, int cols, std::vector<std::vector<cv::Point>> contoursPoints) {
    cv::Mat blackImage(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat imageWithContoursPoints = blackImage.clone();
    for (const auto& points : contoursPoints) {
        cv::Scalar contourColor = randColor();
        for (const cv::Point& point : points) {
            imageWithContoursPoints.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(contourColor[0], contourColor[1], contourColor[2]);
        }

    }

    return imageWithContoursPoints;
}


void test(std::string name, std::string k) {
    std::cout << "Processing " << name << "/" << k << "..." << std::endl;

    std::string full_path = "lesson11/data/" + name + "/" + k + ".png";

    // создаем папочки в которые будем сохранять картинки с промежуточными результатами
    std::filesystem::create_directory("lesson11/resultsData/" + name);
    std::string out_path = "lesson11/resultsData/" + name + "/" + k;
    std::filesystem::create_directory(out_path);

    // считываем оригинальную исходную картинку
    cv::Mat original = cv::imread(full_path);
    rassert(!original.empty(), 238982391080010);
    rassert(original.type() == CV_8UC3, 23823947238900020);

    // сохраняем ее сразу для удобства
    cv::imwrite(out_path + "/00_original.jpg", original);

    // преобразуем в черно-белый цвет и тоже сразу сохраняем
    cv::Mat img;
    cv::cvtColor(original, img, cv::COLOR_BGR2GRAY);
    cv::imwrite(out_path + "/01_grey.jpg", img);

    cv::Mat binary;
    cv::threshold(img, binary, 0xff/2, 0xff, cv::THRESH_BINARY_INV);
    cv::imwrite(out_path + "/02_binary_thresholding.jpg", binary);

    cv::adaptiveThreshold(img, binary, 0xff, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11,10);
    cv::imwrite(out_path + "/03_adaptive_thresholding.jpg", binary);

    cv::Mat binary_eroded;
    cv::erode(binary, binary_eroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));
    cv::imwrite(out_path + "/04_erode.jpg", binary_eroded);


    cv::Mat binary_dilated;
    cv::dilate(binary, binary_dilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));
    cv::imwrite(out_path + "/05_dilate.jpg", binary_dilated);

    binary = binary_dilated;

    std::vector<std::vector<cv::Point>> contoursPoints;
    cv::findContours(binary, contoursPoints, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    std::cout << "Contours: " << contoursPoints.size() << std::endl;
    cv::Mat imageWithContoursPoints = drawContours(img.rows, img.cols, contoursPoints);
    cv::imwrite(out_path + "/06_contours_points.jpg", imageWithContoursPoints);

    std::vector<std::vector<cv::Point>> contoursPoints2;
    cv::findContours(binary, contoursPoints2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    std::cout << "Contours2: " << contoursPoints2.size() << std::endl;
    cv::Mat imageWithContoursPoints2 = drawContours(img.rows, img.cols, contoursPoints2);
    cv::imwrite(out_path + "/07_contours_points2.jpg", imageWithContoursPoints2);

    cv::Mat imgWithBoxes = original.clone();
    for (const auto& points : contoursPoints2) {
        cv::Rect box = cv::boundingRect(points);
        cv::Scalar blackColor(0, 0, 0);
        cv::rectangle(imgWithBoxes, box, blackColor, 2);
    }
    cv::imwrite(out_path + "/08_boxes.jpg", imgWithBoxes);
}

double randFontScale() {
    double min_scale = 2.5;
    double max_scale = 3.0;
    double scale = min_scale + (max_scale - min_scale) * ((rand() % 100) / 100.0);
    return scale;
}

int randThickness() {
    int min_thickness = 2;
    int max_thickness = 3;
    int thickness = min_thickness + rand() % (max_thickness - min_thickness + 1);
    return thickness;
}

int randFont() {
    int fonts[] = {
            cv::FONT_HERSHEY_SIMPLEX,
            cv::FONT_HERSHEY_PLAIN,
            cv::FONT_HERSHEY_DUPLEX,
            cv::FONT_HERSHEY_COMPLEX,
            cv::FONT_HERSHEY_TRIPLEX,
            cv::FONT_HERSHEY_COMPLEX_SMALL,
            cv::FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv::FONT_HERSHEY_SCRIPT_COMPLEX,
    };
    int nfonts = (sizeof(fonts) / sizeof(int));
    int font = rand() % nfonts;
    bool is_italic = ((rand() % 5) == 0);
    if  (is_italic) {
        font = font | cv::FONT_ITALIC;
    }

    return font;
}

cv::Mat generateImage(std::string text, int width=128, int height=128) {
    cv::Scalar white(255, 255, 255);
    cv::Scalar backgroundColor = white;
    // Создаем картинку на которую мы нанесем символ (пока что это просто белый фон)
    cv::Mat img(height, width, CV_8UC3, backgroundColor);

    // Выберем случайные параметры отрисовки текста - шрифт, размер, толщину, цвет
    int font = randFont();
    double fontScale = randFontScale();
    int thickness = randThickness();
    cv::Scalar color = randColor();

    // Узнаем размер текста в пикселях (если его нарисовать с указанными параметрами)
    int baseline = 0;
    cv::Size textPixelSize = cv::getTextSize(text, font, fontScale, thickness, &baseline);

    // Рисуем этот текст идеально в середине картинки
    // (для этого и нужно было узнать размер текста в пикселях - чтобы сделать отступ от середины картинки)
    // (ведь при рисовании мы указываем координаты левого нижнего угла текста)
    int xLeft = (width / 2) - (textPixelSize.width / 2);
    int yBottom = (height / 2) + (textPixelSize.height / 2);
    cv::Point coordsOfLeftBorromCorner(xLeft, yBottom);
    cv::putText(img, text, coordsOfLeftBorromCorner, font, fontScale, color, thickness);

    return img;
}

void generateAllLetters() {
    srand(239017); // фиксируем зерно генератора случайных чисел (чтобы картинки от раза к разу генерировались с одинаковыми шрифтами, размерами и т.п.)

    for (char letter = 'a'; letter <= 'z'; ++letter) {

        // Создаем папку для текущей буквы:
        std::string letterDir = LETTER_DIR_PATH + "/" + letter;
        std::filesystem::create_directory(letterDir);

        for (int sample = 1; sample <= NSAMPLES_PER_LETTER; ++sample) {
            std::string text = std::string("") + letter;
            cv::Mat img = generateImage(text);

            cv::blur(img, img, cv::Size(3, 3));

            std::string letterSamplePath = letterDir + "/" + std::to_string(sample) + ".png";
            cv::imwrite(letterSamplePath, img);
        }
    }
}

std::string detect(const cv::Mat& a) {
    std::string letterMin = "";
    double distMin = DBL_MAX;
    HoG hogA = buildHoG(a);
    for (char letterB = 'a'; letterB <= 'z'; ++letterB) {
        for(int i = 1; i <= NSAMPLES_PER_LETTER; i++){
            cv::Mat b = cv::imread(LETTER_DIR_PATH + "/" + letterB + "/" + std::to_string(i) + ".png");
            HoG hogB = buildHoG(b);
            double d = distance(hogA, hogB);
            if (d <= distMin) {
                distMin = d;
                letterMin = LETTER_DIR_PATH + "/" + letterB + "/" + std::to_string(i) + ".png";
            }
        }
    }
    return letterMin;
}

void finalExperiment(std::string name, std::string k) {
    cv::Mat original = cv::imread("lesson11/data/" + name + "/" + k + ".png");
    auto x = splitSymbols(original);
    std::vector<std::vector<std::pair<cv::Point, cv::Mat>>> vect = sort(x);
    int j = 0;
    for(std::vector<std::pair<cv::Point_<int>, cv::Mat>> ms : vect) {
        int i = 0;
        for(std::pair<cv::Point_<int>, cv::Mat> m : ms) {
            std::cout << detect(m.second) << " ";
            cv::imwrite(LETTER_DIR_PATH + "/res/" + std::to_string(j) + "-" + std::to_string(i) + ".png", m.second);
            i++;
        }
        j++;
    }
}



int main() {
    try {
        finalExperiment("text", "1");

//        std::vector<std::string> names;
//        names.emplace_back("alphabet");
//        names.emplace_back("line");
//        names.emplace_back("text");
//        for (auto & name : names) {
//            for (int j = 1; j <= 5; ++j) {
//                test(name, std::to_string(j));
//            }
//        }
//
//        test("alphabet", "3_gradient");



        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception! " << e.what() << std::endl;
        return 1;
    }
}
