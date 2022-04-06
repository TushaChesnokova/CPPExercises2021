#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <set>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <memory>

#include <libutils/rasserts.h>
#include <libutils/fast_random.h>

bool isPixelMasked(cv::Mat mask, int j, int i) {
    rassert(j >= 0 && j < mask.rows, 372489347280017);
    rassert(i >= 0 && i < mask.cols, 372489347280018);
    rassert(mask.type() == CV_8UC3, 2348732984792380019);

    // TODO проверьте белый ли пиксель
    return mask.at<cv::Vec3b>(j, i) == cv::Vec3b(255, 255, 255);
}


double estimateQuality(cv::Mat mask, cv::Mat image, int j, int i, int ny, int nx, int height, int width) {
    double distance = 0;
    for (int a = -height / 2; a <= height / 2; a++) {
        for (int b = -width / 2; b <= width / 2; b++) {
            if (a + j < 0 || a + j >= image.rows || b + i < 0 || b + i >= image.cols || a + ny < 0 ||
                a + ny >= image.rows || b + nx < 0 || b + nx >= image.cols) {
                distance = distance + 99999999;
            } else if (isPixelMasked(mask, a + ny, b + nx)) {
                distance = distance + 99999999;
            } else {
                cv::Vec3b d = image.at<cv::Vec3b>(j + a, i + b) - image.at<cv::Vec3b>(a + ny, b + nx);
                distance = distance + cv::norm(d);
            }
        }
    }
    return distance;
}

void run(int caseNumber, std::string caseName) {
    std::cout << "_________Case #" << caseNumber << ": " << caseName << "_________" << std::endl;

    cv::Mat original = cv::imread(
            "lesson18/data/" + std::to_string(caseNumber) + "_" + caseName + "/" + std::to_string(caseNumber) +
            "_original.jpg");
    cv::Mat mask = cv::imread(
            "lesson18/data/" + std::to_string(caseNumber) + "_" + caseName + "/" + std::to_string(caseNumber) +
            "_mask.png");
    rassert(!original.empty(), 324789374290018);
    rassert(!mask.empty(), 378957298420019);
    rassert(original.size == mask.size, 282389216392)

    // TODO напишите rassert сверяющий разрешение картинки и маски
    // TODO выведите в консоль это разрешение картинки
    std::cout << "Image resolution: " << original.size << std::endl;

    std::string resultsDir = "lesson18/resultsData/";
    if (!std::filesystem::exists(resultsDir)) {
        std::filesystem::create_directory(resultsDir);
    }
    resultsDir += std::to_string(caseNumber) + "_" + caseName + "/";
    if (!std::filesystem::exists(resultsDir)) {
        std::filesystem::create_directory(resultsDir);
    }

    // сохраняем в папку с результатами оригинальную картинку и маску
    cv::imwrite(resultsDir + "0original.png", original);
    cv::imwrite(resultsDir + "1mask.png", mask);

    // TODO замените белым цветом все пиксели в оригинальной картинке которые покрыты маской
    // TODO сохраните в папку с результатами то что получилось под названием "2_original_cleaned.png"
    // TODO посчитайте и выведите число отмаскированных пикселей (числом и в процентах) - в таком формате:
    // Number of masked pixels: 7899/544850 = 1%

    int numberOfMasked = 0;
    cv::Mat originalCleaned = original.clone();
    for (int j = 0; j < original.rows; j++) {
        for (int i = 0; i < original.cols; i++) {
            if (isPixelMasked(mask, j, i)) {
                originalCleaned.at<cv::Vec3b>(j, i) = cv::Vec3b(255, 255, 255);
                numberOfMasked++;
            }
        }
    }
    cv::imwrite(resultsDir + "2_original_cleaned.png", originalCleaned);

    {
        int numberOfPixels = original.cols * original.rows;
        int percentage = numberOfMasked * 100 / numberOfPixels;
        std::cout
                << "Number of masked pixels: " + std::to_string(numberOfMasked) + "/" + std::to_string(numberOfPixels) +
                   " = " + std::to_string(percentage) + "%" + '\n';
    }


    FastRandom random(32542341); // этот объект поможет вам генерировать случайные гипотезы

    // TODO 10 создайте картинку хранящую относительные смещения - откуда брать донора для заплатки, см. подсказки про то как с нею работать на сайте
    // TODO 11 во всех отмаскированных пикселях: заполните эту картинку с относительными смещениями - случайными смещениями (но чтобы они и их окрестность 5х5 не выходила за пределы картинки)
    // TODO 12 во всех отмаскированных пикселях: замените цвет пиксела А на цвет пикселя Б на который указывает относительное смещение пикселя А
    // TODO 13 сохраните получившуюся картинку на диск
    // TODO 14 выполняйте эти шаги 11-13 много раз, например 1000 раз (оберните просто в цикл, сохраняйте картинку на диск только на каждой десятой или сотой итерации)
    // TODO 15 теперь давайте заменять значение относительного смещения на новой только если новая случайная гипотеза - лучше старой, добавьте оценку "насколько смещенный патч 5х5 похож на патч вокруг пикселя если их наложить"

    cv::Mat shifts(original.rows, original.cols,
                   CV_32SC2);
    cv::Mat image = original;
    for (int n = 0; n <= 100; n++) {
        for (int j = 0; j < image.rows; j++) {
            for (int i = 0; i < image.cols; i++) {
                if (!isPixelMasked(mask, j, i))
                    continue; // пропускаем т.к. его менять не надо
                cv::Vec2i dxy = shifts.at<cv::Vec2i>(j, i);
                cv::Point donor = cv::Point(i + dxy[1], j + dxy[0]);
//                 int (nx, ny) = (i + dxy.x, j + dxy.y); // ЭТО НЕ КОРРЕКТНЫЙ КОД, но он иллюстрирует как рассчитать координаты пикселя-донора из которого мы хотим брать цвет
//                 currentQuality = estimateQuality(image, j, i, ny, nx, 5, 5); // эта функция (создайте ее) считает насколько похож квадрат 5х5 приложенный центром к (i, j)
//                                                                                                                                         // на квадрат 5х5 приложенный центром к (nx, ny)
                int height = 5;
                int width = 5;
                double currentQuality = estimateQuality(mask, image, j, i, donor.y, donor.x, height, width);


                cv::Point randomDonor;
                bool goodPoint = false;
                while (!goodPoint) {
                    goodPoint = true;
                    randomDonor = cv::Point(random.next(3, image.rows - 2), random.next(3, image.cols - 2));
                    for (int a = -height / 2; a <= height / 2; a++) {
                        for (int b = -width / 2; b <= width / 2; b++) {
                            if (mask.at<cv::Vec3b>(randomDonor.y + a, randomDonor.x + b) == cv::Vec3b(255, 255, 255))
                                goodPoint = false;
                        }
                    }
                }
                double randomQuality = estimateQuality(mask, image, j, i, randomDonor.y, randomDonor.x, 5, 5);
                int rx = i - randomDonor.x;
                int ry = j - randomDonor.y;

                //         int (rx, ry) = random.... // создаем случайное смещение относительно нашего пикселя, воспользуйтесь функцией random.next(...);
                //                                      (окрестность вокруг пикселя на который укажет смещение - не должна выходить за пределы картинки и не должна быть отмаскирована)
                //         randomQuality = estimateQuality(image, j, i, j+ry, i+rx, 5, 5); // оцениваем насколько похоже будет если мы приложим эту случайную гипотезу которую только что выбрали
                //

                if (randomQuality < currentQuality) {
                    shifts.at<cv::Vec2i>(j, i) = (ry, rx);
                    image.at<cv::Vec3b>(j, i) = image.at<cv::Vec3b>(randomDonor.y, randomDonor.x);
                } else {
                    image.at<cv::Vec3b>(j, i) = image.at<cv::Vec3b>(donor.y, donor.x);
                }

            }
        }
        if (n % 10 == 0) {
            cv::imwrite(resultsDir + std::to_string(n / 100 + 1) + "_cleaned.png", image);
        }
    }

}


int main() {
    try {
        run(1, "mic");
        // TODO протестируйте остальные случаи:
//        run(2, "flowers");
//        run(3, "baloons");
//        run(4, "brickwall");
//        run(5, "old_photo");
//        run(6, "your_data"); // TODO придумайте свой случай для тестирования (рекомендуется не очень большое разрешение, например 300х300)

        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception! " << e.what() << std::endl;
        return 1;
    }
}