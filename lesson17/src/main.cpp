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


bool isPixelEmpty(const cv::Vec3b& color) {
    return color == cv::Vec3b(0,0,0);
}

double distance(unsigned char r1, unsigned char g1, unsigned char b1, unsigned char r2, unsigned char g2, unsigned char b2){
    return sqrt(pow(r2-r1,2)+pow(g2-g1,2)+pow(b2-b1,2));
}

double distance(cv::Vec3b c1, cv::Vec3b c2){
    return distance(c1[0],c1[1],c1[2], c2[0], c2[1], c2[2]);
}

// Эта функция построит лабиринт - в каждом пикселе будет число которое говорит насколько длинное или короткое ребро выходит из пикселя
cv::Mat buildTheMaze(cv::Mat pano0, cv::Mat pano1) {
    rassert(!pano0.empty(), 324783479230019);
    rassert(!pano1.empty(), 324783479230020);
    rassert(pano0.type() == CV_8UC3, 3447928472389021);
    rassert(pano1.type() == CV_8UC3, 3447928472389022);
    rassert(pano0.cols == pano1.cols, 3295782390572300071);
    rassert(pano0.rows == pano1.rows, 3295782390572300072);

    const int MIN_PENALTY = 1;
    const int BIG_PENALTY = 100000;
    cv::Mat maze(pano0.rows, pano0.cols, CV_32SC1, cv::Scalar(0)); // создали лабиринт, размером с панораму, каждый пиксель - int

    for (int j = 0; j < pano0.rows; ++j) {
        for (int i = 0; i < pano0.cols; ++i) {
            int penalty;
            cv::Vec3b c1 = pano0.at<cv::Vec3b>(j, i);
            cv::Vec3b c2 = pano1.at<cv::Vec3b>(j, i);
            double d = distance(pano0.at<cv::Vec3b>(i, j), pano1.at<cv::Vec3b>(i, j));
            if(isPixelEmpty(c1) && isPixelEmpty(c2))
                penalty = BIG_PENALTY;
            else {
                penalty = d+MIN_PENALTY;
            }

            // BIG_PENALTY - если этот пиксель отсутствует в pano0 или в pano1
            // разница между цветами этого пикселя в pano0 и в pano1 (но не меньше MIN_PENALTY)

            maze.at<int>(j, i) = penalty;
        }
    }

    return maze;
}

struct Edge {
    int u, v; // номера вершин которые это ребро соединяет
    int w; // длина ребра (т.е. насколько длинный путь предстоит преодолеть переходя по этому ребру между вершинами)

    Edge(int u, int v, int w) : u(u), v(v), w(w)
    {}
};

int encodeVertex(int row, int column, int nrows, int ncolumns) {
    rassert(row < nrows, 348723894723980017);
    rassert(column < ncolumns, 347823974239870018);
    int vertexId = row * ncolumns + column;
    return vertexId;
}

// Эта биективная функция по номеру вершины говорит какой пиксель этой вершине соовтетствует (эта функция должна быть симметрична предыдущей!)
cv::Point2i decodeVertex(int vertexId, int nrows, int ncolumns) {
    int row = vertexId/ncolumns;
    int column = vertexId%ncolumns;

    // сверим что функция симметрично сработала:
    rassert(encodeVertex(row, column, nrows, ncolumns) == vertexId, 34782974923035);

    rassert(row < nrows, 34723894720027);
    rassert(column < ncolumns, 3824598237592030);
    return cv::Point2i(column, row);
}

// Скопируйте в эту функцию Дейкстру из позапрошлого задания - mainMaze.cpp
std::vector<cv::Point2i> findBestSeam(cv::Mat maze, cv::Point2i startPoint, cv::Point2i finishPoint) {
    rassert(!maze.empty(), 324783479230019);
    rassert(maze.type() == CV_32SC1, 3447928472389020);
    std::cout << "Maze resolution: " << maze.cols << "x" << maze.rows << std::endl;

    int nvertices = maze.cols * maze.rows;

    std::vector<std::vector<Edge>> edges_by_vertex(nvertices);
    for (int j = 0; j < maze.cols; ++j) {
        for (int i = 0; i < maze.rows; ++i) {
            int w = maze.at<int>(i, j);

            // TODO добавьте в edges_by_vertex ребро вправо и вверх с длинной w
            auto ai = encodeVertex(i,j,maze.rows,maze.cols);
            if(j < maze.cols-1) {
                cv::Vec3b color2 = maze.at<cv::Vec3b>(i,j+1);
                edges_by_vertex[ai].emplace_back(ai, encodeVertex(i,j+1, maze.rows, maze.cols), w);
            }
            if(i > 0) {
                cv::Vec3b color2 = maze.at<cv::Vec3b>(i-1,j);
                edges_by_vertex[ai].emplace_back(ai, encodeVertex(i-1,j, maze.rows, maze.cols), w);
            }
        }
    }

    int start = encodeVertex(startPoint.y, startPoint.x, maze.rows, maze.cols);
    int finish = encodeVertex(finishPoint.y, finishPoint.x, maze.rows, maze.cols);
    rassert(start >= 0 && start < nvertices, 348923840234200127);
    rassert(finish >= 0 && finish < nvertices, 348923840234200128);

    const int INF = std::numeric_limits<int>::max();

    std::vector<int> distances(nvertices, INF);
    std::vector<int> prev(nvertices, -1);
    std::set<std::pair<int, int>> q;        //w, u
    distances[start] = 0;
    prev[start] = start;
    q.emplace(0, start);
    while (!q.empty()){
        auto p = *q.begin();
        q.erase(*q.begin());
        if(distances[p.second] != p.first)
            continue;
        for(auto n : edges_by_vertex[p.second]){
            if(distances[n.v] > p.first+n.w){
                distances[n.v] = p.first+n.w;
                q.emplace(distances[n.v], n.v);
                prev[n.v] = p.second;
            }
        }
    }

    rassert(distances[finish] != INF, 3478289374239000163);
    std::vector<int> path;
    for (int i = finish; prev[i] != i; i = prev[i]) {
        path.emplace_back(i);
    }
    path.emplace_back(start);
    std::vector<cv::Point2i> pathPoints;
    for(auto it = path.rbegin(); it != path.rend(); it++) {
        cv::Point2i p = decodeVertex(*it, maze.rows, maze.cols);
        pathPoints.emplace_back(p);
    }
    return pathPoints;
}


void run(std::string caseName) {
    cv::Mat img0 = cv::imread("lesson16/data/" + caseName + "/0.png");
    cv::Mat img1 = cv::imread("lesson16/data/" + caseName + "/1.png");
    rassert(!img0.empty(), 324789374290018);
    rassert(!img1.empty(), 378957298420019);

    int downscale = 2; // уменьшим картинку в два раза столько раз сколько указано в этой переменной (итоговое уменьшение в 2^downscale раз)
    for (int i = 0; i < downscale; ++i) {
        cv::pyrDown(img0, img0); // уменьшаем картинку в два раза (по каждой из осей)
        cv::pyrDown(img1, img1); // уменьшаем картинку в два раза (по каждой из осей)
    }

    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    // Детектируем и описываем дескрипторы ключевых точек
    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    cv::Mat descriptors0, descriptors1;
    detector->detectAndCompute(img0, cv::noArray(), keypoints0, descriptors0);
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    std::cout << "Keypoints initially: " << keypoints0.size() << ", " << keypoints1.size() << "..." << std::endl;

    // Сопоставляем ключевые точки (находя для каждой точки из первой картинки - две ближайшие, т.е. две самые похожие, с правой картинки)
    std::vector<std::vector<cv::DMatch>> matches01;
    matcher->knnMatch(descriptors0, descriptors1, matches01, 2);
    rassert(keypoints0.size() == matches01.size(), 349723894200068);

    // Фильтруем сопоставления от шума - используя K-ratio тест
    std::vector<cv::Point2f> points0;
    std::vector<cv::Point2f> points1;
    for (int i = 0; i < keypoints0.size(); ++i) {
        int fromKeyPoint0 = matches01[i][0].queryIdx;
        int toKeyPoint1Best = matches01[i][0].trainIdx;
        float distanceBest = matches01[i][0].distance;
        rassert(fromKeyPoint0 == i, 348723974920074);
        rassert(toKeyPoint1Best < keypoints1.size(), 347832974820076);

        int toKeyPoint1SecondBest = matches01[i][1].trainIdx;
        float distanceSecondBest = matches01[i][1].distance;
        rassert(toKeyPoint1SecondBest < keypoints1.size(), 3482047920081);
        rassert(distanceBest <= distanceSecondBest, 34782374920082);

        // простой K-ratio тест, но могло иметь смысл добавить left-right check
        if (distanceBest < 0.7 * distanceSecondBest) {
            points0.push_back(keypoints0[i].pt);
            points1.push_back(keypoints1[toKeyPoint1Best].pt);
        }
    }
    rassert(points0.size() == points1.size(), 234723947289089);
    std::cout << "Matches after K-ratio test: " << points0.size() << std::endl;

    // Находим матрицу преобразования второй картинки в систему координат первой картинки
    cv::Mat H10 = cv::findHomography(points1, points0, cv::RANSAC, 3.0);
    rassert(H10.size() == cv::Size(3, 3), 3482937842900059); // см. документацию https://docs.opencv.org/4.5.1/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    // "Note that whenever an H matrix cannot be estimated, an empty one will be returned."

    // создаем папку в которую будем сохранять результаты - lesson17/resultsData/ИМЯ_НАБОРА/
    std::string resultsDir = "lesson17/resultsData/";
    if (!std::filesystem::exists(resultsDir)) { // если папка еще не создана
        std::filesystem::create_directory(resultsDir); // то создаем ее
    }
    resultsDir += caseName + "/";
    if (!std::filesystem::exists(resultsDir)) { // если папка еще не создана
        std::filesystem::create_directory(resultsDir); // то создаем ее
    }

    cv::imwrite(resultsDir + "0img0.jpg", img0);
    cv::imwrite(resultsDir + "1img1.jpg", img1);

    // находим куда переходят углы второй картинки
    std::vector<cv::Point2f> corners1(4);
    corners1[0] = cv::Point(0, 0); // верхний левый
    corners1[1] = cv::Point(img1.cols, 0); // верхний правый
    corners1[2] = cv::Point(img1.cols, img1.rows); // нижний правый
    corners1[3] = cv::Point(0, img1.rows); // нижний левый
    std::vector<cv::Point2f> corners10(4);
    perspectiveTransform(corners1, corners10, H10);

    // находим какой ширины и высоты наша панорама (как минимум - разрешение первой картинки, но еще нужно учесть куда перешли углы второй картинки)
    int max_x = img0.cols;
    int max_y = img0.rows;
    for (int i = 0; i < 4; ++i) {
        max_x = std::max(max_x, (int) corners10[i].x);
        max_y = std::max(max_y, (int) corners10[i].y);
    }
    int pano_rows = max_y;
    int pano_cols = max_x;

    // преобразуем обе картинки в пространство координат нашей искомой панорамы
    cv::Mat pano0(pano_rows, pano_cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat pano1(pano_rows, pano_cols, CV_8UC3, cv::Scalar(0, 0, 0));
    rassert(img0.type() == CV_8UC3, 3423890003123093);
    rassert(img1.type() == CV_8UC3, 3423890003123094);
    // вторую картинку просто натягиваем в соответствии с ранее найденной матрицей Гомографии
    cv::warpPerspective(img1, pano1, H10, pano1.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    // первую картинку надо положить без каких-то смещений, т.е. используя единичную матрицу:
    cv::Mat identity_matrix = cv::Mat::eye(3, 3, CV_64FC1);
    cv::warpPerspective(img0, pano0, identity_matrix, pano1.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    cv::imwrite(resultsDir + "2pano0.jpg", pano0);
    cv::imwrite(resultsDir + "3pano1.jpg", pano1);

    // давайте сделаем наивную панораму - наложим вторую картинку на первую:
    cv::Mat panoBothNaive = pano0.clone();
    cv::warpPerspective(img1, panoBothNaive, H10, panoBothNaive.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    cv::imwrite(resultsDir + "4panoBothNaive.jpg", panoBothNaive);

    cv::Mat maze = buildTheMaze(pano0, pano1);

    // найдем оптимальный шов разделяющий обе картинки (кратчайший путь в лабиринте)
    cv::Point2i start(0, pano_rows - 1); // из нижнего левого угла
    cv::Point2i finish(pano_cols - 1, 0); // в верхний правый угол
    std::cout << "Searching for optimal seam..." << std::endl;
    std::vector<cv::Point2i> seam = findBestSeam(maze, start, finish);
    for (const auto& pointOnSeam : seam) {
        panoBothNaive.at<cv::Vec3b>(pointOnSeam) = cv::Vec3b(0, 0, 255);
    }
    std::cout << "Length of optimal seam: " << seam.size() << " pixels" << std::endl;
    cv::imwrite(resultsDir + "5panoOptimalSeam.jpg", panoBothNaive);

    const unsigned char PIXEL_NO_DATA = 0; // черные без информации (не покрыты картинкой)
    const unsigned char PIXEL_IS_ON_SEAM = 255; // те что лежат на шве (через них первая картинка не перешагивает, но в конечном счете давайте на шве рисовать вторую картинку)
    const unsigned char PIXEL_FROM_PANO0 = 50; // те что покрыты первой картинкой (сверху слева от шва)
    const unsigned char PIXEL_FROM_PANO1 = 100; // те что покрыты второй картинкой (справа снизу от шва)
    cv::Mat sourceId(pano_rows, pano_cols, CV_8UC1, cv::Scalar(PIXEL_NO_DATA));
    for (const auto& pointOnSeam : seam) {
        sourceId.at<unsigned char>(pointOnSeam.y, pointOnSeam.x) = PIXEL_IS_ON_SEAM;
    }
    cv::Point2i leftUpCorner(0, 0);
    sourceId.at<unsigned char>(leftUpCorner.y, leftUpCorner.x) = PIXEL_FROM_PANO0;
    std::vector<cv::Point2i> curWave;
    curWave.push_back(leftUpCorner);

    while (!curWave.empty()) {
        std::vector<cv::Point2i> nextWave;
        for (auto p : curWave) {
            // кодируем сдвиг координат всех четырех соседей:
            //            слева (dx=-1, dy=0), сверху (dx=0, dy=-1), справа (dx=1, dy=0), снизу (dx=0, dy=1)
            int dxs[4] = {-1, 0, 1, 0};
            int dys[4] = {0, -1, 0, 1};

            for (int k = 0; k < 4; ++k) { // смотрим на четырех соседей
                int nx = p.x + dxs[k];
                int ny = p.y + dys[k];
                if(nx < 0 || ny < 0 || nx >= sourceId.cols || ny >= sourceId.rows || sourceId.at<unsigned char>(ny,nx) == PIXEL_FROM_PANO0 || isPixelEmpty(pano0.at<cv::Vec3b>(ny,nx)))
                    continue;
                if (sourceId.at<unsigned char>(ny,nx) != PIXEL_IS_ON_SEAM)
                    nextWave.emplace_back(cv::Point2i(nx,ny));
                sourceId.at<unsigned char>(ny, nx) = PIXEL_FROM_PANO0;
            }
        }
        curWave = nextWave;
    }
    for (int j = 0; j < pano_rows; ++j) {
        for (int i = 0; i < pano_cols; ++i) {
            if(sourceId.at<unsigned char>(j,i) != PIXEL_FROM_PANO0 && !isPixelEmpty(pano1.at<cv::Vec3b>(j,i)))
                sourceId.at<unsigned char>(j,i) = PIXEL_FROM_PANO1;
        }
    }
    cv::imwrite(resultsDir + "6sourceId.jpg", sourceId);

    cv::Mat newPano(pano_rows, pano_cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int j = 0; j < pano_rows; ++j) {
        for (int i = 0; i < pano_cols; ++i) {
            switch (sourceId.at<unsigned char>(j,i)) {
                case PIXEL_FROM_PANO0:
                    newPano.at<cv::Vec3b>(j,i) = pano0.at<cv::Vec3b>(j,i);
                    break;
                case PIXEL_FROM_PANO1:
                    newPano.at<cv::Vec3b>(j,i) = pano1.at<cv::Vec3b>(j,i);
                    break;
            }
        }
    }
    cv::imwrite(resultsDir + "7newPano.jpg", newPano);
}


int main() {
    try {
        run("1_hanging");
        run("2_hiking");
        run("3_aero");
//        run("4_your_data"); // TODO сфотографируйте что-нибудь сами при этом на второй картинке что-то изменив, см. иллюстрацию на сайте

        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception! " << e.what() << std::endl;
        return 1;
    }
}