#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include <libutils/rasserts.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>


struct Edge {
    int u, v; // номера вершин которые это ребро соединяет
    int w; // длина ребра (т.е. насколько длинный путь предстоит преодолеть переходя по этому ребру между вершинами)

    Edge(int u, int v, int w) : u(u), v(v), w(w)
    {}
};

// Эта биективная функция по координате пикселя (строчка и столбик) + размерам картинки = выдает номер вершины
int encodeVertex(int row, int column, int nrows, int ncolumns) {
    rassert(row < nrows, 348723894723980017);
    rassert(column < ncolumns, 347823974239870018);
    int vertexId = row * ncolumns + column;
    return vertexId;
}

// Эта биективная функция по номеру вершины говорит какой пиксель этой вершине соовтетствует (эта функция должна быть симметрична предыдущей!)
cv::Point2i decodeVertex(int vertexId, int nrows, int ncolumns) {

    // TODO: придумайте как найти номер строки и столбика пикселю по номеру вершины (просто поймите предыдущую функцию и эта функция не будет казаться сложной)
    int column = vertexId%ncolumns;
    int row = vertexId/ncolumns;

    // сверим что функция симметрично сработала:
    rassert(encodeVertex(row, column, nrows, ncolumns) == vertexId, 34782974923035);

    rassert(row < nrows, 34723894720027);
    rassert(column < ncolumns, 3824598237592030);
    return cv::Point2i(column, row);
}
int distance(unsigned char r1, unsigned char g1, unsigned char b1, unsigned char r2, unsigned char g2, unsigned char b2){
    return round(sqrt(pow(r2-r1,2)+pow(g2-g1,2)+pow(b2-b1,2)));
}

void run(int mazeNumber) {
    cv::Mat maze = cv::imread("lesson15/data/mazesImages/maze" + std::to_string(mazeNumber) + ".png");
    rassert(!maze.empty(), 324783479230019);
    rassert(maze.type() == CV_8UC3, 3447928472389020);
    std::cout << "Maze resolution: " << maze.cols << "x" << maze.rows << std::endl;

    int nvertices = maze.cols*maze.rows; // TODO

    std::vector<std::vector<Edge>> edges_by_vertex(nvertices);
    for (int i = 0; i < maze.rows; ++i) {
        for (int j = 0; j < maze.cols; ++j) {
            cv::Vec3b color = maze.at<cv::Vec3b>(i,j);
            unsigned char blue = color[0];
            unsigned char green = color[1];
            unsigned char red = color[2];

            auto ai = encodeVertex(i,j,maze.rows,maze.cols);
            if(i > 0) {
                cv::Vec3b color2 = maze.at<cv::Vec3b>(i-1,j);
                edges_by_vertex[ai].emplace_back(ai, encodeVertex(i-1,j, maze.rows, maze.cols),
                                                 distance(color2[0], color2[1], color2[2], red, green, blue)+1);
            }
            if(i < maze.rows-1) {
                cv::Vec3b color2 = maze.at<cv::Vec3b>(i+1,j);
                edges_by_vertex[ai].emplace_back(ai, encodeVertex(i+1,j, maze.rows, maze.cols),
                                                 distance(color2[0], color2[1], color2[2], red, green, blue)+1);
            }
            if(j > 0) {
                cv::Vec3b color2 = maze.at<cv::Vec3b>(i,j-1);
                edges_by_vertex[ai].emplace_back(ai, encodeVertex(i,j-1, maze.rows, maze.cols),
                                                 distance(color2[0], color2[1], color2[2], red, green, blue)+1);
            }
            if(j < maze.cols-1) {
                cv::Vec3b color2 = maze.at<cv::Vec3b>(i,j+1);
                edges_by_vertex[ai].emplace_back(ai, encodeVertex(i,j+1, maze.rows, maze.cols),
                                                 distance(color2[0], color2[1], color2[2], red, green, blue)+1);
            }
        }
    }

    int start, finish;
    if (mazeNumber >= 1 && mazeNumber <= 3) { // Первые три лабиринта очень похожи но кое чем отличаются...
        start = encodeVertex(300, 300, maze.rows, maze.cols);
        finish = encodeVertex(0, 305, maze.rows, maze.cols);
    } else if (mazeNumber == 4) {
        start = encodeVertex(154, 312, maze.rows, maze.cols);
        finish = encodeVertex(477, 312, maze.rows, maze.cols);
    } else if (mazeNumber == 5) { // Лабиринт в большом разрешении, добровольный (на случай если вы реализовали быструю Дейкстру с приоритетной очередью)
        start = encodeVertex(1200, 1200, maze.rows, maze.cols);
        finish = encodeVertex(1200, 1200, maze.rows, maze.cols);
    } else {
        rassert(false, 324289347238920081);
    }

    const int INF = std::numeric_limits<int>::max();

    cv::Mat window = maze.clone(); // на этой картинке будем визуализировать до куда сейчас дошла прокладка маршрута

    std::vector<int> distances(nvertices, INF);
    std::vector <bool> grizly(nvertices, false);
    std::vector <int> panda(nvertices,start);
    distances[start]=0;
    while (true) {
        int minv = INF;
        for(int i=0; i<nvertices; i++){
            if ((minv==INF||distances[i]<distances[minv])&&!grizly[i]) minv=i;
        }
        if (minv==INF) break;
        for (int i=0; i<edges_by_vertex[minv].size(); i++){
            Edge a = edges_by_vertex[minv][i];
            if (distances[a.v]>distances[a.u]+a.w) {
                distances[a.v]=distances[a.u]+a.w;
                panda[a.v]=a.u;
            }
        }
        grizly[minv]=true;
    }
    std::vector <int> path;

    if (distances[finish] != INF) {
        std::vector<int> path;
        for (int i = finish; panda[i] != i; i = panda[i]) {
            path.emplace_back(i);
        }
        path.emplace_back(start);
        for(auto it = path.rbegin(); it != path.rend(); it++) {
            cv::Point2i p = decodeVertex(*it, maze.rows, maze.cols);
            window.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(0, 255, 0);
            cv::imshow("Maze", window);
            cv::waitKey(1);
        }
        std::cout << std::endl;
    } else {
        std::cout << -1 << std::endl;
    }

    std::cout << "Finished!" << std::endl;

    // Показываем результат пока пользователь не насладиться до конца и не нажмет Escape
    while (cv::waitKey(10) != 27) {
        cv::imshow("Maze", window);
    }
}

int main() {
    try {
        int mazeNumber = 1;
        run(mazeNumber);

        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception! " << e.what() << std::endl;
        return 1;
    }
}
