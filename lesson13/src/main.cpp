#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <filesystem>

#include <libutils/rasserts.h>


std::vector<cv::Point2f> filterPoints(std::vector<cv::Point2f> points, std::vector<unsigned char> matchIsGood) {
    rassert(points.size() == matchIsGood.size(), 234827348927016);

    std::vector<cv::Point2f> goodPoints;
    for (int i = 0; i < matchIsGood.size(); ++i) {
        if (matchIsGood[i]) {
            goodPoints.push_back(points[i]);
        }
    }
    return goodPoints;
}


void test1() {
    std::string caseName = "1_box2";

    std::string path = "lesson13/data/" + caseName + "/";
    std::string results = "lesson13/resultsData/" + caseName + "/";

    if (!std::filesystem::exists(results)) {    
        std::filesystem::create_directory(results);    
    }

    cv::Mat img0 = cv::imread(path + "box0.png");
    cv::Mat img1 = cv::imread(path + "box1.png");
    cv::Mat img1_alternative = cv::imread(path + "box1_nesquik.png");
    rassert(!img0.empty() && !img1.empty() && !img1_alternative.empty(), 2389827851080019);
    rassert(img0.type() == CV_8UC3, 2389827851080020);
    rassert(img1.type() == CV_8UC3, 2389827851080021);
    rassert(img1_alternative.type() == CV_8UC3, 2389827851080023);

       
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();

       
    std::vector<cv::KeyPoint> keypoints0, keypoints1;    
    cv::Mat descriptors0, descriptors1;    
    std::cout << "Detecting SIFT keypoints and describing them (computing their descriptors)..." << std::endl;
    detector->detectAndCompute(img0, cv::noArray(), keypoints0, descriptors0);
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    std::cout << "SIFT keypoints detected and described: " << keypoints0.size() << " and " << keypoints1.size() << std::endl;    

    {
           
        cv::Mat img0withKeypoints, img1withKeypoints;
        cv::drawKeypoints(img0, keypoints0, img0withKeypoints);
        cv::drawKeypoints(img1, keypoints1, img1withKeypoints);
        cv::imwrite(results + "01keypoints0.jpg", img0withKeypoints);
        cv::imwrite(results + "01keypoints1.jpg", img1withKeypoints);
    }

       
       
    std::vector<std::vector<cv::DMatch>> matches01;
    std::cout << "Matching " << keypoints0.size() << " points with " << keypoints1.size() << "..." << std::endl;    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher->knnMatch(descriptors0, descriptors1, matches01, 2);    
    std::cout << "matching done" << std::endl;
       
    rassert(keypoints0.size() == matches01.size(), 234728972980049);
    for (int i = 0; i < matches01.size(); ++i) {
        rassert(matches01[i].size() == 2, 3427890347902051);
        rassert(matches01[i][0].queryIdx == i, 237812974128941);    
        rassert(matches01[i][1].queryIdx == i, 237812974128942);    

        rassert(matches01[i][0].trainIdx < keypoints1.size(), 237812974128943);    
        rassert(matches01[i][1].trainIdx < keypoints1.size(), 237812974128943);    

        rassert(matches01[i][0].distance <= matches01[i][1].distance, 328493778);    
    }

    std::vector<double> distances01;
    for (auto & i : matches01) {
        distances01.push_back(i[0].distance);
    }
    std::sort(distances01.begin(), distances01.end());
    std::cout << "matches01 distances min/median/max: " << distances01[0] << "/" << distances01[distances01.size() / 2] << "/" << distances01[distances01.size() - 1] << std::endl;
    for (int k = 0; k < 2; ++k) {
        std::vector<cv::DMatch> matchesK;
        for (int i = 0; i < matches01.size(); ++i) {
            matchesK.push_back(matches01[i][k]);
        }
           
        cv::Mat imgWithMatches;
        cv::drawMatches(img0, keypoints0, img1, keypoints1, matchesK, imgWithMatches);
        cv::imwrite(results + "02matches01_k" + std::to_string(k) + ".jpg", imgWithMatches);
    }

       
    std::vector<std::vector<cv::DMatch>> matches10;
    std::cout << "Matching " << keypoints1.size() << " points with " << keypoints0.size() << "..." << std::endl;
    matcher->knnMatch(descriptors1, descriptors0, matches10, 2);
    std::cout << "matching done" << std::endl;
    rassert(keypoints1.size() == matches10.size(), 234728972980049);
    for (int i = 0; i < matches10.size(); ++i) {
        rassert(matches10[i].size() == 2, 3427890347902051);
        rassert(matches10[i][0].queryIdx == i, 237812974128941);
        rassert(matches10[i][1].queryIdx == i, 237812974128942);

        rassert(matches10[i][0].trainIdx < keypoints0.size(), 237812974128943);
        rassert(matches10[i][1].trainIdx < keypoints0.size(), 237812974128943);

        rassert(matches10[i][0].distance <= matches10[i][1].distance, 328493778);
    }

    std::vector<double> distances10;
    for (auto & i : matches10) {
        distances10.push_back(i[0].distance);
    }
    std::sort(distances10.begin(), distances10.end());
    std::cout << "matches10 distances min/median/max: " << distances10[0] << "/" << distances10[distances10.size() / 2] << "/" << distances10[distances10.size() - 1] << std::endl;
    for (int k = 0; k < 2; ++k) {
        std::vector<cv::DMatch> matchesK;
        for (int i = 0; i < matches10.size(); ++i) {
            matchesK.push_back(matches10[i][k]);
        }
           
        cv::Mat imgWithMatches;
        cv::drawMatches(img1, keypoints1, img0, keypoints0, matchesK, imgWithMatches);
        cv::imwrite(results + "02matches10_k" + std::to_string(k) + ".jpg", imgWithMatches);
    }

       
    std::cout << "Filtering matches..." << std::endl;
    std::vector<cv::Point2f> points0, points1;    
    std::vector<unsigned char> matchIsGood;    
    int nmatchesGood = 0;    
    for (int i = 0; i < keypoints0.size(); ++i) {
        cv::DMatch match = matches01[i][0];
        rassert(match.queryIdx == i, 234782749278097);    
        int j = match.trainIdx;    
        rassert(j < keypoints1.size(), 38472957238099);    

        points0.push_back(keypoints0[i].pt);
        points1.push_back(keypoints1[j].pt);

        bool isOk = true;

        if (match.distance > distances01[distances01.size()/2]) {
            isOk = false;
        }

        double k = 0.7;
        cv::DMatch match2 = matches01[i][1];
        if (match.distance > k*match2.distance) {
            isOk = false;
        }

        cv::DMatch match01 = match;
        cv::DMatch match10 = matches10[match01.trainIdx][0];
        if (match10.trainIdx != i) {
            isOk = false;
        }


        if (isOk) {
            ++nmatchesGood;
            matchIsGood.push_back(true);
        } else {
            matchIsGood.push_back(false);
        }
    }
    rassert(points0.size() == points1.size(), 3497282579850108);
    rassert(points0.size() == matchIsGood.size(), 3497282579850109);
    std::cout << nmatchesGood << "/" << points0.size() << " good matches left" << std::endl;

    {
        std::vector<cv::DMatch> goodMatches;
        for (int i = 0; i < matches01.size(); ++i) {
            cv::DMatch match = matches01[i][0];
            rassert(match.queryIdx == i, 2347982739280182);
            rassert(match.trainIdx < points1.size(), 2347982739280182);
            if (!matchIsGood[i])
                continue;

            goodMatches.push_back(match);
        }
        cv::Mat imgWithMatches;
           
        cv::drawMatches(img0, keypoints0, img1, keypoints1, goodMatches, imgWithMatches);
        cv::imwrite(results + "04goodMatches01.jpg", imgWithMatches);
    }

       
    std::cout << "Finding Homography matrix from image0 to image1..." << std::endl;
    const double ransacReprojThreshold = 3.0;
    std::vector<cv::Point2f> pointsGood0 = filterPoints(points0, matchIsGood);
    std::vector<cv::Point2f> pointsGood1 = filterPoints(points1, matchIsGood);
    std::vector<unsigned char> inliersMask;    
       
    cv::Mat H01 = cv::findHomography(pointsGood0, pointsGood1, cv::RANSAC, ransacReprojThreshold, inliersMask);
       
    std::cout << "homography matrix found:" << std::endl;
    std::cout << H01 << std::endl;
    {
        std::vector<cv::DMatch> inliersMatches;
        std::vector<cv::KeyPoint> inlierKeypoints0, inlierKeypoints1;

        int ninliers = 0;
        for (int i = 0; i < inliersMask.size(); ++i) {
            if (inliersMask[i]) {
                ++ninliers;
                cv::DMatch match;
                match.queryIdx = inlierKeypoints0.size();
                match.trainIdx = inlierKeypoints1.size();
                inliersMatches.push_back(match);
                inlierKeypoints0.push_back(cv::KeyPoint(pointsGood0[i], 3.0));
                inlierKeypoints1.push_back(cv::KeyPoint(pointsGood1[i], 3.0));
            }
        }

           
        std::cout << "(with " << ninliers << "/" << nmatchesGood << " inliers matches)" << std::endl;
        cv::Mat imgWithInliersMatches;
        cv::drawMatches(img0, inlierKeypoints0, img1, inlierKeypoints1, inliersMatches, imgWithInliersMatches);
        cv::imwrite(results + "05inliersMatches01.jpg", imgWithInliersMatches);
    }

    cv::imwrite(results + "06img1.jpg", img1);    

    cv::Mat img0to1;
    cv::warpPerspective(img0, img0to1, H01, img1.size());    
    cv::imwrite(results + "07img0to1.jpg", img0to1);    




    cv::imwrite(results + "08img0.jpg", img0);    

    cv::Mat H10 = H01.inv();    
    cv::Mat img1to0;
    cv::warpPerspective(img1,img1to0, H10, img0.size());
    cv::imwrite(results + "09img1to0.jpg", img1to0);

    img1to0 = img0.clone();    
    cv::warpPerspective(img1, img1to0, H10, img1to0.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    cv::imwrite(results + "10img0with1to0.jpg", img1to0);

    img1to0 = img0.clone();
    cv::warpPerspective(img1_alternative, img1to0, H10, img1to0.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);    
    cv::imwrite(results + "11img0withNesquik.jpg", img1to0);
}

void test2() {
    std::string caseName = "2_hiking2";

       
       

    std::string path = "lesson13/data/" + caseName + "/";
    std::string results = "lesson13/resultsData/" + caseName + "/";

    if (!std::filesystem::exists(results)) {    
        std::filesystem::create_directory(results);    
    }

    cv::Mat img0 = cv::imread(path + "hiking0.png");
    cv::Mat img1 = cv::imread(path + "hiking1.png");
    rassert(!img0.empty() && !img1.empty(), 2389827851080019);
    rassert(img0.type() == CV_8UC3, 2389827851080020);
    rassert(img1.type() == CV_8UC3, 2389827851080021);

       
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();

       
    std::vector<cv::KeyPoint> keypoints0, keypoints1;    
    cv::Mat descriptors0, descriptors1;    
    std::cout << "Detecting SIFT keypoints and describing them (computing their descriptors)..." << std::endl;
    detector->detectAndCompute(img0, cv::noArray(), keypoints0, descriptors0);
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    std::cout << "SIFT keypoints detected and described: " << keypoints0.size() << " and " << keypoints1.size() << std::endl;    

    {
           
        cv::Mat img0withKeypoints, img1withKeypoints;
        cv::drawKeypoints(img0, keypoints0, img0withKeypoints);
        cv::drawKeypoints(img1, keypoints1, img1withKeypoints);
        cv::imwrite(results + "01keypoints0.jpg", img0withKeypoints);
        cv::imwrite(results + "01keypoints1.jpg", img1withKeypoints);
    }

       
       
    std::vector<std::vector<cv::DMatch>> matches01;
    std::cout << "Matching " << keypoints0.size() << " points with " << keypoints1.size() << "..." << std::endl;    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher->knnMatch(descriptors0, descriptors1, matches01, 2);    
    std::cout << "matching done" << std::endl;
       
    rassert(keypoints0.size() == matches01.size(), 234728972980049);
    for (int i = 0; i < matches01.size(); ++i) {
        rassert(matches01[i].size() == 2, 3427890347902051);
        rassert(matches01[i][0].queryIdx == i, 237812974128941);    
        rassert(matches01[i][1].queryIdx == i, 237812974128942);    

        rassert(matches01[i][0].trainIdx < keypoints1.size(), 237812974128943);    
        rassert(matches01[i][1].trainIdx < keypoints1.size(), 237812974128943);    

        rassert(matches01[i][0].distance <= matches01[i][1].distance, 328493778);    
    }

    std::vector<double> distances01;
    for (auto & i : matches01) {
        distances01.push_back(i[0].distance);
    }
    std::sort(distances01.begin(), distances01.end());
    std::cout << "matches01 distances min/median/max: " << distances01[0] << "/" << distances01[distances01.size() / 2] << "/" << distances01[distances01.size() - 1] << std::endl;
    for (int k = 0; k < 2; ++k) {
        std::vector<cv::DMatch> matchesK;
        for (int i = 0; i < matches01.size(); ++i) {
            matchesK.push_back(matches01[i][k]);
        }
           
        cv::Mat imgWithMatches;
        cv::drawMatches(img0, keypoints0, img1, keypoints1, matchesK, imgWithMatches);
        cv::imwrite(results + "02matches01_k" + std::to_string(k) + ".jpg", imgWithMatches);
    }

       
    std::vector<std::vector<cv::DMatch>> matches10;
    std::cout << "Matching " << keypoints1.size() << " points with " << keypoints0.size() << "..." << std::endl;
    matcher->knnMatch(descriptors1, descriptors0, matches10, 2);
    std::cout << "matching done" << std::endl;
    rassert(keypoints1.size() == matches10.size(), 234728972980049);
    for (int i = 0; i < matches10.size(); ++i) {
        rassert(matches10[i].size() == 2, 3427890347902051);
        rassert(matches10[i][0].queryIdx == i, 237812974128941);
        rassert(matches10[i][1].queryIdx == i, 237812974128942);

        rassert(matches10[i][0].trainIdx < keypoints0.size(), 237812974128943);
        rassert(matches10[i][1].trainIdx < keypoints0.size(), 237812974128943);

        rassert(matches10[i][0].distance <= matches10[i][1].distance, 328493778);
    }

    std::vector<double> distances10;
    for (auto & i : matches10) {
        distances10.push_back(i[0].distance);
    }
    std::sort(distances10.begin(), distances10.end());
    std::cout << "matches10 distances min/median/max: " << distances10[0] << "/" << distances10[distances10.size() / 2] << "/" << distances10[distances10.size() - 1] << std::endl;
    for (int k = 0; k < 2; ++k) {
        std::vector<cv::DMatch> matchesK;
        for (int i = 0; i < matches10.size(); ++i) {
            matchesK.push_back(matches10[i][k]);
        }
        cv::Mat imgWithMatches;
        cv::drawMatches(img1, keypoints1, img0, keypoints0, matchesK, imgWithMatches);
        cv::imwrite(results + "02matches10_k" + std::to_string(k) + ".jpg", imgWithMatches);
    }

    std::cout << "Filtering matches..." << std::endl;
    std::vector<cv::Point2f> points0, points1;
    std::vector<unsigned char> matchIsGood;
    int nmatchesGood = 0;
    for (int i = 0; i < keypoints0.size(); ++i) {
        cv::DMatch match = matches01[i][0];
        rassert(match.queryIdx == i, 234782749278097);    
        int j = match.trainIdx;    
        rassert(j < keypoints1.size(), 38472957238099);    

        points0.push_back(keypoints0[i].pt);
        points1.push_back(keypoints1[j].pt);

        bool isOk = true;


        if (match.distance > distances01[distances01.size()/2]) {
            isOk = false;
        }


        double k = 0.7;
        cv::DMatch match2 = matches01[i][1];
        if (match.distance > k*match2.distance) {
            isOk = false;
        }

        cv::DMatch match01 = match;
        cv::DMatch match10 = matches10[match01.trainIdx][0];
        if (match10.trainIdx != i) {
            isOk = false;
        }

           
           
           
           
           
           


        if (isOk) {
            ++nmatchesGood;
            matchIsGood.push_back(true);
        } else {
            matchIsGood.push_back(false);
        }
    }
    rassert(points0.size() == points1.size(), 3497282579850108);
    rassert(points0.size() == matchIsGood.size(), 3497282579850109);
    std::cout << nmatchesGood << "/" << points0.size() << " good matches left" << std::endl;

    {
        std::vector<cv::DMatch> goodMatches;
        for (int i = 0; i < matches01.size(); ++i) {
            cv::DMatch match = matches01[i][0];
            rassert(match.queryIdx == i, 2347982739280182);
            rassert(match.trainIdx < points1.size(), 2347982739280182);
            if (!matchIsGood[i])
                continue;

            goodMatches.push_back(match);
        }
        cv::Mat imgWithMatches;
           
        cv::drawMatches(img0, keypoints0, img1, keypoints1, goodMatches, imgWithMatches);
        cv::imwrite(results + "04goodMatches01.jpg", imgWithMatches);
    }

       
    std::cout << "Finding Homography matrix from image0 to image1..." << std::endl;
    const double ransacReprojThreshold = 3.0;
    std::vector<cv::Point2f> pointsGood0 = filterPoints(points0, matchIsGood);
    std::vector<cv::Point2f> pointsGood1 = filterPoints(points1, matchIsGood);
    std::vector<unsigned char> inliersMask;    
       
    cv::Mat H01 = cv::findHomography(pointsGood0, pointsGood1, cv::RANSAC, ransacReprojThreshold, inliersMask);
       
    std::cout << "homography matrix found:" << std::endl;
    std::cout << H01 << std::endl;
    {
        std::vector<cv::DMatch> inliersMatches;
        std::vector<cv::KeyPoint> inlierKeypoints0, inlierKeypoints1;

        int ninliers = 0;
        for (int i = 0; i < inliersMask.size(); ++i) {
            if (inliersMask[i]) {
                ++ninliers;
                cv::DMatch match;
                match.queryIdx = inlierKeypoints0.size();
                match.trainIdx = inlierKeypoints1.size();
                inliersMatches.push_back(match);
                inlierKeypoints0.push_back(cv::KeyPoint(pointsGood0[i], 3.0));
                inlierKeypoints1.push_back(cv::KeyPoint(pointsGood1[i], 3.0));
            }
        }

           
        std::cout << "(with " << ninliers << "/" << nmatchesGood << " inliers matches)" << std::endl;
        cv::Mat imgWithInliersMatches;
        cv::drawMatches(img0, inlierKeypoints0, img1, inlierKeypoints1, inliersMatches, imgWithInliersMatches);
        cv::imwrite(results + "05inliersMatches01.jpg", imgWithInliersMatches);
    }


    cv::imwrite(results + "08img0.jpg", img0);    

    cv::Mat H10 = H01.inv();    
    cv::Mat img1to0;
    cv::warpPerspective(img1,img1to0, H10, img0.size());
    cv::imwrite(results + "09img1to0.jpg", img1to0);

    img1to0 = img0.clone();    
    cv::warpPerspective(img1, img1to0, H10, img1to0.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    cv::imwrite(results + "10img0with1to0.jpg", img1to0);
}

void test3() {
    std::string caseName = "3_hanging2";

    std::string path = "lesson13/data/" + caseName + "/";
    std::string results = "lesson13/resultsData/" + caseName + "/";

    if (!std::filesystem::exists(results)) {    
        std::filesystem::create_directory(results);    
    }

    cv::Mat img0 = cv::imread(path + "hanging0.png");
    cv::Mat img1 = cv::imread(path + "hanging1.png");
    rassert(!img0.empty() && !img1.empty(), 2389827851080019);
    rassert(img0.type() == CV_8UC3, 2389827851080020);
    rassert(img1.type() == CV_8UC3, 2389827851080021);

    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints0, keypoints1;    
    cv::Mat descriptors0, descriptors1;    
    std::cout << "Detecting SIFT keypoints and describing them (computing their descriptors)..." << std::endl;
    detector->detectAndCompute(img0, cv::noArray(), keypoints0, descriptors0);
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    std::cout << "SIFT keypoints detected and described: " << keypoints0.size() << " and " << keypoints1.size() << std::endl;    

    {
        cv::Mat img0withKeypoints, img1withKeypoints;
        cv::drawKeypoints(img0, keypoints0, img0withKeypoints);
        cv::drawKeypoints(img1, keypoints1, img1withKeypoints);
        cv::imwrite(results + "01keypoints0.jpg", img0withKeypoints);
        cv::imwrite(results + "01keypoints1.jpg", img1withKeypoints);
    }


    std::vector<std::vector<cv::DMatch>> matches01;
    std::cout << "Matching " << keypoints0.size() << " points with " << keypoints1.size() << "..." << std::endl;    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher->knnMatch(descriptors0, descriptors1, matches01, 2);    
    std::cout << "matching done" << std::endl;
       
    rassert(keypoints0.size() == matches01.size(), 234728972980049);
    for (int i = 0; i < matches01.size(); ++i) {
        rassert(matches01[i].size() == 2, 3427890347902051);
        rassert(matches01[i][0].queryIdx == i, 237812974128941);    
        rassert(matches01[i][1].queryIdx == i, 237812974128942);    

        rassert(matches01[i][0].trainIdx < keypoints1.size(), 237812974128943);    
        rassert(matches01[i][1].trainIdx < keypoints1.size(), 237812974128943);    

        rassert(matches01[i][0].distance <= matches01[i][1].distance, 328493778);    
    }

    std::vector<double> distances01;
    for (auto & i : matches01) {
        distances01.push_back(i[0].distance);
    }
    std::sort(distances01.begin(), distances01.end());
    std::cout << "matches01 distances min/median/max: " << distances01[0] << "/" << distances01[distances01.size() / 2] << "/" << distances01[distances01.size() - 1] << std::endl;
    for (int k = 0; k <=1; ++k) {
        std::vector<cv::DMatch> matchesK;
        for (int i = 0; i < matches01.size(); ++i) {
            matchesK.push_back(matches01[i][k]);
        }

        cv::Mat imgWithMatches;
        cv::drawMatches(img0, keypoints0, img1, keypoints1, matchesK, imgWithMatches);
        cv::imwrite(results + "02matches01_k" + std::to_string(k) + ".jpg", imgWithMatches);
    }


    std::vector<std::vector<cv::DMatch>> matches10;
    std::cout << "Matching " << keypoints1.size() << " points with " << keypoints0.size() << "..." << std::endl;
    matcher->knnMatch(descriptors1, descriptors0, matches10, 2);
    std::cout << "matching done" << std::endl;
    rassert(keypoints1.size() == matches10.size(), 234728972980049);
    for (int i = 0; i < matches10.size(); ++i) {
        rassert(matches10[i].size() == 2, 3427890347902051);
        rassert(matches10[i][0].queryIdx == i, 237812974128941);
        rassert(matches10[i][1].queryIdx == i, 237812974128942);

        rassert(matches10[i][0].trainIdx < keypoints0.size(), 237812974128943);
        rassert(matches10[i][1].trainIdx < keypoints0.size(), 237812974128943);

        rassert(matches10[i][0].distance <= matches10[i][1].distance, 328493778);
    }

    std::vector<double> distances10;
    for (auto & i : matches10) {
        distances10.push_back(i[0].distance);
    }
    std::sort(distances10.begin(), distances10.end());
    std::cout << "matches10 distances min/median/max: " << distances10[0] << "/" << distances10[distances10.size() / 2] << "/" << distances10[distances10.size() - 1] << std::endl;
    for (int k = 0; k <=1; ++k) {
        std::vector<cv::DMatch> matchesK;
        for (int i =    0; i < matches10.size();    ++i) {
            matchesK.push_back(matches10[i][k]);
        }
        cv::Mat imgWithMatches;
        cv::drawMatches(img1, keypoints1, img0, keypoints0, matchesK, imgWithMatches);
        cv::imwrite(results + "02matches10_k" + std::to_string(k) + ".jpg", imgWithMatches);
    }

    std::cout << "Filtering matches..." << std::endl;
    std::vector<cv::Point2f> points0, points1;
    std::vector<unsigned char> matchIsGood;
    int nmatchesGood = 0;
    for (int i = 0; i < keypoints0.size(); ++i) {
        cv::DMatch match = matches01[i][0];
        rassert(match.queryIdx == i, 234782749278097);
        int j = match.trainIdx;
        rassert(j < keypoints1.size(), 38472957238099);

        points0.push_back(keypoints0[i].pt);
        points1.push_back(keypoints1[j].pt);

        bool isOk =          true;

        if (match.distance > distances01[distances01.size()/2]) {
            isOk =    false;
        }

        double k = 0.7;
        cv::DMatch match2 = matches01[i][1];
        if (match.distance > k*match2.distance) {
            isOk = false;
        }

        cv::DMatch match01 = match;
        cv::DMatch match10 = matches10[match01.trainIdx][0];
        if (match10.trainIdx != i) {
            isOk = false;
        }

           
           
           
           



        if (isOk) {
            ++nmatchesGood;
            matchIsGood.push_back(true);
        } else {
            matchIsGood.push_back(false);
        }
    }
    rassert(points0.size() == points1.size(), 3497282579850108);
    rassert(points0.size() == matchIsGood.size(), 3497282579850109);
    std::cout << nmatchesGood << "/" << points0.size() << " good matches left" << std::endl;

    {
        std::vector<cv::DMatch> goodMatches;
        for (int i = 0; i < matches01.size(); ++i) {
            cv::DMatch match = matches01[i][0];
            rassert(match.queryIdx == i, 2347982739280182);
            if (!matchIsGood[i])
                continue;

            goodMatches.push_back(match);
        }
        cv::Mat imgWithMatches;
        cv::drawMatches(img0, keypoints0, img1, keypoints1, goodMatches, imgWithMatches);
        cv::imwrite(results + "04goodMatches01.jpg", imgWithMatches);
    }
    std::cout << "Finding Homography matrix from image0 to image1..." << std::endl;
    const double ransacReprojThreshold = 3.0;
    std::vector<cv::Point2f> pointsGood0 = filterPoints(points0, matchIsGood);
    std::vector<cv::Point2f> pointsGood1 = filterPoints(points1, matchIsGood);
    std::vector<unsigned char> inliersMask;    
    cv::Mat H01 = cv::findHomography(pointsGood0, pointsGood1, cv::RANSAC, ransacReprojThreshold, inliersMask);
    std::cout << "homography matrix found:" << std::endl;
    std::cout << H01 << std::endl;
    {
        std::vector<cv::DMatch> inliersMatches;
        std::vector<cv::KeyPoint> inlierKeypoints0, inlierKeypoints1;

        int ninliers = 0;
        for (int i = 0; i < inliersMask.size(); ++i) {
            if (inliersMask[i]) {
                ++ninliers;
                cv::DMatch match;
                match.queryIdx = inlierKeypoints0.size();
                match.trainIdx = inlierKeypoints1.size();
                inliersMatches.push_back(match);
                inlierKeypoints0.push_back(cv::KeyPoint(pointsGood0[i], 3.0));
                inlierKeypoints1.push_back(cv::KeyPoint(pointsGood1[i], 3.0));
            }
        }

           
        std::cout << "(with " << ninliers << "/" << nmatchesGood << " inliers matches)" << std::endl;
        cv::Mat imgWithInliersMatches;
        cv::drawMatches(img0, inlierKeypoints0, img1, inlierKeypoints1, inliersMatches, imgWithInliersMatches);
        cv::imwrite(results + "05inliersMatches01.jpg", imgWithInliersMatches);
    }


    cv::imwrite(results + "08img0.jpg", img0);    

    cv::Mat H10 = H01.inv();    
    cv::Mat img1to0;
    cv::warpPerspective(img1,img1to0, H10, img0.size());
    cv::imwrite(results + "09img1to0.jpg", img1to0);

    img1to0 = img0.clone();    
    cv::warpPerspective(img1, img1to0, H10, img1to0.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
    cv::imwrite(results + "10img0with1to0.jpg", img1to0);
}

void test4() {
    std::string caseName = "4_aero9";
}


int main() {
    try {
        test1();    
        test2();    
        test3();    

   

        return 0;
    } catch (const std::exception &e) {
        std::cout << "Exception! " << e.what() << std::endl;
        return 1;
    }
}