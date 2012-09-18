#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "SuperResolution.h"

using namespace std;
using namespace cv;

#define MEASURE_TIME(op, msg) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << msg << " Time : " << tm.getTimeMilli() << " ms" << endl; \
    }

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        cerr << "Missing image file name" << endl;
        return -1;
    }

    Mat goldImage = imread(argv[1]);
    if (goldImage.empty())
    {
        cerr << "Can't open image " << argv[1] << endl;
        return -1;
    }

    if (goldImage.cols % 2 != 0)
        goldImage = goldImage.colRange(0, goldImage.cols - 1);
    if (goldImage.rows % 2 != 0)
        goldImage = goldImage.rowRange(0, goldImage.rows - 1);

    Mat lowResImage;
    pyrDown(goldImage, lowResImage);

    SuperResolution superRes;
    Mat highResImage;

//    MEASURE_TIME(superRes.train(lowResImage), "Train");
    MEASURE_TIME(superRes.train(goldImage), "Train");

    MEASURE_TIME(superRes(lowResImage, highResImage), "Process");

    Mat diff;
    absdiff(goldImage, highResImage, diff);

    namedWindow("Gold Image", WINDOW_NORMAL);
    imshow("Gold Image", goldImage);

    namedWindow("Super Resolution", WINDOW_NORMAL);
    imshow("Super Resolution", highResImage);

    namedWindow("Diff", WINDOW_NORMAL);
    imshow("Diff", diff);

    waitKey();

    return 0;
}
