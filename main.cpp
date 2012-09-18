#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "SuperResolution.h"

using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        cerr << "Missing image file name" << endl;
        return -1;
    }

    Mat image = imread(argv[1]/*, IMREAD_GRAYSCALE*/);
    if (image.empty())
    {
        cerr << "Can't open image " << argv[1] << endl;
        return -1;
    }

    TickMeter tm;

    SuperResolution superRes;
    Mat superRes_result;

    tm.start();
    superRes(image, superRes_result);
    tm.stop();

    cout << "Time : " << tm.getTimeMilli() << " ms" << endl;

    namedWindow("Super Resolution", WINDOW_NORMAL);
    imshow("Super Resolution", superRes_result);

    waitKey();

    return 0;
}
