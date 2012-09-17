#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

    SuperResolution superRes;

    Mat superRes_result;
    superRes(image, superRes_result);
    namedWindow("Super Resolution", WINDOW_NORMAL);
    imshow("Super Resolution", superRes_result);

    Mat bicubic_result;
    resize(image, bicubic_result, Size(), 2, 2, INTER_CUBIC);
    namedWindow("Bi-Cubic interpolation", WINDOW_NORMAL);
    imshow("Bi-Cubic interpolation", bicubic_result);

    waitKey();

    return 0;
}
