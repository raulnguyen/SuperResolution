#include "nlm_based.hpp"
#include "extract_patch.hpp"
#include <iostream>

using namespace std;
using namespace cv;

Ptr<VideoSuperResolution> NlmBased::create()
{
    return Ptr<VideoSuperResolution>(new NlmBased);
}

NlmBased::NlmBased()
{
    scale = 2;
    searchAreaSize = 5;
    timeAreaSize = 5;
    timeStep = 1;
    lowResPatchSize = 7;
    sigma = 1.0;
}

void NlmBased::process(VideoCapture& cap, Mat& dst)
{
    // input set of low resolution and noisy images
    vector<Mat> y(timeAreaSize);
    Mat imTmp;
    for (int i = 0; i < 20; ++i)
        cap >> imTmp;
    for (int t = 0; t < timeAreaSize; ++t)
    {
        for (int i = 0; i < timeStep; ++i)
            cap >> imTmp;

        CV_Assert(!imTmp.empty());

        imTmp.copyTo(y[t]);
    }

    const int s = scale;                // the desired scaling factor
    const int q = lowResPatchSize;      // the size of the low resolution patch
    const int p = s * (q - 1) + 1;      // the size of the high resolution patch
    const int rad = searchAreaSize / 2; // the radius of search area

    // An initial estimate of the super-resolved sequence.
    vector<Mat> Y(timeAreaSize);
    for (int t = 0; t < timeAreaSize; ++t)
        pyrUp(y[t], Y[t], Size(y[t].cols * s, y[t].rows * s));

    Mat_<Point3d> V;
    Y.back().convertTo(V, CV_64F);
    Mat_<Point3d> W(V.size(), Point3d(1,1,1));

    const double weightScale = 1.0 / (2.0 * sigma * sigma);

    vector<double> patch1;
    vector<double> patch2;

    for (int k = 0; k < V.rows; ++k)
    {
        cout << "Process : " << static_cast<double>(k) / V.rows * 100 << " %" << endl;

        for (int l = 0; l < V.cols; ++l)
        {
            for (int t = 0; t < timeAreaSize; ++t)
            {
                for (int i = k / s - rad; i <= k / s + rad; ++i)
                {
                    for (int j = l / s - rad; j <= l / s + rad; ++j)
                    {
                        // Compute Weight

                        extractPatch(Y.back(), Point2d(l, k), patch1, p, INTER_NEAREST);
                        extractPatch(Y[t], Point2d(s * j, s * i), patch2, p, INTER_NEAREST);

                        double norm2 = 0.0;
                        for (size_t n = 0; n < patch1.size(); ++n)
                        {
                            const double diff = patch1[n] - patch2[n];
                            norm2 += diff * diff;
                        }

                        const double w = exp(-norm2 * weightScale);

                        // Accumulate Inputs
                        // Accumulate Weights

                        // Extract the low-resolution patch
                        extractPatch(y[t], Point2d(j, i), patch1, q, INTER_NEAREST);

                        // Upscale it by zero-filling
                        // Accumulate it in its proper location
                        Mat_<Point3d> lowResPatch(q, q, (Point3d*) &patch1[0]);
                        for (int y = 0; y < q; ++y)
                        {
                            const int hy = k + s * (y - q / 2);

                            if (hy >= 0 && hy < V.rows)
                            {
                                for (int x = 0; x < q; ++x)
                                {
                                    const int hx = l + s * (x - q / 2);

                                    if (hx >= 0 && hx < V.cols)
                                    {
                                        const Point3d val = lowResPatch(y, x);

                                        V(hy, hx) += w * val;
                                        W(hy, hx) += Point3d(w, w, w);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Normalization
    Mat Z;
    divide(V, W, Z);

    // TODO : Deblurring
    Z.convertTo(dst, CV_8U);
}
