// Copyright (c) 2012, Vladislav Vinogradov (jet47)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "super_resolution.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

namespace
{
    double getPSNR(const Mat& src1, const Mat& src2)
    {
        CV_Assert( src1.type() == CV_8UC1 );
        CV_Assert( src2.size() == src1.size() );
        CV_Assert( src2.type() == src1.type() );

        double sse = 0.0;
        for (int y = 0; y < src1.rows; ++y)
        {
            const uchar* src1Row = src1.ptr(y);
            const uchar* src2Row = src2.ptr(y);

            for (int x = 0; x < src1.cols; ++x)
                sse += (src1Row[x] - src2Row[x]) * (src1Row[x] - src2Row[x]);
        }

        if (sse == 0.0)
            return 0;

        const double mse = sse / src1.size().area();

        return 10.0 * log10(255 * 255 / mse);
    }

    void addGaussNoise(Mat& image, double sigma, RNG& rng)
    {
        Mat_<float> noise(image.size());
        rng.fill(noise, RNG::NORMAL, 0.0, sigma);

        addWeighted(image, 1.0, noise, 1.0, 0.0, image, image.depth());
    }

    void addSpikeNoise(Mat& image, int frequency, RNG& rng)
    {
        Mat_<uchar> mask(image.size(), 0);

        for (int y = 0; y < mask.rows; ++y)
        {
            for (int x = 0; x < mask.cols; ++x)
            {
                if (rng.uniform(0, frequency) < 1)
                    mask(y, x) = 255;
            }
        }

        image.setTo(Scalar::all(255), mask);
    }

    Mat createDegradedImage(const Mat& src, int scale, RNG& rng, bool doShift = true)
    {
        const double dscale = scale;
        const double iscale = 1.0 / dscale;

        Mat shifted;
        if (!doShift)
            shifted = src;
        else
        {
            Point2d move;
            move.x = rng.uniform(-dscale, dscale);
            move.y = rng.uniform(-dscale, dscale);

            const double theta = rng.uniform(-CV_PI / 30, CV_PI / 30);

            Mat_<float> M(2, 3);
            M << cos(theta), -sin(theta), move.x,
                 sin(theta),  cos(theta), move.y;

            warpAffine(src, shifted, M, src.size(), INTER_NEAREST);
        }

        Mat blurred;
        GaussianBlur(shifted, blurred, Size(5, 5), 0);

        Mat deg;
        resize(blurred, deg, Size(), iscale, iscale, INTER_NEAREST);

        addGaussNoise(deg, 10.0, rng);
        addSpikeNoise(deg, 500, rng);

        return deg;
    }
}

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ i image | boy.png | Input image }"
        "{ s scale | 4       | Scale factor }"
        "{ h help  |         | Print help message }"
    );

    if (cmd.has("help"))
    {
        cmd.about("This sample measure quality of Super Resolution algorithm");
        cmd.printMessage();
        return 0;
    }

    const string imageFileName = cmd.get<string>("image");
    const int scale = cmd.get<int>("scale");

    Mat gold = imread(imageFileName, IMREAD_GRAYSCALE);
    if (gold.empty())
    {
        cerr << "Can't open image " << imageFileName << endl;
        return -1;
    }

    if (gold.rows % scale != 0 || gold.cols % scale != 0)
        gold = gold(Rect(0, 0, (gold.cols / scale) * scale, (gold.rows / scale) * scale));

    RNG rng(12345678);

    const int degImagesCount = 9;
    vector<Mat> degImages(degImagesCount);

    degImages[0] = createDegradedImage(gold, scale, rng, false);
    for (int i = 1; i < degImagesCount; ++i)
        degImages[i] = createDegradedImage(gold, scale, rng);

    BTV_L1_Base alg;
    alg.scale = scale;

    Mat highResImage;
    alg.process(degImages, highResImage);

    Rect inner(alg.btvKernelSize, alg.btvKernelSize, gold.cols - 2 * alg.btvKernelSize, gold.rows - 2 * alg.btvKernelSize);
    cout << "PSNR : " << getPSNR(gold(inner), highResImage) << " dB" << endl;

    imshow("Gold", gold);
    imshow("Super Resolution", highResImage);

    waitKey();

    return 0;
}
