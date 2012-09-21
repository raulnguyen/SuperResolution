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

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "image_super_resolution.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

#define MEASURE_TIME(op, msg) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << msg << " Time : " << tm.getTimeMilli() << " ms" << endl; \
    }

namespace
{
    vector<string> split_string(const string& _str, char symbol)
    {
        string str = _str;
        string word;
        vector<string> vec;

        while (!str.empty())
        {
            if (str[0] == symbol)
            {
                if (!word.empty())
                {
                    vec.push_back(word);
                    word = "";
                }
            }
            else
            {
                word += str[0];
            }
            str = str.substr(1, str.length() - 1);
        }

        if (!word.empty())
        {
            vec.push_back(word);
        }

        return vec;
    }
}

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ image i | boy.png | Input image }"
        "{ scale s | 2       | Scale factor }"
        "{ train t | none    | Train images (separated by :) }"
    );

    const string imageFileName = cmd.get<string>("image");
    const double scale = cmd.get<double>("scale");

    Mat image = imread(imageFileName);
    if (image.empty())
    {
        cerr << "Can't open image " << imageFileName << endl;
        return -1;
    }

    vector<Mat> trainImages;
    const string trainImagesStr = cmd.get<string>("train");
    if (trainImagesStr != "none")
    {
        const vector<string> trainImagesStrVec = split_string(trainImagesStr, ':');
        for (size_t i = 0; i < trainImagesStrVec.size(); ++i)
        {
            Mat curImage = imread(trainImagesStrVec[i]);
            if (image.empty())
            {
                cerr << "Can't open image " << trainImagesStrVec[i] << endl;
                return -1;
            }
            trainImages.push_back(curImage);
        }
    }

    Ptr<ImageSuperResolution> superRes = ImageSuperResolution::create(IMAGE_SR_EXAMPLE_BASED);
    Mat highResImage;

    superRes->set("scale", scale);

    if (!trainImages.empty())
    {
        MEASURE_TIME(superRes->train(trainImages), "Train");
    }

    MEASURE_TIME(superRes->process(image, highResImage), "Process");

    Mat bicubic;
    resize(image, bicubic, Size(), scale, scale, INTER_CUBIC);

    namedWindow("Input Image", WINDOW_NORMAL);
    imshow("Input Image", image);

    namedWindow("Super Resolution", WINDOW_NORMAL);
    imshow("Super Resolution", highResImage);

    namedWindow("Bi-Cubic Interpolation", WINDOW_NORMAL);
    imshow("Bi-Cubic Interpolation", bicubic);

    waitKey();

    return 0;
}
