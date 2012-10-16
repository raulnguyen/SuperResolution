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
#include <iomanip>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include "super_resolution.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;

#define MEASURE_TIME(op) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << tm.getTimeSec() << " sec" << endl; \
    }

class GrayScaleVideoSource : public IFrameSource
{
public:
    explicit GrayScaleVideoSource(const Ptr<IFrameSource>& base) : base(base) {}

    void reset();
    Mat nextFrame();

private:
    Ptr<IFrameSource> base;
    Mat buf;
};

void GrayScaleVideoSource::reset()
{
    base->reset();
}

Mat GrayScaleVideoSource::nextFrame()
{
    Mat frame = base->nextFrame();

    if (frame.channels() == 1)
        return frame;

    cvtColor(frame, buf, frame.channels() == 3 ? COLOR_BGR2GRAY : COLOR_BGRA2GRAY);
    return buf;
}

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ video v | text.avi | Input video }"
        "{ scale s | 4        | Scale factor }"
        "{ gpu     |          | Use GPU }"
        "{ help h  |          | Print help message }"
    );

    if (!cmd.check())
    {
        cmd.printErrors();
        return -1;
    }

    if (cmd.has("help"))
    {
        cmd.about("This sample demonstrates Super Resolution algorithms for video sequence");
        cmd.printMessage();
        return 0;
    }

    const string inputVideoName = cmd.get<string>("video");
    const int scale = cmd.get<int>("scale");
    const bool useGpu = cmd.has("gpu");

    Ptr<SuperResolution> superRes;
    if (useGpu)
        superRes = new BTV_L1_GPU;
    else
        superRes = new BTV_L1;

    superRes->set("scale", scale);

    Ptr<IFrameSource> superResSource(new GrayScaleVideoSource(new VideoFileSource(inputVideoName)));
    Ptr<IFrameSource> bicubicSource(new GrayScaleVideoSource(new VideoFileSource(inputVideoName)));

    // skip first frame, it is usually corrupted
    {
        superResSource->nextFrame();
        Mat frame = bicubicSource->nextFrame();
        cout << "Input size : " << frame.cols << 'x' << frame.rows << endl;
        cout << "Scale factor : " << scale << endl;
        cout << "Mode : " << (useGpu ? "GPU" : "CPU") << endl;
    }

    superRes->setFrameSource(superResSource);

    for (int i = 0;; ++i)
    {
        Mat frame = bicubicSource->nextFrame();

        if (frame.empty())
            break;

        cout << '[' << setw(3) << i << "] : ";

        Mat result;
        MEASURE_TIME(result = superRes->nextFrame());

        if (result.empty())
            break;

        Mat bicubic;
        resize(frame, bicubic, Size(), scale, scale, INTER_CUBIC);

        imshow("Input", frame);
        imshow("Super Resolution", result);
        imshow("BiCubic", bicubic);

        if (waitKey(30) > 0)
            break;
    }

    return 0;
}
