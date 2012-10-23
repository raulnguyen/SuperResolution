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

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ @0           | car.avi   | Input video }"
        "{ s scale      | 4         | Scale factor }"
        "{ i iterations | 180       | Iteration count }"
        "{ t temporal   | 4         | Radius of the temporal search area }"
        "{ f opt-flow   | farneback | Optical flow algorithm (farneback, simple, tvl1, brox, pyrlk) }"
        "{ gpu          |           | Use GPU }"
        "{ h help       |           | Print help message }"
    );

    if (cmd.has("help"))
    {
        cmd.about("This sample demonstrates Super Resolution algorithms for video sequence");
        cmd.printMessage();
        return 0;
    }

    const string inputVideoName = cmd.get<string>(0);
    const int scale = cmd.get<int>("scale");
    const int iterations = cmd.get<int>("iterations");
    const int temporalAreaRadius = cmd.get<int>("temporal");
    const string optFlow = cmd.get<string>("opt-flow");
    const bool useGpu = cmd.has("gpu");

    if (!cmd.check())
    {
        cmd.printErrors();
        return -1;
    }

    Ptr<DenseOpticalFlow> optFlowAlg;
    if (optFlow == "farneback")
    {
        if (useGpu)
            optFlowAlg = new Farneback_GPU;
        else
            optFlowAlg = new Farneback;
    }
    else if (optFlow == "simple")
        optFlowAlg = new Simple;
    else if (optFlow == "tvl1")
        optFlowAlg = new Dual_TVL1;
    else if (optFlow == "brox")
        optFlowAlg = new Brox_GPU;
    else if (optFlow == "pyrlk")
        optFlowAlg = new PyrLK_GPU;
    else
    {
        cerr << "Incorrect Optical Flow algorithm - " << optFlow << endl;
        cmd.printMessage();
        return -1;
    }

    Ptr<SuperResolution> superRes;
    if (useGpu)
        superRes = new BTV_L1_GPU;
    else
        superRes = new BTV_L1;

    superRes->set("scale", scale);
    superRes->set("iterations", iterations);
    superRes->set("temporalAreaRadius", temporalAreaRadius);
    superRes->set("opticalFlow", optFlowAlg);

    Ptr<IFrameSource> frameSource(new VideoFileSource(inputVideoName));
    // skip first frame, it is usually corrupted
    {
        Mat frame = frameSource->nextFrame();
        cout << "Input             : " << inputVideoName << " " << frame.size() << endl;
        cout << "Scale factor      : " << scale << endl;
        cout << "Iterations        : " << iterations << endl;
        cout << "Frames to process : " << temporalAreaRadius * 2 + 1 << endl;
        cout << "Optical Flow      : " << optFlow << endl;
        cout << "Mode              : " << (useGpu ? "GPU" : "CPU") << endl;
    }

    superRes->setFrameSource(frameSource);

    for (int i = 0;; ++i)
    {
        cout << '[' << setw(3) << i << "] : ";

        Mat result;
        MEASURE_TIME(result = superRes->nextFrame());

        if (result.empty())
            break;

        imshow("Super Resolution", result);

        if (waitKey(1000) > 0)
            break;
    }

    return 0;
}
