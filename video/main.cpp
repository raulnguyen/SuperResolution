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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "video_super_resolution.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

#define MEASURE_TIME(op, msg) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << msg << " Time : " << tm.getTimeSec() << " s" << endl; \
    }

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ video v | small.avi | Input video }");

    const string inputVideoName = cmd.get<string>("video");

    VideoCapture cap(inputVideoName);
    if (!cap.isOpened())
    {
        cerr << "Can't open video " << inputVideoName << endl;
        return -1;
    }

    Ptr<VideoSuperResolution> superRes = VideoSuperResolution::create(VIDEO_SR_NLM_BASED);

    Mat dst;
    MEASURE_TIME(superRes->process(cap, dst), "Process");

    namedWindow("dst", WINDOW_NORMAL);
    imshow("dst", dst);
    waitKey();

    return 0;
}
