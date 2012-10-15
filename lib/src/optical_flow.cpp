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

#include "optical_flow.hpp"
#include <opencv2/core/internal.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::superres;

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(Farneback, "DenseOpticalFlow.Farneback",
                          obj.info()->addParam(obj, "pyrScale", obj.pyrScale);
                          obj.info()->addParam(obj, "numLevels", obj.numLevels);
                          obj.info()->addParam(obj, "winSize", obj.winSize);
                          obj.info()->addParam(obj, "numIters", obj.numIters);
                          obj.info()->addParam(obj, "polyN", obj.polyN);
                          obj.info()->addParam(obj, "polySigma", obj.polySigma);
                          obj.info()->addParam(obj, "flags", obj.flags));
    }
}

cv::superres::Farneback::Farneback()
{
    pyrScale = 0.5;
    numLevels = 5;
    winSize = 13;
    numIters = 10;
    polyN = 5;
    polySigma = 1.1;
    flags = 0;
}

void cv::superres::Farneback::calc(InputArray _frame0, InputArray _frame1, OutputArray flow)
{
    Mat frame0 = _frame0.getMat();
    Mat frame1 = _frame1.getMat();

    CV_DbgAssert( frame0.depth() == CV_8U );
    CV_DbgAssert( frame0.channels() == 1 || frame0.channels() == 3 || frame0.channels() == 4 );
    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    const Mat* input0;
    const Mat* input1;

    if (frame0.channels() == 1)
    {
        input0 = &frame0;
        input1 = &frame1;
    }
    else if (frame0.channels() == 3)
    {
        cvtColor(frame0, buf0, COLOR_BGR2GRAY);
        cvtColor(frame1, buf1, COLOR_BGR2GRAY);
        input0 = &buf0;
        input1 = &buf1;
    }
    else
    {
        cvtColor(frame0, buf0, COLOR_BGRA2GRAY);
        cvtColor(frame1, buf1, COLOR_BGRA2GRAY);
        input0 = &buf0;
        input1 = &buf1;
    }

    calcOpticalFlowFarneback(*input0, *input1, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);
}

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(SimpleFlow, "DenseOpticalFlow.SimpleFlow",
                          obj.info()->addParam(obj, "layers", obj.layers);
                          obj.info()->addParam(obj, "averagingBlockSize", obj.averagingBlockSize);
                          obj.info()->addParam(obj, "maxFlow", obj.maxFlow);
                          obj.info()->addParam(obj, "sigmaDist", obj.sigmaDist);
                          obj.info()->addParam(obj, "sigmaColor", obj.sigmaColor);
                          obj.info()->addParam(obj, "postProcessWindow", obj.postProcessWindow);
                          obj.info()->addParam(obj, "sigmaDistFix", obj.sigmaDistFix);
                          obj.info()->addParam(obj, "sigmaColorFix", obj.sigmaColorFix);
                          obj.info()->addParam(obj, "occThr", obj.occThr);
                          obj.info()->addParam(obj, "upscaleAveragingRadius", obj.upscaleAveragingRadius);
                          obj.info()->addParam(obj, "upscaleSigmaDist", obj.upscaleSigmaDist);
                          obj.info()->addParam(obj, "upscaleSigmaColor", obj.upscaleSigmaColor);
                          obj.info()->addParam(obj, "speedUpThr", obj.speedUpThr));
    }
}

cv::superres::SimpleFlow::SimpleFlow()
{
    layers = 3;
    averagingBlockSize = 2;
    maxFlow = 4;
    sigmaDist = 4.1;
    sigmaColor = 25.5;
    postProcessWindow = 18;
    sigmaDistFix = 55.0;
    sigmaColorFix = 25.5;
    occThr = 0.35;
    upscaleAveragingRadius = 18;
    upscaleSigmaDist = 55.0;
    upscaleSigmaColor = 25.5;
    speedUpThr = 10;
}

void cv::superres::SimpleFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray flow)
{
    Mat frame0 = _frame0.getMat();
    Mat frame1 = _frame1.getMat();

    CV_DbgAssert( frame0.depth() == CV_8U );
    CV_DbgAssert( frame0.channels() == 1 || frame0.channels() == 3 || frame0.channels() == 4 );
    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    Mat* input0;
    Mat* input1;

    if (frame0.channels() == 3)
    {
        input0 = &frame0;
        input1 = &frame1;
    }
    else if (frame0.channels() == 1)
    {
        cvtColor(frame0, buf0, COLOR_GRAY2BGR);
        cvtColor(frame1, buf1, COLOR_GRAY2BGR);
        input0 = &buf0;
        input1 = &buf1;
    }
    else
    {
        cvtColor(frame0, buf0, COLOR_BGRA2BGR);
        cvtColor(frame1, buf1, COLOR_BGRA2BGR);
        input0 = &buf0;
        input1 = &buf1;
    }

    calcOpticalFlowSF(*input0, *input1, flow.getMatRef(),
                      layers,
                      averagingBlockSize,
                      maxFlow,
                      sigmaDist,
                      sigmaColor,
                      postProcessWindow,
                      sigmaDistFix,
                      sigmaColorFix,
                      occThr,
                      upscaleAveragingRadius,
                      upscaleSigmaDist,
                      upscaleSigmaColor,
                      speedUpThr);
}
