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

#include "tv-l1.hpp"
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videostab/ring_buffer.hpp>

using namespace std;
using namespace cv;
using namespace cv::videostab;
using namespace cv::superres;

cv::superres::TV_L1_Base::TV_L1_Base()
{
    scale = 4;
    iterations = 150;
    lambda = 0.4;
    tau = 0.01;
}

namespace
{
    void calcOpticalFlow(const Mat& frame0, const Mat& frame1, Mat_<Point2f>& flow)
    {
        Mat gray0, gray1;

        if (frame0.channels() == 1)
        {
            gray0 = frame0;
            gray1 = frame1;
        }
        else if (frame0.channels() == 3)
        {
            cvtColor(frame0, gray0, COLOR_BGR2GRAY);
            cvtColor(frame1, gray1, COLOR_BGR2GRAY);
        }
        else
        {
            cvtColor(frame0, gray0, COLOR_BGRA2GRAY);
            cvtColor(frame1, gray1, COLOR_BGRA2GRAY);
        }

        calcOpticalFlowFarneback(gray0, gray1, flow,
                                 /*pyrScale =*/ 0.5,
                                 /*numLevels =*/ 5,
                                 /*winSize =*/ 13,
                                 /*numIters =*/ 10,
                                 /*polyN =*/ 5,
                                 /*polySigma =*/ 1.1,
                                 /*flags =*/ 0);
    }

    void calcMotions(const vector<Mat>& src, int startIdx, int procIdx, int endIdx, vector<Mat_<Point2f> >& motions)
    {
        motions.resize(src.size());

        at(procIdx, motions).create(at(procIdx, src).size());
        at(procIdx, motions).setTo(Scalar::all(0));

        for (int i = startIdx; i < procIdx; ++i)
            calcOpticalFlow(at(i, src), at(i + 1, src), at(i, motions));
        for (int i = procIdx + 1; i <= endIdx; ++i)
            calcOpticalFlow(at(i, src), at(i - 1, src), at(i, motions));

        for (int i = procIdx - 1; i > startIdx; --i)
            add(at(i - 1, motions), at(i, motions), at(i - i, motions));
        for (int i = procIdx + 1; i < endIdx; ++i)
            add(at(i + 1, motions), at(i, motions), at(i + 1, motions));
    }

    void builWarpMaps(const Mat_<Point2f>& motion, Mat_<Point2f>& forward, Mat_<Point2f>& backward)
    {
        forward.create(motion.size());
        backward.create(motion.size());

        for (int y = 0; y < motion.rows; ++y)
        {
            for (int x = 0; x < motion.cols; ++x)
            {
                forward(y, x) = Point2f(x, y) - motion(y, x);
                backward(y, x) = Point2f(x, y) + motion(y, x);
            }
        }
    }
}

void cv::superres::TV_L1_Base::process(const vector<Mat>& src, Mat& dst, int startIdx, int procIdx, int endIdx)
{
    CV_DbgAssert( !src.empty() );
    CV_DbgAssert( procIdx >= startIdx && endIdx >= procIdx );
    CV_DbgAssert( scale > 1 );
#ifdef _DEBUG
    for (size_t i = 1; i < src.size(); ++i)
    {
        CV_DbgAssert( src[i].size() == src[0].size() );
        CV_DbgAssert( src[i].type() == src[0].type() );
    }
#endif

    const Size lowResSize = src[0].size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

    vector<Mat_<Point2f> > lowResMotions;
    calcMotions(src, startIdx, procIdx, endIdx, lowResMotions);

    vector<Mat_<Point2f> > highResMotions(lowResMotions.size());
    for (size_t i = 0; i < lowResMotions.size(); ++i)
    {
        resize(lowResMotions[i], highResMotions[i], highResSize);
        multiply(highResMotions[i], Scalar::all(scale), highResMotions[i]);
    }

    vector<Mat_<Point2f> > forward(highResMotions.size()), backward(highResMotions.size());
    for (size_t i = 0; i < highResMotions.size(); ++i)
        builWarpMaps(highResMotions[i], forward[i], backward[i]);

    vector<Mat> src_f(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        src[i].convertTo(src_f[i], CV_32F);

    Mat highRes(highResSize, CV_32FC(src[0].channels()), Scalar::all(0));
    Mat sum(highResSize, highRes.type());
    Mat a, b, c, d;
    Mat diff;
    Mat lap;
    Mat term;

    for (int i = 0; i < iterations; ++i)
    {
        sum.setTo(Scalar::all(0));

        for (size_t k = 0; k < src.size(); ++k)
        {
            remap(highRes, a, backward[k], noArray(), INTER_NEAREST);
            GaussianBlur(a, b, Size(5, 5), 0);
            resize(b, c, lowResSize, 0, 0, INTER_NEAREST);

            subtract(src_f[k], c, diff);

            resize(diff, d, highResSize, 0, 0, INTER_NEAREST);
            GaussianBlur(d, b, Size(5, 5), 0);
            remap(b, a, forward[k], noArray(), INTER_NEAREST);

            add(sum, a, sum);
        }

        Laplacian(highRes, lap, highRes.depth());

        addWeighted(sum, 1.0, lap, -lambda, 0.0, term);

        addWeighted(highRes, 1.0, term, tau, 0.0, highRes);
    }

    highRes.convertTo(dst, src[0].depth());
}

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(TV_L1, "SuperResolution.TV_L1",
                          obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0,
                                               "Scale factor.");
                          obj.info()->addParam(obj, "iterations", obj.iterations, false, 0, 0,
                                               "Iteration count.");
                         obj.info()->addParam(obj, "lambda", obj.lambda, false, 0, 0,
                                              "Weight parameter to balance data term and smoothness term.");
                          obj.info()->addParam(obj, "tau", obj.tau, false, 0, 0,
                                               "Asymptotic value of steepest descent method.");
                          obj.info()->addParam(obj, "temporalAreaRadius", obj.temporalAreaRadius, false, 0, 0,
                                               "Radius of the temporal search area."));
    }
}

bool cv::superres::TV_L1::init()
{
    return !TV_L1_info_auto.name().empty();
}

Ptr<SuperResolution> cv::superres::TV_L1::create()
{
    Ptr<SuperResolution> alg(new TV_L1);
    return alg;
}

cv::superres::TV_L1::TV_L1()
{
    temporalAreaRadius = 4;
}

void cv::superres::TV_L1::initImpl(Ptr<IFrameSource>& frameSource)
{
    const int cacheSize = 2 * temporalAreaRadius + 1;

    frames.resize(cacheSize);
    results.resize(cacheSize);

    storePos = -1;

    for (int t = -temporalAreaRadius; t <= temporalAreaRadius; ++t)
    {
        Mat frame = frameSource->nextFrame();
        CV_Assert( !frame.empty() );
        addNewFrame(frame);
    }

    for (int i = 0; i <= temporalAreaRadius; ++i)
        processFrame(i);

    procPos = temporalAreaRadius;
    outPos = -1;
}

Mat cv::superres::TV_L1::processImpl(Ptr<IFrameSource>& frameSource)
{
    Mat frame = frameSource->nextFrame();
    addNewFrame(frame);

    if (procPos < storePos)
    {
        ++procPos;
        processFrame(procPos);
    }

    if (outPos < storePos)
    {
        ++outPos;
        return at(outPos, results);
    }

    return Mat();
}

void cv::superres::TV_L1::addNewFrame(const Mat& frame)
{
    if (frame.empty())
        return;

    CV_DbgAssert( storePos < 0 || frame.size() == at(storePos, frames).size() );

    ++storePos;
    frame.copyTo(at(storePos, frames));
}

void cv::superres::TV_L1::processFrame(int idx)
{
    process(frames, at(idx, results), idx - temporalAreaRadius, idx, idx + temporalAreaRadius);
}
