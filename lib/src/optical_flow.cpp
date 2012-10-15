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
#include <limits>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::gpu;

namespace
{
    Mat getCpuMat(InputArray arr, Mat& buf)
    {
        if (arr.kind() == _InputArray::GPU_MAT)
        {
            arr.getGpuMat().download(buf);
            return buf;
        }

        return arr.getMat();
    }

    GpuMat getGpuMat(InputArray arr, GpuMat& buf)
    {
        if (arr.kind() != _InputArray::GPU_MAT)
        {
            buf.upload(arr.getMat());
            return buf;
        }

        return arr.getGpuMat();
    }

    void set(OutputArray dst, const Mat& val)
    {
        if (dst.kind() == _InputArray::GPU_MAT)
            dst.getGpuMatRef().upload(val);
        else
            val.copyTo(dst);
    }

    void set(OutputArray dst, const GpuMat& val)
    {
        if (dst.kind() == _InputArray::GPU_MAT)
            val.copyTo(dst.getGpuMatRef());
        else
            val.download(dst.getMatRef());
    }

    Mat convertToType(const Mat& src, int depth, int cn, Mat& buf0, Mat& buf1)
    {
        CV_DbgAssert( depth == CV_8U || depth == CV_32F );
        CV_DbgAssert( cn == 1 || cn == 3 || cn == 4 );

        Mat result;

        if (src.depth() == depth)
            result = src;
        else
        {
            static const double maxVals[] =
            {
                numeric_limits<uchar>::max(),
                numeric_limits<schar>::max(),
                numeric_limits<ushort>::max(),
                numeric_limits<short>::max(),
                numeric_limits<int>::max(),
                1.0,
                1.0,
            };

            const double scale = maxVals[depth] / maxVals[src.depth()];

            src.convertTo(buf0, depth, scale);
            result = buf0;
        }

        if (result.channels() == cn)
            return result;

        static const int codes[5][5] =
        {
            {-1, -1, -1, -1, -1},
            {-1, -1, -1, COLOR_GRAY2BGR, COLOR_GRAY2BGRA},
            {-1, -1, -1, -1, -1},
            {-1, COLOR_BGR2GRAY, -1, -1, COLOR_BGR2BGRA},
            {-1, COLOR_BGRA2GRAY, -1, COLOR_BGRA2BGR, -1},
        };

        const int code = codes[src.channels()][cn];
        CV_DbgAssert( code >= 0 );

        cvtColor(result, buf1, code, cn);
        return buf1;
    }

    GpuMat convertToType(const GpuMat& src, int depth, int cn, GpuMat& buf0, GpuMat& buf1)
    {
        CV_DbgAssert( depth == CV_8U || depth == CV_32F );
        CV_DbgAssert( cn == 1 || cn == 3 || cn == 4 );

        GpuMat result;

        if (src.depth() == depth)
            result = src;
        else
        {
            static const double maxVals[] =
            {
                numeric_limits<uchar>::max(),
                numeric_limits<schar>::max(),
                numeric_limits<ushort>::max(),
                numeric_limits<short>::max(),
                numeric_limits<int>::max(),
                1.0,
                1.0,
            };

            const double scale = maxVals[depth] / maxVals[src.depth()];

            src.convertTo(buf0, depth, scale);
            result = buf0;
        }

        if (result.channels() == cn)
            return result;

        static const int codes[5][5] =
        {
            {-1, -1, -1, -1, -1},
            {-1, -1, -1, COLOR_GRAY2BGR, COLOR_GRAY2BGRA},
            {-1, -1, -1, -1, -1},
            {-1, COLOR_BGR2GRAY, -1, -1, COLOR_BGR2BGRA},
            {-1, COLOR_BGRA2GRAY, -1, COLOR_BGRA2BGR, -1},
        };

        const int code = codes[src.channels()][cn];
        CV_DbgAssert( code >= 0 );

        gpu::cvtColor(result, buf1, code, cn);
        return buf1;
    }
}

///////////////////////////////////////////////////////////////////
// FarnebackOpticalFlow

cv::superres::FarnebackOpticalFlow::FarnebackOpticalFlow()
{
    pyrScale = 0.5;
    numLevels = 5;
    winSize = 13;
    numIters = 10;
    polyN = 5;
    polySigma = 1.1;
    flags = 0;
}

void cv::superres::FarnebackOpticalFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray flow1, OutputArray flow2)
{
    Mat frame0 = getCpuMat(_frame0, buf0);
    Mat frame1 = getCpuMat(_frame1, buf1);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    Mat input0 = convertToType(frame0, CV_8U, 1, buf2, buf3);
    Mat input1 = convertToType(frame1, CV_8U, 1, buf4, buf5);

    calcOpticalFlowFarneback(input0, input1, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);

    if (!flow2.needed())
        ::set(flow1, flow);
    else
    {
        split(flow, flows);

        flows[0].copyTo(flow1);
        flows[1].copyTo(flow2);
    }
}

///////////////////////////////////////////////////////////////////
// SimpleOpticalFlow

cv::superres::SimpleOpticalFlow::SimpleOpticalFlow()
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

void cv::superres::SimpleOpticalFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray flow1, OutputArray flow2)
{
    Mat frame0 = getCpuMat(_frame0, buf0);
    Mat frame1 = getCpuMat(_frame1, buf1);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    Mat input0 = convertToType(frame0, CV_8U, 3, buf2, buf3);
    Mat input1 = convertToType(frame1, CV_8U, 3, buf4, buf5);

    calcOpticalFlowSF(input0, input1, flow,
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

    if (!flow2.needed())
        ::set(flow1, flow);
    else
    {
        split(flow, flows);

        flows[0].copyTo(flow1);
        flows[1].copyTo(flow2);
    }
}

///////////////////////////////////////////////////////////////////
// BroxOpticalFlow_GPU

cv::superres::BroxOpticalFlow_GPU::BroxOpticalFlow_GPU() : alg(0.197, 50.0, 0.8, 10, 77, 10)
{
    alpha = alg.alpha;
    gamma = alg.gamma;
    scaleFactor = alg.scale_factor;
    innerIterations = alg.inner_iterations;
    outerIterations = alg.outer_iterations;
    solverIterations = alg.solver_iterations;
}

void cv::superres::BroxOpticalFlow_GPU::calc(InputArray _frame0, InputArray _frame1, OutputArray flow1, OutputArray flow2)
{
    GpuMat frame0 = getGpuMat(_frame0, buf0);
    GpuMat frame1 = getGpuMat(_frame1, buf1);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    GpuMat input0 = convertToType(frame0, CV_32F, 1, buf2, buf3);
    GpuMat input1 = convertToType(frame1, CV_32F, 1, buf4, buf5);

    alg.alpha = alpha;
    alg.gamma = gamma;
    alg.scale_factor = scaleFactor;
    alg.inner_iterations = innerIterations;
    alg.outer_iterations = outerIterations;
    alg.solver_iterations = solverIterations;
    alg(input0, input1, u, v);

    if (flow2.needed())
    {
        ::set(flow1, u);
        ::set(flow2, v);
    }
    else
    {
        GpuMat src[] = {u, v};
        gpu::merge(src, 2, flow);
        ::set(flow1, flow);
    }
}

///////////////////////////////////////////////////////////////////
// PyrLKOpticalFlow_GPU

cv::superres::PyrLKOpticalFlow_GPU::PyrLKOpticalFlow_GPU()
{
    winSize = alg.winSize.width;
    maxLevel = alg.maxLevel;
    iterations = alg.iters;
}

void cv::superres::PyrLKOpticalFlow_GPU::calc(InputArray _frame0, InputArray _frame1, OutputArray flow1, OutputArray flow2)
{
    GpuMat frame0 = getGpuMat(_frame0, buf0);
    GpuMat frame1 = getGpuMat(_frame1, buf1);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    GpuMat input0 = convertToType(frame0, CV_8U, 1, buf2, buf3);
    GpuMat input1 = convertToType(frame1, CV_8U, 1, buf4, buf5);

    alg.winSize.width = winSize;
    alg.winSize.height = winSize;
    alg.maxLevel = maxLevel;
    alg.iters = iterations;
    alg.dense(input0, input1, u, v);

    if (flow2.needed())
    {
        ::set(flow1, u);
        ::set(flow2, v);
    }
    else
    {
        GpuMat src[] = {u, v};
        gpu::merge(src, 2, flow);
        ::set(flow1, flow);
    }
}

///////////////////////////////////////////////////////////////////
// FarnebackOpticalFlow_GPU

cv::superres::FarnebackOpticalFlow_GPU::FarnebackOpticalFlow_GPU()
{
    pyrScale = alg.pyrScale;
    numLevels = alg.numLevels;
    winSize = alg.winSize;
    numIters = alg.numIters;
    polyN = alg.polyN;
    polySigma = alg.polySigma;
    flags = alg.flags;
}

void cv::superres::FarnebackOpticalFlow_GPU::calc(InputArray _frame0, InputArray _frame1, OutputArray flow1, OutputArray flow2)
{
    GpuMat frame0 = getGpuMat(_frame0, buf0);
    GpuMat frame1 = getGpuMat(_frame1, buf1);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    GpuMat input0 = convertToType(frame0, CV_8U, 1, buf2, buf3);
    GpuMat input1 = convertToType(frame1, CV_8U, 1, buf4, buf5);

    alg.pyrScale = pyrScale;
    alg.numLevels = numLevels;
    alg.winSize = winSize;
    alg.numIters = numIters;
    alg.polyN = polyN;
    alg.polySigma = polySigma;
    alg.flags = flags;
    alg(input0, input1, u, v);

    if (flow2.needed())
    {
        ::set(flow1, u);
        ::set(flow2, v);
    }
    else
    {
        GpuMat src[] = {u, v};
        gpu::merge(src, 2, flow);
        ::set(flow1, flow);
    }
}
