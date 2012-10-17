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
#include <opencv2/video/tracking.hpp>
#include "input_array_utility.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::superres;

void cv::superres::DenseOpticalFlow::collectGarbage()
{
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

void cv::superres::FarnebackOpticalFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
{
    Mat frame0 = ::getMat(_frame0, buf[0]);
    Mat frame1 = ::getMat(_frame1, buf[1]);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    Mat input0 = ::convertToType(frame0, CV_8U, 1, buf[2], buf[3]);
    Mat input1 = ::convertToType(frame1, CV_8U, 1, buf[4], buf[5]);

    if (!_flow2.needed() && _flow1.kind() != _InputArray::GPU_MAT)
    {
        call(input0, input1, _flow1);
        return;
    }

    call(input0, input1, flow);

    if (!_flow2.needed())
    {
        ::copy(_flow1, flow);
    }
    else
    {
        split(flow, flows);

        ::copy(_flow1, flows[0]);
        ::copy(_flow2, flows[1]);
    }
}

void cv::superres::FarnebackOpticalFlow::call(const Mat& input0, const Mat& input1, OutputArray dst)
{
    calcOpticalFlowFarneback(input0, input1, dst, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);
}

void cv::superres::FarnebackOpticalFlow::collectGarbage()
{
    for (int i = 0; i < 6; ++i)
        buf[i].release();
    flow.release();
    flows.clear();
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

void cv::superres::SimpleOpticalFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
{
    Mat frame0 = ::getMat(_frame0, buf[0]);
    Mat frame1 = ::getMat(_frame1, buf[1]);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    Mat input0 = ::convertToType(frame0, CV_8U, 3, buf[2], buf[3]);
    Mat input1 = ::convertToType(frame1, CV_8U, 3, buf[4], buf[5]);

    if (!_flow2.needed() && _flow1.kind() == _InputArray::MAT)
    {
        call(input0, input1, _flow1.getMatRef());
        return;
    }

    call(input0, input1, flow);

    if (!_flow2.needed())
    {
        ::copy(_flow1, flow);
    }
    else
    {
        split(flow, flows);

        ::copy(_flow1, flows[0]);
        ::copy(_flow2, flows[1]);
    }
}

void cv::superres::SimpleOpticalFlow::call(Mat input0, Mat input1, Mat& dst)
{
    calcOpticalFlowSF(input0, input1, dst,
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

void cv::superres::SimpleOpticalFlow::collectGarbage()
{
    for (int i = 0; i < 6; ++i)
        buf[i].release();
    flow.release();
    flows.clear();
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

void cv::superres::BroxOpticalFlow_GPU::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
{
    GpuMat frame0 = ::getGpuMat(_frame0, buf[0]);
    GpuMat frame1 = ::getGpuMat(_frame1, buf[1]);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    GpuMat input0 = ::convertToType(frame0, CV_32F, 1, buf[2], buf[3]);
    GpuMat input1 = ::convertToType(frame1, CV_32F, 1, buf[4], buf[5]);

    if (_flow2.needed() && _flow1.kind() == _InputArray::GPU_MAT && _flow2.kind() == _InputArray::GPU_MAT)
    {
        call(input0, input1, _flow1.getGpuMatRef(), _flow2.getGpuMatRef());
        return;
    }

    call(input0, input1, u, v);

    if (_flow2.needed())
    {
        ::copy(_flow1, u);
        ::copy(_flow2, v);
    }
    else
    {
        GpuMat src[] = {u, v};
        gpu::merge(src, 2, flow);
        ::copy(_flow1, flow);
    }
}

void cv::superres::BroxOpticalFlow_GPU::call(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
{
    alg.alpha = alpha;
    alg.gamma = gamma;
    alg.scale_factor = scaleFactor;
    alg.inner_iterations = innerIterations;
    alg.outer_iterations = outerIterations;
    alg.solver_iterations = solverIterations;
    alg(input0, input1, dst1, dst2);
}

void cv::superres::BroxOpticalFlow_GPU::collectGarbage()
{
    alg.buf.release();
    for (int i = 0; i < 6; ++i)
        buf[i].release();
    u.release();
    v.release();
    flow.release();
}

///////////////////////////////////////////////////////////////////
// PyrLKOpticalFlow_GPU

cv::superres::PyrLKOpticalFlow_GPU::PyrLKOpticalFlow_GPU()
{
    winSize = alg.winSize.width;
    maxLevel = alg.maxLevel;
    iterations = alg.iters;
}

void cv::superres::PyrLKOpticalFlow_GPU::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
{
    GpuMat frame0 = ::getGpuMat(_frame0, buf[0]);
    GpuMat frame1 = ::getGpuMat(_frame1, buf[1]);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    GpuMat input0 = ::convertToType(frame0, CV_8U, 1, buf[2], buf[3]);
    GpuMat input1 = ::convertToType(frame1, CV_8U, 1, buf[4], buf[5]);

    if (_flow2.needed() && _flow1.kind() == _InputArray::GPU_MAT && _flow2.kind() == _InputArray::GPU_MAT)
    {
        call(input0, input1, _flow1.getGpuMatRef(), _flow2.getGpuMatRef());
        return;
    }

    call(input0, input1, u, v);

    if (_flow2.needed())
    {
        ::copy(_flow1, u);
        ::copy(_flow2, v);
    }
    else
    {
        GpuMat src[] = {u, v};
        gpu::merge(src, 2, flow);
        ::copy(_flow1, flow);
    }
}

void cv::superres::PyrLKOpticalFlow_GPU::call(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
{
    alg.winSize.width = winSize;
    alg.winSize.height = winSize;
    alg.maxLevel = maxLevel;
    alg.iters = iterations;
    alg.dense(input0, input1, dst1, dst2);
}

void cv::superres::PyrLKOpticalFlow_GPU::collectGarbage()
{
    alg.releaseMemory();
    for (int i = 0; i < 6; ++i)
        buf[i].release();
    u.release();
    v.release();
    flow.release();
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

void cv::superres::FarnebackOpticalFlow_GPU::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
{
    GpuMat frame0 = ::getGpuMat(_frame0, buf[0]);
    GpuMat frame1 = ::getGpuMat(_frame1, buf[1]);

    CV_DbgAssert( frame1.type() == frame0.type() );
    CV_DbgAssert( frame1.size() == frame0.size() );

    GpuMat input0 = ::convertToType(frame0, CV_8U, 1, buf[2], buf[3]);
    GpuMat input1 = ::convertToType(frame1, CV_8U, 1, buf[4], buf[5]);

    if (_flow2.needed() && _flow1.kind() == _InputArray::GPU_MAT && _flow2.kind() == _InputArray::GPU_MAT)
    {
        call(input0, input1, _flow1.getGpuMatRef(), _flow2.getGpuMatRef());
        return;
    }

    call(input0, input1, u, v);

    if (_flow2.needed())
    {
        ::copy(_flow1, u);
        ::copy(_flow2, v);
    }
    else
    {
        GpuMat src[] = {u, v};
        gpu::merge(src, 2, flow);
        ::copy(_flow1, flow);
    }
}

void cv::superres::FarnebackOpticalFlow_GPU::call(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
{
    alg.pyrScale = pyrScale;
    alg.numLevels = numLevels;
    alg.winSize = winSize;
    alg.numIters = numIters;
    alg.polyN = polyN;
    alg.polySigma = polySigma;
    alg.flags = flags;
    alg(input0, input1, dst1, dst2);
}

void cv::superres::FarnebackOpticalFlow_GPU::collectGarbage()
{
    alg.releaseMemory();
    for (int i = 0; i < 6; ++i)
        buf[i].release();
    u.release();
    v.release();
    flow.release();
}
