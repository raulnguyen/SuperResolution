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

#include "motion_estimation.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;
using namespace cv::videostab;
using namespace cv::superres;
using namespace cv::gpu;

namespace
{
    Mat getCpuMat(InputArray m, Mat& buf)
    {
        if (m.kind() == _InputArray::GPU_MAT)
        {
            m.getGpuMat().download(buf);
            return buf;
        }

        return m.getMat();
    }

    void setCpuMat(const Mat& src, OutputArray dst)
    {
        if (dst.kind() == _InputArray::GPU_MAT)
            dst.getGpuMatRef().upload(src);

        src.copyTo(dst);
    }

    GpuMat getGpuMat(InputArray m, GpuMat& buf)
    {
        if (m.kind() == _InputArray::GPU_MAT)
            return m.getGpuMat();

        Mat h_m = m.getMat();

        ensureSizeIsEnough(h_m.size(), h_m.type(), buf);
        buf.upload(h_m);

        return buf;
    }

    void setGpuMat(const GpuMat& src, OutputArray dst)
    {
        if (dst.kind() == _InputArray::GPU_MAT)
            src.copyTo(dst.getGpuMatRef());

        dst.create(src.size(), src.type());

        Mat h_m = dst.getMat();
        src.download(h_m);
    }

    //////////////////////

    class GlobalMotionEstimator : public MotionEstimator
    {
    public:
        explicit GlobalMotionEstimator(MotionModel model);

        bool estimate(InputArray frame0, InputArray frame1, OutputArray m1, OutputArray m2);

    private:
        Ptr<ImageMotionEstimatorBase> motionEstimator;
        Mat h_frame0;
        Mat h_frame1;
    };

    GlobalMotionEstimator::GlobalMotionEstimator(MotionModel model)
    {
        Ptr<MotionEstimatorBase> baseEstimator(new MotionEstimatorRansacL2(model));
        motionEstimator = new KeypointBasedMotionEstimator(baseEstimator);
    }

    bool GlobalMotionEstimator::estimate(InputArray _frame0, InputArray _frame1, OutputArray m1, OutputArray m2)
    {
        Mat frame0 = getCpuMat(_frame0, h_frame0);
        Mat frame1 = getCpuMat(_frame1, h_frame1);

        CV_DbgAssert(frame0.size() == frame1.size());
        CV_DbgAssert(frame0.type() == frame1.type());

        bool ok;
        Mat M = motionEstimator->estimate(frame0, frame1, &ok);

        setCpuMat(M, m1);
        m2.release();

        return ok;
    }

    //////////////////////

    class GlobalMotionEstimator_GPU : public MotionEstimator
    {
    public:
        explicit GlobalMotionEstimator_GPU(MotionModel model);

        bool estimate(InputArray frame0, InputArray frame1, OutputArray m1, OutputArray m2);

    private:
        KeypointBasedMotionEstimatorGpu motionEstimator;
        GpuMat d_frame0;
        GpuMat d_frame1;
    };

    GlobalMotionEstimator_GPU::GlobalMotionEstimator_GPU(MotionModel model) : motionEstimator(new MotionEstimatorRansacL2(model))
    {
    }

    bool GlobalMotionEstimator_GPU::estimate(InputArray _frame0, InputArray _frame1, OutputArray m1, OutputArray m2)
    {
        GpuMat frame0 = getGpuMat(_frame0, d_frame0);
        GpuMat frame1 = getGpuMat(_frame1, d_frame1);

        CV_DbgAssert(frame0.size() == frame1.size());
        CV_DbgAssert(frame0.type() == frame1.type());

        bool ok;
        Mat M = motionEstimator.estimate(frame0, frame1, &ok);

        setCpuMat(M, m1);
        m2.release();

        return ok;
    }

    //////////////////////

    class GeneralMotionEstimator : public MotionEstimator
    {
    public:
        bool estimate(InputArray frame0, InputArray frame1, OutputArray m1, OutputArray m2);

    private:
        Mat h_frame0;
        Mat h_frame1;
        Mat frame0gray;
        Mat frame1gray;
        Mat flow;
        vector<Mat> ch;
    };

    bool GeneralMotionEstimator::estimate(InputArray _frame0, InputArray _frame1, OutputArray m1, OutputArray m2)
    {
        Mat frame0 = getCpuMat(_frame0, h_frame0);
        Mat frame1 = getCpuMat(_frame1, h_frame1);

        CV_DbgAssert(frame0.size() == frame1.size());
        CV_DbgAssert(frame0.type() == frame1.type());

        Mat input0, input1;

        if (frame0.channels() == 1)
        {
            input0 = frame0;
            input1 = frame1;
        }
        else if (frame0.channels() == 3)
        {
            cvtColor(frame0, frame0gray, COLOR_BGR2GRAY);
            cvtColor(frame1, frame1gray, COLOR_BGR2GRAY);
            input0 = frame0gray;
            input1 = frame1gray;
        }
        else
        {
            cvtColor(frame0, frame0gray, COLOR_BGRA2GRAY);
            cvtColor(frame1, frame1gray, COLOR_BGRA2GRAY);
            input0 = frame0gray;
            input1 = frame1gray;
        }

        calcOpticalFlowFarneback(input0, input1, flow,
                                 /*pyrScale =*/ 0.5,
                                 /*numLevels =*/ 5,
                                 /*winSize =*/ 13,
                                 /*numIters =*/ 10,
                                 /*polyN =*/ 5,
                                 /*polySigma =*/ 1.1,
                                 /*flags =*/ 0);

        split(flow, ch);

        setCpuMat(ch[0], m1);
        setCpuMat(ch[1], m2);

        return true;
    }

    //////////////////////

    class GeneralMotionEstimator_GPU : public MotionEstimator
    {
    public:
        bool estimate(InputArray frame0, InputArray frame1, OutputArray m1, OutputArray m2);

    private:
        GpuMat d_frame0;
        GpuMat d_frame1;
        GpuMat frame0gray;
        GpuMat frame1gray;
        FarnebackOpticalFlow flow;
        GpuMat flowx;
        GpuMat flowy;
    };

    bool GeneralMotionEstimator_GPU::estimate(InputArray _frame0, InputArray _frame1, OutputArray m1, OutputArray m2)
    {
        GpuMat frame0 = getGpuMat(_frame0, d_frame0);
        GpuMat frame1 = getGpuMat(_frame1, d_frame1);

        CV_DbgAssert(frame0.size() == frame1.size());
        CV_DbgAssert(frame0.type() == frame1.type());

        GpuMat input0, input1;

        if (frame0.channels() == 1)
        {
            input0 = frame0;
            input1 = frame1;
        }
        else if (frame0.channels() == 3)
        {
            cvtColor(frame0, frame0gray, COLOR_BGR2GRAY);
            cvtColor(frame1, frame1gray, COLOR_BGR2GRAY);
            input0 = frame0gray;
            input1 = frame1gray;
        }
        else
        {
            cvtColor(frame0, frame0gray, COLOR_BGRA2GRAY);
            cvtColor(frame1, frame1gray, COLOR_BGRA2GRAY);
            input0 = frame0gray;
            input1 = frame1gray;
        }

        flow(input0, input1, flowx, flowy);

        setGpuMat(flowx, m1);
        setGpuMat(flowy, m2);

        return true;
    }
}

Ptr<MotionEstimator> MotionEstimator::create(MotionModel model, bool useGpu)
{
    if (useGpu)
    {
        if (model <= MM_HOMOGRAPHY)
            return Ptr<MotionEstimator>(new GlobalMotionEstimator_GPU(model));

        return Ptr<MotionEstimator>(new GeneralMotionEstimator_GPU);
    }

    if (model <= MM_HOMOGRAPHY)
        return Ptr<MotionEstimator>(new GlobalMotionEstimator(model));

    return Ptr<MotionEstimator>(new GeneralMotionEstimator);
}

MotionEstimator::~MotionEstimator()
{
}
