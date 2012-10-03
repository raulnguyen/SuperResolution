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

using namespace std;
using namespace cv;
using namespace cv::videostab;
using namespace cv::superres;

namespace
{
    class GlobalMotionEstimator : public MotionEstimator
    {
    public:
        explicit GlobalMotionEstimator(MotionModel model);

        bool estimate(InputArray frame0, InputArray frame1, OutputArray m1, OutputArray m2);

    private:
        Ptr<ImageMotionEstimatorBase> motionEstimator;
    };

    GlobalMotionEstimator::GlobalMotionEstimator(MotionModel model)
    {
        Ptr<MotionEstimatorBase> baseEstimator(new MotionEstimatorRansacL2(model));
        motionEstimator = new KeypointBasedMotionEstimator(baseEstimator);
    }

    bool GlobalMotionEstimator::estimate(InputArray frame0, InputArray frame1, OutputArray m1, OutputArray m2)
    {
        bool ok;
        Mat M = motionEstimator->estimate(frame0.getMat(), frame1.getMat(), &ok);

        if (motionEstimator->motionModel() == MM_HOMOGRAPHY)
            M.copyTo(m1);
        else
            M(Rect(0, 0, 3, 2)).copyTo(m1);

        m2.release();

        return ok;
    }

    class GeneralMotionEstimator : public MotionEstimator
    {
    public:
        bool estimate(InputArray frame0, InputArray frame1, OutputArray m1, OutputArray m2);

    private:
        Mat frame0gray;
        Mat frame1gray;
        Mat flow;
        vector<Mat> ch;
    };

    bool GeneralMotionEstimator::estimate(InputArray _frame0, InputArray _frame1, OutputArray m1, OutputArray m2)
    {
        Mat frame0 = _frame0.getMat();
        Mat frame1 = _frame1.getMat();

        CV_DbgAssert(frame0.size() == frame1.size());
        CV_DbgAssert(frame0.type() == frame1.type());

        if (frame0.channels() == 1)
        {
            frame0.copyTo(frame0gray);
            frame1.copyTo(frame1gray);
        }
        else if (frame0.channels() == 3)
        {
            cvtColor(frame0, frame0gray, COLOR_BGR2GRAY);
            cvtColor(frame1, frame1gray, COLOR_BGR2GRAY);
        }
        else
        {
            cvtColor(frame0, frame0gray, COLOR_BGRA2GRAY);
            cvtColor(frame1, frame1gray, COLOR_BGRA2GRAY);
        }

        calcOpticalFlowFarneback(frame0gray, frame1gray, flow,
                                 /*pyrScale =*/ 0.5,
                                 /*numLevels =*/ 5,
                                 /*winSize =*/ 13,
                                 /*numIters =*/ 10,
                                 /*polyN =*/ 5,
                                 /*polySigma =*/ 1.1,
                                 /*flags =*/ 0);

        split(flow, ch);

        ch[0].copyTo(m1);
        ch[1].copyTo(m2);

        return true;
    }
}

Ptr<MotionEstimator> MotionEstimator::create(MotionModel model)
{
    if (model <= MM_HOMOGRAPHY)
        return Ptr<MotionEstimator>(new GlobalMotionEstimator(model));

    return Ptr<MotionEstimator>(new GeneralMotionEstimator);
}

MotionEstimator::~MotionEstimator()
{
}
