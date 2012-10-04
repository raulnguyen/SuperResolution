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

#include "nlm.hpp"
#include <opencv2/core/internal.hpp>
#include <opencv2/videostab/ring_buffer.hpp>
#include "extract_patch.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(Nlm, "VideoSuperResolution.Nlm",
                          obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0,
                                               "Scale factor.");
                          obj.info()->addParam(obj, "searchWindowRadius", obj.searchWindowRadius, false, 0, 0,
                                               "Radius of the patch search window in low resolution image.");
                          obj.info()->addParam(obj, "temporalAreaRadius", obj.temporalAreaRadius, false, 0, 0,
                                               "Radius of the temporal search area.");
                          obj.info()->addParam(obj, "patchRadius", obj.patchRadius, false, 0, 0,
                                               "Radius of the patch.");
                          obj.info()->addParam(obj, "sigma", obj.sigma, false, 0, 0,
                                               "Weight of the patch difference");
                          obj.info()->addParam(obj, "doDeblurring", obj.doDeblurring, false, 0, 0,
                                               "Perform deblurring operation"));
    }
}

bool cv::superres::Nlm::init()
{
    return !Nlm_info_auto.name().empty();
}

Ptr<VideoSuperResolution> cv::superres::Nlm::create()
{
    return Ptr<VideoSuperResolution>(new Nlm);
}

cv::superres::Nlm::Nlm()
{
    scale = 2;
    searchWindowRadius = 10;
    temporalAreaRadius = 2;
    patchRadius = 3;
    sigma = 3;
    doDeblurring = true;

    Ptr<MotionEstimatorBase> baseEstimator(new MotionEstimatorRansacL2);
    motionEstimator = new KeypointBasedMotionEstimator(baseEstimator);
    deblurer = new WeightingDeblurer;

    deblurer->setFrames(outFrames);
    deblurer->setMotions(motions);
    deblurer->setBlurrinessRates(blurrinessRates);
}

namespace
{
    Point2d calcNlmWeight(const Mat_<Vec3b>& patch1, const Mat_<Vec3b>& patch2, double patchDiffWeight)
    {
        CV_DbgAssert(patch1.size() == patch2.size());

        Point2d patchDiff(0, 0);
        for (int y = 0; y < patch1.rows; ++y)
        {
            for (int x = 0; x < patch1.cols; ++x)
            {
                Vec3b val1 = patch1(y, x);
                Vec3b val2 = patch2(y, x);

                Point3d val1d(val1[0], val1[1], val1[2]);
                Point3d val2d(val2[0], val2[1], val2[2]);

                Point3d diff = val1d - val2d;

                patchDiff.x += diff.x * diff.x;
                patchDiff.y += diff.y * diff.y + diff.z * diff.z;
            }
        }

        patchDiff.x = exp(-patchDiff.x * patchDiffWeight);
        patchDiff.y = exp(-patchDiff.y * patchDiffWeight / 2);

        return patchDiff;
    }

    struct ProcessBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        int scale;
        int searchWindowRadius;
        int temporalAreaRadius;
        int patchRadius;
        double sigma;

        int procPos;

        vector<Mat>* y;
        vector<Mat>* Y;

        mutable Mat_<Point3d> V;
        mutable Mat_<Point3d> W;
    };

    void ProcessBody::operator ()(const Range& range) const
    {
        const int patchSize = 2 * patchRadius + 1;
        const double patchDiffWeight = 1.0 / (2.0 * sigma * sigma * patchSize * patchSize);

        const Mat& Z = at(procPos, *Y);

        Point Z_loc;
        for (Z_loc.y = range.start; Z_loc.y < range.end; ++Z_loc.y)
        {
            for (Z_loc.x = patchRadius; Z_loc.x < Z.cols - patchRadius; ++Z_loc.x)
            {
                const Mat_<Vec3b> Z_patch = extractPatch(Z, Z_loc, patchRadius);

                for (int t = -temporalAreaRadius; t <= temporalAreaRadius; ++t)
                {
                    const Mat& Yt = at(procPos + t, *Y);
                    const Mat& yt = at(procPos + t, *y);

                    Point yt_loc;
                    Point Yt_loc;
                    for (int i = -searchWindowRadius; i <= searchWindowRadius; ++i)
                    {
                        yt_loc.y = Z_loc.y / scale + i;
                        Yt_loc.y = yt_loc.y * scale;

                        if (yt_loc.y < 0 || yt_loc.y >= yt.rows || Yt_loc.y - patchRadius < 0 || Yt_loc.y + patchRadius >= Yt.rows)
                            continue;

                        for (int j = -searchWindowRadius; j <= searchWindowRadius; ++j)
                        {
                            yt_loc.x = Z_loc.x / scale + j;
                            Yt_loc.x = yt_loc.x * scale;

                            if (yt_loc.x < 0 || yt_loc.x >= yt.cols || Yt_loc.x - patchRadius < 0 || Yt_loc.x + patchRadius >= Yt.cols)
                                continue;

                            const Mat_<Vec3b> Yt_patch = extractPatch(Yt, Yt_loc, patchRadius);

                            const Point2d w = calcNlmWeight(Z_patch, Yt_patch, patchDiffWeight);

                            Vec3b val = yt.at<Vec3b>(yt_loc);
                            V(Z_loc) += Point3d(w.x * val[0], w.y * val[1], w.y * val[2]);
                            W(Z_loc) += Point3d(w.x, w.y, w.y);
                        }
                    }
                }
            }
        }
    }
}

void cv::superres::Nlm::initImpl(Ptr<IFrameSource>& frameSource)
{
    deblurer->setRadius(temporalAreaRadius);

    const int cacheSize = 3 * temporalAreaRadius + 2;

    y.resize(cacheSize);
    Y.resize(cacheSize);
    outFrames.resize(cacheSize);
    motions.resize(cacheSize);
    blurrinessRates.resize(cacheSize);

    storePos = -1;
    procPos = storePos - 2 * temporalAreaRadius;
    outPos = procPos - temporalAreaRadius - 1;

    for (int t = -temporalAreaRadius; t <= 2 * temporalAreaRadius; ++t)
    {
        Mat frame = frameSource->nextFrame();

        CV_Assert(!frame.empty());

        addNewFrame(frame);

        if (doDeblurring)
        {
            at(storePos, blurrinessRates) = calcBlurriness(at(storePos, outFrames));
            if (storePos > 0)
                at(storePos - 1, motions) = motionEstimator->estimate(at(storePos - 1, outFrames), at(storePos, outFrames));
        }
    }

    processFrame(procPos);
    for (int t = 1; t <= temporalAreaRadius; ++t)
        processFrame(procPos + t);
    processFrame(procPos);

    if (doDeblurring)
    {
        for (int t = -1; t <= temporalAreaRadius; ++t)
        {
            at(procPos + t, motions) = motionEstimator->estimate(at(procPos + t, outFrames), at(procPos + t + 1, outFrames));
            at(procPos + t, blurrinessRates) = calcBlurriness(at(procPos + t, outFrames));
        }
    }
}

Mat cv::superres::Nlm::processImpl(const Mat& frame)
{
    addNewFrame(frame);

    if (doDeblurring)
    {
        at(storePos, blurrinessRates) = calcBlurriness(at(storePos, outFrames));
        at(storePos - 1, motions) = motionEstimator->estimate(at(storePos - 1, Y), at(storePos, outFrames));
    }

    processFrame(procPos + temporalAreaRadius);
    processFrame(procPos);

    if (doDeblurring)
    {
        at(procPos - 1, motions) = motionEstimator->estimate(at(procPos - 1, outFrames), at(procPos, outFrames));
        at(procPos, motions) = motionEstimator->estimate(at(procPos, outFrames), at(procPos + 1, outFrames));

        at(procPos + temporalAreaRadius - 1, motions) = motionEstimator->estimate(at(procPos + temporalAreaRadius - 1, outFrames), at(procPos + temporalAreaRadius, outFrames));

        at(procPos, blurrinessRates) = calcBlurriness(at(procPos, outFrames));
        at(procPos + temporalAreaRadius, blurrinessRates) = calcBlurriness(at(procPos + temporalAreaRadius, outFrames));

        deblurer->deblur(procPos, at(procPos, outFrames));
    }

    return at(outPos, outFrames);
}

void cv::superres::Nlm::processFrame(int idx)
{
    at(idx, Y).convertTo(V, V.depth());
    W.create(V.size());
    W.setTo(Scalar::all(1));

    ProcessBody body;

    body.scale = scale;
    body.searchWindowRadius = searchWindowRadius;
    body.temporalAreaRadius = temporalAreaRadius;
    body.patchRadius = patchRadius;
    body.sigma = sigma;

    body.procPos = idx;

    body.y = &y;
    body.Y = &Y;

    body.V = V;
    body.W = W;

    parallel_for_(Range(patchRadius, V.rows - patchRadius), body);

    divide(V, W, Z);

    Z.convertTo(at(idx, Y), CV_8U);

    cvtColor(at(idx, Y), at(idx, outFrames), COLOR_Lab2LBGR);
}

void cv::superres::Nlm::addNewFrame(const Mat& frame)
{
    CV_DbgAssert(frame.type() == CV_8UC3);
    CV_DbgAssert(storePos < 0 || frame.size() == at(storePos, y).size());

    ++storePos;
    ++procPos;
    ++outPos;

    resize(frame, buf, Size(), scale, scale, INTER_CUBIC);

    cvtColor(frame, at(storePos, y), COLOR_LBGR2Lab);
    cvtColor(buf, at(storePos, Y), COLOR_LBGR2Lab);
    buf.copyTo(at(storePos, outFrames));
}
