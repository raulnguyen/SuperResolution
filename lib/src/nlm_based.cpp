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

#include "nlm_based.hpp"
#include <opencv2/core/internal.hpp>
#include "extract_patch.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

CV_INIT_ALGORITHM(NlmBased, "VideoSuperResolution.NlmBased",
                  obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0,
                                       "Scale factor.");
                  obj.info()->addParam(obj, "searchAreaRadius", obj.searchAreaRadius, false, 0, 0,
                                       "Radius of the patch search area in low resolution image.");
                  obj.info()->addParam(obj, "timeRadius", obj.timeRadius, false, 0, 0,
                                       "Radius of the time search area.");
                  obj.info()->addParam(obj, "patchRadius", obj.patchRadius, false, 0, 0,
                                       "Radius of the patch.");
                  obj.info()->addParam(obj, "sigma", obj.sigma, false, 0, 0,
                                       "Weight of the patch difference"));

bool NlmBased::init()
{
    return !NlmBased_info_auto.name().empty();
}

Ptr<VideoSuperResolution> NlmBased::create()
{
    return Ptr<VideoSuperResolution>(new NlmBased);
}

NlmBased::NlmBased()
{
    scale = 2;
    searchAreaRadius = 10;
    timeRadius = 2;
    patchRadius = 3;
    sigma = 3;
}

namespace
{
    double calcNlmWeight(const Mat_<Vec3b>& patch1, const Mat_<Vec3b>& patch2, double patchDiffWeight)
    {
        CV_DbgAssert(patch1.size() == patch2.size());

        double patchDiff = 0.0;
        for (int y = 0; y < patch1.rows; ++y)
        {
            for (int x = 0; x < patch1.cols; ++x)
            {
                Vec3b val1 = patch1(y, x);
                Vec3b val2 = patch2(y, x);

                Point3d val1d(val1[0], val1[1], val1[2]);
                Point3d val2d(val2[0], val2[1], val2[2]);

                Point3d diff = val1d - val2d;

                patchDiff += diff.ddot(diff);
            }
        }

        return exp(-patchDiff * patchDiffWeight);
    }

    struct LoopBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        int scale;
        int searchAreaRadius;
        int timeRadius;
        int patchRadius;
        double sigma;

        int procPos;

        vector<Mat>* y;
        vector<Mat>* Y;

        mutable Mat_<Point3d> V;
        mutable Mat_<Point3d> W;
    };

    void LoopBody::operator ()(const Range& range) const
    {
        const double patchDiffWeight = 1.0 / (2.0 * sigma * sigma);

        const Mat& Z = at(procPos, *Y);

        Point Z_loc;
        for (Z_loc.y = range.start; Z_loc.y < range.end; ++Z_loc.y)
        {
            for (Z_loc.x = patchRadius; Z_loc.x < Z.cols - patchRadius; ++Z_loc.x)
            {
                const Mat_<Vec3b> Z_patch = extractPatch(Z, Z_loc, patchRadius);

                for (int t = -timeRadius; t <= timeRadius; ++t)
                {
                    const Mat& Yt = at(procPos + t, *Y);
                    const Mat& yt = at(procPos + t, *y);

                    Point yt_loc;
                    Point Yt_loc;
                    for (int i = -searchAreaRadius; i <= searchAreaRadius; ++i)
                    {
                        yt_loc.y = Z_loc.y / scale + i;
                        Yt_loc.y = yt_loc.y * scale;

                        if (yt_loc.y < 0 || yt_loc.y >= yt.rows || Yt_loc.y - patchRadius < 0 || Yt_loc.y + patchRadius >= Yt.rows)
                            continue;

                        for (int j = -searchAreaRadius; j <= searchAreaRadius; ++j)
                        {
                            yt_loc.x = Z_loc.x / scale + j;
                            Yt_loc.x = yt_loc.x * scale;

                            if (yt_loc.x < 0 || yt_loc.x >= yt.cols || Yt_loc.x - patchRadius < 0 || Yt_loc.x + patchRadius >= Yt.cols)
                                continue;

                            const Mat_<Vec3b> Yt_patch = extractPatch(Yt, Yt_loc, patchRadius);

                            const double w = calcNlmWeight(Z_patch, Yt_patch, patchDiffWeight);

                            if (w >= 0.1)
                            {
                                Vec3b val = yt.at<Vec3b>(yt_loc);
                                Point3d vald(val[0], val[1], val[2]);

                                V(Z_loc) += w * vald;
                                W(Z_loc) += Point3d(w, w, w);
                            }
                        }
                    }
                }
            }
        }
    }
}

void NlmBased::initImpl(cv::Ptr<IFrameSource>& frameSource)
{
    y.resize(3 * timeRadius + 2);
    Y.resize(3 * timeRadius + 2);

    storePos = -1;
    procPos = storePos - 2 * timeRadius;
    outPos = procPos - timeRadius - 1;

    for (int t = -timeRadius; t <= 2 * timeRadius; ++t)
    {
        Mat frame = frameSource->nextFrame();

        CV_Assert(!frame.empty());

        addNewFrame(frame);
    }

    processFrame(procPos);
    for (int t = 1; t <= timeRadius; ++t)
        processFrame(procPos + t);
    processFrame(procPos);
}

Mat NlmBased::processImpl(const Mat& frame)
{
    addNewFrame(frame);

    processFrame(procPos + timeRadius);
    processFrame(procPos);

    return at(outPos, Y);
}

void NlmBased::processFrame(int idx)
{
    Mat& procY = at(idx, Y);

    procY.convertTo(V, CV_64F);
    W.create(V.size());
    W.setTo(Scalar::all(1));

    LoopBody body;

    body.scale = scale;
    body.searchAreaRadius = searchAreaRadius;
    body.timeRadius = timeRadius;
    body.patchRadius = patchRadius;
    body.sigma = sigma;

    body.procPos = idx;

    body.y = &y;
    body.Y = &Y;

    body.V = V;
    body.W = W;

    parallel_for_(Range(patchRadius, V.rows - patchRadius), body);

    divide(V, W, Z);

    Z.convertTo(procY, CV_8U);
}

void NlmBased::addNewFrame(const cv::Mat& frame)
{
    CV_DbgAssert(frame.type() == CV_8UC3);
    CV_DbgAssert(storePos < 0 || frame.size() == at(storePos, y).size());

    ++storePos;
    ++procPos;
    ++outPos;

    frame.copyTo(at(storePos, y));
    resize(frame, at(storePos, Y), Size(), scale, scale, INTER_CUBIC);
}
