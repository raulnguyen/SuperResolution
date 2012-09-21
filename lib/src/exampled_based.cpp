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

#include "exampled_based.hpp"
#include <cstring>
#include <cmath>
#include <fstream>
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "extract_patch.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

CV_INIT_ALGORITHM(ExampledBased, "SuperResolution.ExampledBased",
                  obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0,
                                       "Scale factor.");
                  obj.info()->addParam(obj, "patchStep", obj.patchStep, false, 0, 0,
                                       "Step between patches in training.");
                  obj.info()->addParam(obj, "lowResPatchSize", obj.lowResPatchSize, false, 0, 0,
                                       "Size of low-resolution patches.");
                  obj.info()->addParam(obj, "highResPatchSize", obj.highResPatchSize, false, 0, 0,
                                       "Size of high-resolution patches.");
                  obj.info()->addParam(obj, "stdDevThresh", obj.stdDevThresh, false, 0, 0,
                                       "Threshold value for patch standard deviation, only patches with high deviation will be processed.");
                  obj.info()->addParam<DescriptorMatcher>(obj, "matcher", obj.matcher, false, 0, 0,
                                                          "Matching algorithm."));

bool ExampledBased::init()
{
    return !ExampledBased_info_auto.name().empty();
}

Ptr<ImageSuperResolution> ExampledBased::create()
{
    return Ptr<ImageSuperResolution>(new ExampledBased);
}

ExampledBased::ExampledBased()
{
    scale = 2.0;
    patchStep = 1.0;
    lowResPatchSize = 7;
    highResPatchSize = 5;
    stdDevThresh = 15.0;
    matcher = new BFMatcher(NORM_L2SQR);
}

void ExampledBased::train(const vector<Mat>& images)
{
    vector<Mat> lowResPatchesVec;
    vector<Mat> highResPatchesVec;

    int totalCount = 0;
    for (size_t i = 0; i < images.size(); ++i)
    {
        Mat low, high;
        buildPatchLists(images[i], low, high);

        if (!low.empty())
        {
            CV_DbgAssert(low.cols == lowResPatchesVec[0].cols);
            CV_DbgAssert(low.type() == lowResPatchesVec[0].type());
            CV_DbgAssert(high.cols == highResPatchesVec[0].cols);
            CV_DbgAssert(high.type() == highResPatchesVec[0].type());

            totalCount += low.rows;

            lowResPatchesVec.push_back(low);
            highResPatchesVec.push_back(high);
        }
    }

    if (totalCount == 0)
    {
        clear();
    }
    else
    {
        lowResPatches.create(totalCount, lowResPatchesVec[0].cols, lowResPatchesVec[0].type());
        highResPatches.create(totalCount, highResPatchesVec[0].cols, highResPatchesVec[0].type());

        int startRow = 0;
        for (size_t i = 0; i < lowResPatchesVec.size(); ++i)
        {
            const int count = lowResPatchesVec[i].rows;

            lowResPatchesVec[i].copyTo(lowResPatches.rowRange(startRow, startRow + count));
            highResPatchesVec[i].copyTo(highResPatches.rowRange(startRow, startRow + count));

            startRow += count;
        }
    }
}

void ExampledBased::train(const Mat& image)
{
    buildPatchLists(image, lowResPatches, highResPatches);
}

void ExampledBased::clear()
{
    lowResPatches.release();
    highResPatches.release();
}

void ExampledBased::process(const Mat& src, Mat& dst)
{
    CV_DbgAssert(scale > 1);
    CV_DbgAssert(lowResPatchSize > 0 && lowResPatchSize % 2 != 0);
    CV_DbgAssert(highResPatchSize > 0 && highResPatchSize % 2 != 0);
    CV_DbgAssert(!matcher.empty());

    if (lowResPatches.empty())
        train(src);

    const int cn = src.channels();

    const float alpha = static_cast<float>(0.1 * lowResPatchSize * lowResPatchSize / (2.0 * highResPatchSize - 1.0));

    const int highRad = highResPatchSize / 2;

    resize(src, dst, Size(), scale, scale, INTER_CUBIC);

    if (lowResPatches.empty())
        return;

    CV_DbgAssert(lowResPatches.rows == highResPatches.rows);
    CV_DbgAssert(lowResPatches.cols == (lowResPatchSize * lowResPatchSize + 2 * highResPatchSize - 1) * cn);
    CV_DbgAssert(highResPatches.cols == highResPatchSize * highResPatchSize * cn);

    Mat lowRes = src;
    Mat highRes = dst;
    vector<float> lowResPatch;
    vector<float> highResPatch;

    vector<DMatch> matches;

    Mat dstNew;

    const double weight = 1.0 / (highRad * highRad + 1.0);
    Mat result;
    dst.convertTo(result, CV_32F, weight);

    for (int i = 0; i < highRad; ++i)
    {
        for (int j = 0; j < highRad; ++j)
        {
            dst.convertTo(dstNew, CV_32F);

            Point pLow, pHigh;
            for (pHigh.y = i, pLow.y = pHigh.y / 2; pHigh.y < dst.rows + highRad; pHigh.y += highResPatchSize - 1, pLow.y = pHigh.y / 2)
            {
                for (pHigh.x = j, pLow.x = pHigh.x / 2; pHigh.x < dst.cols + highRad; pHigh.x += highResPatchSize - 1, pLow.x = pHigh.x / 2)
                {
                    extractPatch(lowRes, pLow, lowResPatch, lowResPatchSize);

                    Scalar mean, stddev;
                    meanStdDev(Mat(lowResPatchSize, lowResPatchSize, CV_32FC(cn), &lowResPatch[0]), mean, stddev);

                    if (stddev[0] < stdDevThresh && stddev[1] < stdDevThresh && stddev[2] < stdDevThresh && stddev[3] < stdDevThresh)
                        continue;

                    mean[0] += numeric_limits<double>::epsilon();
                    mean[1] += numeric_limits<double>::epsilon();
                    mean[2] += numeric_limits<double>::epsilon();
                    mean[3] += numeric_limits<double>::epsilon();

                    scalePatch(lowResPatch, lowResPatchSize, cn, mean);

                    // load top-left border from high res patch
                    extractPatch(highRes, pHigh, highResPatch, highResPatchSize);

                    scalePatch(highResPatch, highResPatchSize, cn, mean);

                    Mat_<float> patch(highResPatchSize, highResPatchSize * cn, &highResPatch[0]);
                    for (int k = 0; k < highResPatchSize * cn; ++k)
                        lowResPatch.push_back(alpha * patch(0, k));
                    for (int k = 1; k < highResPatchSize; ++k)
                        for (int c = 0; c < cn; ++c)
                            lowResPatch.push_back(alpha * patch(k, c));

                    // find nearest patch from low res training base
                    matcher->match(Mat(1, lowResPatch.size(), CV_32FC1, &lowResPatch[0]), lowResPatches, matches);

                    // copy appropriate high res patch to dst

                    Mat_<float> newPatch(highResPatchSize, highResPatchSize * cn, highResPatches.ptr<float>(matches[0].trainIdx));

                    for (int k = 0, ny = pHigh.y - highRad; k < highResPatchSize; ++k, ++ny)
                    {
                        if (ny >= 0 && ny < dst.rows)
                        {
                            for (int m = 0, nx = pHigh.x - highRad; m < highResPatchSize; ++m, ++nx)
                            {
                                if (nx >= 0 && nx < dst.cols)
                                {
                                    for (int c = 0; c < cn; ++c)
                                        dstNew.at<float>(ny, nx * cn + c) = static_cast<float>(mean[c] *  newPatch(k, m * cn + c));
                                }
                            }
                        }
                    }
                }
            }

            addWeighted(result, 1.0, dstNew, weight, 0.0, result, CV_32F);
        }
    }

    result.convertTo(dst, CV_8U);
}

void ExampledBased::buildPatchLists(const Mat& highRes, Mat& lowResPatches, Mat& highResPatches)
{
    CV_DbgAssert(patchStep > 0);

    const int cn = highRes.channels();

    Mat lowRes;
    resize(highRes, lowRes, Size(), 1.0 / scale, 1.0 / scale, INTER_CUBIC);

    const float alpha = static_cast<float>(0.1 * lowResPatchSize * lowResPatchSize / (2.0 * highResPatchSize - 1.0));

    const int lowRad = lowResPatchSize / 2;

    const int maxPatchNumber = cvCeil((lowRes.cols + lowRad) / patchStep) * cvCeil((lowRes.rows + lowRad) / patchStep);

    lowResPatches.create(maxPatchNumber, (lowResPatchSize * lowResPatchSize + 2 * highResPatchSize - 1) * cn, CV_32F);
    highResPatches.create(maxPatchNumber, highResPatchSize * highResPatchSize * cn, CV_32F);

    vector<float> lowResPatch;
    vector<float> highResPatch;

    int patchNum = 0;
    Point2d pLow, pHigh;
    for (pLow.y = 0, pHigh.y = pLow.y * 2; pLow.y < lowRes.rows + lowRad; pLow.y += patchStep, pHigh.y = pLow.y * 2)
    {
        for (pLow.x = 0, pHigh.x = pLow.x * 2; pLow.x < lowRes.cols + lowRad; pLow.x += patchStep, pHigh.x = pLow.x * 2)
        {
            extractPatch(lowRes, pLow, lowResPatch, lowResPatchSize, cv::INTER_LINEAR);

            Scalar mean, stddev;
            meanStdDev(Mat(lowResPatchSize, lowResPatchSize, CV_32FC(cn), &lowResPatch[0]), mean, stddev);

            if (stddev[0] < stdDevThresh && stddev[1] < stdDevThresh && stddev[2] < stdDevThresh && stddev[3] < stdDevThresh)
                continue;

            mean[0] += numeric_limits<double>::epsilon();
            mean[1] += numeric_limits<double>::epsilon();
            mean[2] += numeric_limits<double>::epsilon();
            mean[3] += numeric_limits<double>::epsilon();

            scalePatch(lowResPatch, lowResPatchSize, cn, mean);

            extractPatch(highRes, pHigh, highResPatch, highResPatchSize, cv::INTER_LINEAR);

            scalePatch(highResPatch, highResPatchSize, cn, mean);

            // load top-left border from high res patch
            Mat_<float> patch(highResPatchSize, highResPatchSize * cn, &highResPatch[0]);
            for (int i = 0; i < highResPatchSize * cn; ++i)
                lowResPatch.push_back(alpha * patch(0, i));
            for (int i = 1; i < highResPatchSize; ++i)
                for (int c = 0; c < cn; ++c)
                    lowResPatch.push_back(alpha * patch(i, c));

            memcpy(lowResPatches.ptr(patchNum), &lowResPatch[0], lowResPatch.size() * sizeof(float));
            memcpy(highResPatches.ptr(patchNum), &highResPatch[0], highResPatch.size() * sizeof(float));

            ++patchNum;

            if (patchNum == maxPatchNumber)
                return;
        }
    }

    if (patchNum == 0)
    {
        clear();
    }
    else
    {
        lowResPatches = lowResPatches.rowRange(0, patchNum);
        highResPatches = highResPatches.rowRange(0, patchNum);
    }
}

void ExampledBased::save(const string& fileName) const
{
    CV_DbgAssert(!empty());

    FileStorage fs(fileName, FileStorage::WRITE);

    cv::write(fs, "lowResPatchSize", lowResPatchSize);
    cv::write(fs, "highResPatchSize", highResPatchSize);

    cv::write(fs, "lowResPatches", lowResPatches);
    cv::write(fs, "highResPatches", highResPatches);
}

void ExampledBased::load(const string& fileName)
{
    FileStorage fs(fileName, FileStorage::READ);

    cv::read(fs["lowResPatchSize"], lowResPatchSize, 0);
    cv::read(fs["highResPatchSize"], highResPatchSize, 0);

    cv::read(fs["lowResPatches"], lowResPatches);
    cv::read(fs["highResPatches"], highResPatches);

    CV_DbgAssert(lowResPatches.rows == highResPatches.rows);
}

bool ExampledBased::empty() const
{
    return lowResPatches.empty();
}
