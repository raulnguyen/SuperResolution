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
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "extract_patch.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(ExampledBased, "ImageSuperResolution.ExampledBased",
                          obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0,
                                               "Scale factor.");
                          obj.info()->addParam(obj, "patchStep", obj.patchStep, false, 0, 0,
                                               "Step between patches in training.");
                          obj.info()->addParam(obj, "trainInterpolation", obj.trainInterpolation, false, 0, 0,
                                               "Interpolation method used for downsampling train images.");
                          obj.info()->addParam(obj, "lowResPatchSize", obj.lowResPatchSize, false, 0, 0,
                                               "Size of low-resolution patches.");
                          obj.info()->addParam(obj, "highResPatchSize", obj.highResPatchSize, false, 0, 0,
                                               "Size of high-resolution patches.");
                          obj.info()->addParam(obj, "stdDevThresh", obj.stdDevThresh, false, 0, 0,
                                               "Threshold value for patch standard deviation, only patches with high deviation will be processed.");
                          obj.info()->addParam(obj, "saveTrainBase", obj.saveTrainBase, false, 0, 0,
                                               "Store train base in write method."));
    }
}

bool cv::superres::ExampledBased::init()
{
    return !ExampledBased_info_auto.name().empty();
}

Ptr<ImageSuperResolution> cv::superres::ExampledBased::create()
{
    return Ptr<ImageSuperResolution>(new ExampledBased);
}

cv::superres::ExampledBased::ExampledBased()
{
    scale = 2.0;
    patchStep = 1.0;
    trainInterpolation = INTER_CUBIC;
    lowResPatchSize = 7;
    highResPatchSize = 5;
    stdDevThresh = 15.0;
    saveTrainBase = true;
    matcher = new BFMatcher(NORM_L2SQR);
}

void cv::superres::ExampledBased::train(InputArrayOfArrays _images)
{
    vector<Mat> images;

    if (_images.kind() == _InputArray::STD_VECTOR_MAT)
        _images.getMatVector(images);
    else
    {
        Mat image = _images.getMat();
        images.push_back(image);
    }

    trainImpl(images);
}

void cv::superres::ExampledBased::trainImpl(const std::vector<Mat>& images)
{
    CV_DbgAssert(!matcher.empty());

    for (size_t i = 0; i < images.size(); ++i)
    {
        Mat low, high;
        buildPatchList(images[i], low, high);

        if (!low.empty() && !high.empty())
        {
            lowResPatches.push_back(low);
            highResPatches.push_back(high);
        }
    }

    matcher->clear();
    matcher->add(lowResPatches);
}

void cv::superres::ExampledBased::buildPatchList(const Mat& highRes, Mat& lowResPatches, Mat& highResPatches)
{
    CV_DbgAssert(scale > 1);
    CV_DbgAssert(patchStep > 0);
    CV_DbgAssert(lowResPatchSize > 0 && lowResPatchSize % 2 != 0);
    CV_DbgAssert(highResPatchSize > 0 && highResPatchSize % 2 != 0);

    Mat lowRes;
    resize(highRes, lowRes, Size(), 1.0 / scale, 1.0 / scale, trainInterpolation);

    const int cn = highRes.channels();
    const float alpha = static_cast<float>(0.1 * lowResPatchSize * lowResPatchSize / (2.0 * highResPatchSize - 1.0));
    const int lowRad = lowResPatchSize / 2;
    const int maxPatchNumber = cvCeil((lowRes.cols + lowRad) / patchStep) * cvCeil((lowRes.rows + lowRad) / patchStep);

    lowResPatches.create(maxPatchNumber, (lowResPatchSize * lowResPatchSize + 2 * highResPatchSize - 1) * cn, CV_32F);
    highResPatches.create(maxPatchNumber, highResPatchSize * highResPatchSize * cn, CV_32F);

    vector<float> lowResPatch;
    vector<float> highResPatch;

    int patchNum = 0;
    Point2d pLow, pHigh;
    for (pLow.y = lowRad, pHigh.y = pLow.y * 2; pLow.y < lowRes.rows - lowRad; pLow.y += patchStep, pHigh.y = pLow.y * 2)
    {
        for (pLow.x = lowRad, pHigh.x = pLow.x * 2; pLow.x < lowRes.cols - lowRad; pLow.x += patchStep, pHigh.x = pLow.x * 2)
        {
            extractPatch(lowRes, pLow, lowResPatch, lowResPatchSize, INTER_LINEAR);

            Scalar mean, stddev;
            meanStdDev(Mat(lowResPatchSize, lowResPatchSize, CV_32FC(cn), &lowResPatch[0]), mean, stddev);

            if (stddev[0] < stdDevThresh && stddev[1] < stdDevThresh && stddev[2] < stdDevThresh && stddev[3] < stdDevThresh)
                continue;

            mean[0] += numeric_limits<double>::epsilon();
            mean[1] += numeric_limits<double>::epsilon();
            mean[2] += numeric_limits<double>::epsilon();
            mean[3] += numeric_limits<double>::epsilon();

            scalePatch(lowResPatch, lowResPatchSize, cn, mean);

            extractPatch(highRes, pHigh, highResPatch, highResPatchSize, INTER_LINEAR);

            scalePatch(highResPatch, highResPatchSize, cn, mean);

            // load top-left border from high res patch
            Mat_<float> patch(highResPatchSize, highResPatchSize * cn, &highResPatch[0]);
            for (int i = 0; i < highResPatchSize * cn; ++i)
                lowResPatch.push_back(alpha * patch(0, i));
            for (int i = 1; i < highResPatchSize; ++i)
                for (int c = 0; c < cn; ++c)
                    lowResPatch.push_back(alpha * patch(i, c));

            std::memcpy(lowResPatches.ptr(patchNum), &lowResPatch[0], lowResPatch.size() * sizeof(float));
            std::memcpy(highResPatches.ptr(patchNum), &highResPatch[0], highResPatch.size() * sizeof(float));

            ++patchNum;

            if (patchNum == maxPatchNumber)
                return;
        }
    }

    if (patchNum == 0)
    {
        lowResPatches = Mat();
        highResPatches = Mat();
    }
    else
    {
        lowResPatches = lowResPatches.rowRange(0, patchNum);
        highResPatches = highResPatches.rowRange(0, patchNum);
    }
}

bool cv::superres::ExampledBased::empty() const
{
    return lowResPatches.empty();
}

void cv::superres::ExampledBased::clear()
{
    CV_DbgAssert(!matcher.empty());

    lowResPatches.clear();
    highResPatches.clear();
    matcher->clear();
}

void cv::superres::ExampledBased::process(InputArray _src, OutputArray _dst)
{
    CV_DbgAssert(!matcher.empty());

    Mat src = _src.getMat();

    // if training base is empty, we train on source image itself

    if (empty())
        train(src);

    const int cn = src.channels();
    const float alpha = static_cast<float>(0.1 * lowResPatchSize * lowResPatchSize / (2.0 * highResPatchSize - 1.0));
    const int highRad = highResPatchSize / 2;
    const double weight = 1.0 / (highRad * highRad + 1.0);

    // initial estimation with bi-cubic interpolation

    resize(src, _dst, Size(), scale, scale, INTER_CUBIC);

    // if training base is stil empty return initial estimation

    if (lowResPatches.empty())
        return;

    Mat lowRes = src;
    Mat highRes = _dst.getMat();

    vector<float> lowResPatch;
    vector<float> highResPatch;

    vector<DMatch> matches;

    Mat dstNew;

    Mat result;
    highRes.convertTo(result, CV_32F, weight);

    for (int i = 0; i < highRad; ++i)
    {
        for (int j = 0; j < highRad; ++j)
        {
            highRes.convertTo(dstNew, CV_32F);

            Point pLow, pHigh;
            for (pHigh.y = i, pLow.y = pHigh.y / 2; pHigh.y < highRes.rows + highRad; pHigh.y += highResPatchSize - 1, pLow.y = pHigh.y / 2)
            {
                for (pHigh.x = j, pLow.x = pHigh.x / 2; pHigh.x < highRes.cols + highRad; pHigh.x += highResPatchSize - 1, pLow.x = pHigh.x / 2)
                {
                    extractPatch(lowRes, pLow, lowResPatch, lowResPatchSize, INTER_NEAREST);

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
                    extractPatch(highRes, pHigh, highResPatch, highResPatchSize, INTER_NEAREST);

                    scalePatch(highResPatch, highResPatchSize, cn, mean);

                    Mat_<float> patch(highResPatchSize, highResPatchSize * cn, &highResPatch[0]);
                    for (int k = 0; k < highResPatchSize * cn; ++k)
                        lowResPatch.push_back(alpha * patch(0, k));
                    for (int k = 1; k < highResPatchSize; ++k)
                        for (int c = 0; c < cn; ++c)
                            lowResPatch.push_back(alpha * patch(k, c));

                    // find nearest patch from low res training base
                    matcher->match(Mat(1, lowResPatch.size(), CV_32FC1, &lowResPatch[0]), matches);

                    // copy appropriate high res patch to dst

                    Mat_<float> newPatch(highResPatchSize, highResPatchSize * cn, highResPatches[matches[0].imgIdx].ptr<float>(matches[0].trainIdx));

                    for (int k = 0, ny = pHigh.y - highRad; k < highResPatchSize; ++k, ++ny)
                    {
                        if (ny >= 0 && ny < highRes.rows)
                        {
                            for (int m = 0, nx = pHigh.x - highRad; m < highResPatchSize; ++m, ++nx)
                            {
                                if (nx >= 0 && nx < highRes.cols)
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

    result.convertTo(_dst, CV_8U);
}

void cv::superres::ExampledBased::write(FileStorage& fs) const
{
    ImageSuperResolution::write(fs);

    if (saveTrainBase && !empty())
    {
        cv::write(fs, "lowResPatches", lowResPatches);
        cv::write(fs, "highResPatches", highResPatches);
    }
}

void cv::superres::ExampledBased::read(const FileNode& fn)
{
    ImageSuperResolution::read(fn);

    if (saveTrainBase)
    {
        cv::read(fn["lowResPatches"], lowResPatches);
        cv::read(fn["highResPatches"], highResPatches);
    }
}

///////////////////////////////////////////////////////////////
// Tests

#ifdef WITH_TESTS

namespace cv
{
    namespace superres
    {
        TEST(ExampledBased, TrainOnSingleHomogeneousImage)
        {
            Mat image(10, 10, CV_8U, Scalar::all(127));

            ExampledBased alg;
            alg.train(image);

            EXPECT_TRUE(alg.empty());
        }

        TEST(ExampledBased, TrainOnSeveralHomogeneousImages)
        {
            vector<Mat> images(5);

            for (size_t i = 0; i < images.size(); ++i)
                images[i] = Mat(10, 10, CV_8U, Scalar::all(127));

            ExampledBased alg;
            alg.train(images);

            EXPECT_TRUE(alg.empty());
        }

        TEST(ExampledBased, BuildPatchList)
        {
            const int lowResPatchSize = 3;
            const int highResPatchSize = 5;

            const uchar goldHighData[] =
            {
                 1, 0, 100, 0, 50,
                 0, 0,   0, 0,  0,
                20, 0, 200, 0, 30,
                 0, 0,   0, 0,  0,
                 5, 0, 210, 0, 15
            };
            const uchar goldLowData[] =
            {
                 1, 100, 50,
                20, 200, 30,
                 5, 210, 15
            };

            Mat_<uchar> trainImage(6, 6, (uchar)0);
            trainImage(0, 0) = goldHighData[ 0]; trainImage(0, 2) = goldHighData[ 2]; trainImage(0, 4) = goldHighData[ 4];
            trainImage(2, 0) = goldHighData[10]; trainImage(2, 2) = goldHighData[12]; trainImage(2, 4) = goldHighData[14];
            trainImage(4, 0) = goldHighData[20]; trainImage(4, 2) = goldHighData[22]; trainImage(4, 4) = goldHighData[24];

            ExampledBased alg;

            alg.set("scale", 2.0);
            alg.set("patchStep", 1.0);
            alg.set("trainInterpolation", INTER_NEAREST);
            alg.set("lowResPatchSize", lowResPatchSize);
            alg.set("highResPatchSize", highResPatchSize);

            Mat low, high;
            alg.buildPatchList(trainImage, low, high);

            EXPECT_EQ(CV_32F, low.type());
            EXPECT_EQ(1, low.rows);
            EXPECT_EQ(lowResPatchSize * lowResPatchSize + 2 * highResPatchSize - 1, low.cols);

            EXPECT_EQ(CV_32F, high.type());
            EXPECT_EQ(1, high.rows);
            EXPECT_EQ(highResPatchSize * highResPatchSize, high.cols);

            Scalar mean, stddev;
            meanStdDev(Mat(lowResPatchSize, lowResPatchSize, CV_8UC1, const_cast<uchar*>(goldLowData)), mean, stddev);

            mean[0] += numeric_limits<double>::epsilon();
            mean[1] += numeric_limits<double>::epsilon();
            mean[2] += numeric_limits<double>::epsilon();
            mean[3] += numeric_limits<double>::epsilon();

            multiply(low, mean, low);
            multiply(high, mean, high);

            low.convertTo(low, CV_8U);
            high.convertTo(high, CV_8U);

            EXPECT_EQ(0, std::memcmp(goldLowData, low.data, sizeof(goldLowData)));
            EXPECT_EQ(0, std::memcmp(goldHighData, high.data, sizeof(goldHighData)));
        }

        TEST(ExampledBased, ReadWriteConsistency)
        {
            Mat_<uchar> trainImage(6, 6, (uchar)0);
            trainImage(0, 0) =  1; trainImage(0, 2) = 100; trainImage(0, 4) = 50;
            trainImage(2, 0) = 20; trainImage(2, 2) = 200; trainImage(2, 4) = 30;
            trainImage(4, 0) =  5; trainImage(4, 2) = 201; trainImage(4, 4) = 15;

            string tempFileName = tempfile("ExampledBased_ReadWriteConsistency.xml");

            // write
            ExampledBased algWrite;

            algWrite.set("scale", 2.0);
            algWrite.set("patchStep", 1.0);
            algWrite.set("trainInterpolation", INTER_NEAREST);
            algWrite.set("lowResPatchSize", 3);
            algWrite.set("highResPatchSize", 5);
            algWrite.set("saveTrainBase", true);

            algWrite.train(trainImage);
            ASSERT_FALSE(algWrite.empty());

            FileStorage outFs(tempFileName, FileStorage::WRITE);
            algWrite.write(outFs);
            outFs.release();

            // read
            ExampledBased algRead;

            FileStorage inFs(tempFileName, FileStorage::READ);
            algRead.read(inFs.root());

            // check consistency

            EXPECT_EQ(algWrite.scale, algRead.scale);
            EXPECT_EQ(algWrite.patchStep, algRead.patchStep);
            EXPECT_EQ(algWrite.lowResPatchSize, algRead.lowResPatchSize);
            EXPECT_EQ(algWrite.highResPatchSize, algRead.highResPatchSize);
            EXPECT_EQ(algWrite.stdDevThresh, algRead.stdDevThresh);
            EXPECT_EQ(algWrite.saveTrainBase, algRead.saveTrainBase);
            EXPECT_EQ(algWrite.lowResPatches.size(), algRead.lowResPatches.size());
            EXPECT_EQ(algWrite.highResPatches.size(), algRead.highResPatches.size());

            for (size_t i = 0; i < algWrite.lowResPatches.size(); ++i)
            {
                const double lowResDiff = norm(algWrite.lowResPatches[i], algRead.lowResPatches[i], NORM_INF);
                EXPECT_EQ(0.0, lowResDiff);

                const double highResDiff = norm(algWrite.highResPatches[i], algRead.highResPatches[i], NORM_INF);
                EXPECT_EQ(0.0, highResDiff);
            }
        }
    }
}

#endif // WITH_TESTS
