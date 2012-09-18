#include "SuperResolution.h"

#include <cstring>
#include <cmath>

#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////////////
// Extract Patch

namespace
{
    template <typename T, typename D>
    void extractPatchImpl(const Mat& src, Point p, vector<D>& patch, int patchSize)
    {
        CV_DbgAssert(patchSize % 2 != 0);

        const int cn = src.channels();
        const int rad = patchSize / 2;

        patch.resize(patchSize * patchSize * cn);

        typename vector<D>::iterator patchIt = patch.begin();

        for (int dy = -rad; dy <= rad; ++dy)
        {
            const int y = borderInterpolate(p.y + dy, src.rows, BORDER_REFLECT_101);
            const T* srcRow = src.ptr<T>(y);

            for (int dx = -rad; dx <= rad; ++dx)
            {
                const int x = borderInterpolate(p.x + dx, src.cols, BORDER_REFLECT_101);

                for (int c = 0; c < cn; ++c)
                    *patchIt++ = saturate_cast<D>(srcRow[x * cn + c]);
            }
        }
    }

    template <typename T>
    void extractPatch(const Mat& src, Point p, vector<T>& patch, int patchSize)
    {
        typedef void (*func_t)(const Mat& src, Point p, vector<T>& patch, int patchSize);
        static const func_t funcs[] =
        {
            extractPatchImpl<uchar , T>,
            extractPatchImpl<schar , T>,
            extractPatchImpl<ushort, T>,
            extractPatchImpl<short , T>,
            extractPatchImpl<int   , T>,
            extractPatchImpl<float , T>,
            extractPatchImpl<double, T>
        };

        CV_DbgAssert(src.depth() >= CV_8U && src.depth() <= CV_64F);

        const func_t func = funcs[src.depth()];
        CV_DbgAssert(func != 0);

        func(src, p, patch, patchSize);
    }

    template <typename T>
    void scalePatch(vector<T>& patch, int patchSize, int cn, Scalar scale)
    {
        CV_DbgAssert(patch.size() == patchSize * patchSize * cn);

        for (int i = 0, ind = 0; i < patchSize * patchSize; ++i)
            for (int c = 0; c < cn; ++c, ++ind)
                patch[ind] /= scale[c];
    }
}

///////////////////////////////////////////////////////////////////////
// SuperResolution

SuperResolution::SuperResolution()
{
    lowResPatchSize = 7;
    highResPatchSize = 5;
    matcher = new BFMatcher(NORM_L2SQR);
}

void SuperResolution::train(const vector<Mat>& images, int step)
{
    vector<Mat> lowResPatchesVec;
    vector<Mat> highResPatchesVec;

    int totalCount = 0;
    for (size_t i = 0; i < images.size(); ++i)
    {
        cv::Mat low, high;
        buildPatchLists(images[i], low, high, step);

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

void SuperResolution::train(const Mat& image, int step)
{
    buildPatchLists(image, lowResPatches, highResPatches, step);
}

void SuperResolution::clear()
{
    lowResPatches.release();
    highResPatches.release();
}

void SuperResolution::operator ()(const Mat& src, Mat& dst)
{
    CV_Assert(lowResPatchSize % 2 != 0);
    CV_Assert(highResPatchSize % 2 != 0);

    if (lowResPatches.empty())
        train(src);

    const int cn = src.channels();

    const float alpha = static_cast<float>(0.1 * lowResPatchSize * lowResPatchSize / (2.0 * highResPatchSize - 1.0));

    const int highRad = highResPatchSize / 2;

    pyrUp(src, dst);

    if (lowResPatches.empty())
        return;

    Mat lowRes = src;
    Mat highRes = dst;
    vector<float> lowResPatch;
    vector<float> highResPatch;

    vector<DMatch> matches;

    Mat dstNew(dst.size(), CV_32FC(cn));

    const double weight = 1.0 / (highRad * highRad + 1.0);
    Mat result;
    dst.convertTo(result, CV_32F, weight);

    for (int i = 0; i < highRad; ++i)
    {
        for (int j = 0; j < highRad; ++j)
        {
            Point pLow, pHigh;
            for (pHigh.y = i, pLow.y = pHigh.y / 2; pHigh.y < dst.rows + highRad; pHigh.y += highResPatchSize - 1, pLow.y = pHigh.y / 2)
            {
                for (pHigh.x = j, pLow.x = pHigh.x / 2; pHigh.x < dst.cols + highRad; pHigh.x += highResPatchSize - 1, pLow.x = pHigh.x / 2)
                {
                    extractPatch(lowRes, pLow, lowResPatch, lowResPatchSize);

                    Scalar mean, stddev;
                    meanStdDev(Mat(lowResPatchSize, lowResPatchSize, CV_32FC(cn), &lowResPatch[0]), mean, stddev);

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

void SuperResolution::buildPatchLists(const Mat& highRes, Mat& lowResPatches, Mat& highResPatches, int step)
{
    const int cn = highRes.channels();

    Mat lowRes;
    pyrDown(highRes, lowRes);

    const float alpha = static_cast<float>(0.1 * lowResPatchSize * lowResPatchSize / (2.0 * highResPatchSize - 1.0));

    const int lowRad = lowResPatchSize / 2;

    const int maxPatchNumber = cvCeil((lowRes.cols + lowRad) / step) * cvCeil((lowRes.rows + lowRad) / step);

    lowResPatches.create(maxPatchNumber, (lowResPatchSize * lowResPatchSize + 2 * highResPatchSize - 1) * cn, CV_32F);
    highResPatches.create(maxPatchNumber, highResPatchSize * highResPatchSize * cn, CV_32F);

    vector<float> lowResPatch;
    vector<float> highResPatch;

    int patchNum = 0;
    Point pLow, pHigh;
    for (pLow.y = 0, pHigh.y = pLow.y * 2; pLow.y < lowRes.rows + lowRad; pLow.y += step, pHigh.y = pLow.y * 2)
    {
        for (pLow.x = 0, pHigh.x = pLow.x * 2; pLow.x < lowRes.cols + lowRad; pLow.x += step, pHigh.x = pLow.x * 2)
        {
            extractPatch(lowRes, pLow, lowResPatch, lowResPatchSize);

            Scalar mean, stddev;
            meanStdDev(Mat(lowResPatchSize, lowResPatchSize, CV_32FC(cn), &lowResPatch[0]), mean, stddev);

            mean[0] += numeric_limits<double>::epsilon();
            mean[1] += numeric_limits<double>::epsilon();
            mean[2] += numeric_limits<double>::epsilon();
            mean[3] += numeric_limits<double>::epsilon();

            scalePatch(lowResPatch, lowResPatchSize, cn, mean);

            extractPatch(highRes, pHigh, highResPatch, highResPatchSize);

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
