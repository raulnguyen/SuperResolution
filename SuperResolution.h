#pragma once

#ifndef __SUPER_RESOLUTION_H__
#define __SUPER_RESOLUTION_H__

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class SuperResolution
{
public:
    SuperResolution();

    void release();

    template <class Iter>
    void train(Iter begin, Iter end, double step = 1.0);
    void train(const cv::Mat& image, double step = 1.0) { train(&image, &image + 1, step); }

    void operator ()(const cv::Mat& src, cv::Mat& dst);

    int lowResPatchSize;
    int highResPatchSize;
    int patchInterpolation;
    cv::Ptr<cv::DescriptorMatcher> matcher;

private:
    void buildPatchLists(const cv::Mat& src, cv::Mat& lowResPatches, cv::Mat& highResPatches, double step);

    cv::Mat lowResPatches;
    cv::Mat highResPatches;
};

template <class Iter>
void SuperResolution::train(Iter begin, Iter end, double step)
{
    std::vector<cv::Mat> lowResPatchesVec;
    std::vector<cv::Mat> highResPatchesVec;

    int totalCount = 0;
    for (Iter it = begin; it != end; ++it)
    {
        cv::Mat low, high;
        buildPatchLists(*it, low, high, step);

        if (!low.empty())
        {
            totalCount += low.rows;
            lowResPatchesVec.push_back(low);
            highResPatchesVec.push_back(high);
        }
    }

    if (totalCount == 0)
    {
        release();
    }
    else
    {
        lowResPatches.create(totalCount, lowResPatchesVec[0].cols, lowResPatchesVec[0].type());
        highResPatches.create(totalCount, highResPatchesVec[0].cols, highResPatchesVec[0].type());

        int startRow = 0;
        for (size_t i = 0; i < lowResPatchesVec.size(); ++i)
        {
            lowResPatchesVec[i].copyTo(lowResPatches.rowRange(startRow, startRow + lowResPatchesVec[i].rows));
            highResPatchesVec[i].copyTo(highResPatches.rowRange(startRow, startRow + lowResPatchesVec[i].rows));
            startRow += lowResPatchesVec[i].rows;
        }
    }
}

#endif // __SUPER_RESOLUTION_H__
