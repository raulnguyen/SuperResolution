#pragma once

#ifndef __SUPER_RESOLUTION_HPP__
#define __SUPER_RESOLUTION_HPP__

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#if defined WIN32 || defined _WIN32 || defined WINCE
#   if defined SUPER_RESOLUTION_SHARED
#       define SUPER_RESOLUTION_EXPORTS __declspec(dllexport)
#   else
#       define SUPER_RESOLUTION_EXPORTS __declspec(dllimport)
#   endif
#else
#   define SUPER_RESOLUTION_EXPORTS
#endif

class SUPER_RESOLUTION_EXPORTS SuperResolution
{
public:
    SuperResolution();

    void train(const std::vector<cv::Mat>& images, int step = 1);
    template <class Iter> void train(Iter begin, Iter end, int step = 1);
    void train(const cv::Mat& image, int step = 1);

    void clear();

    void operator ()(const cv::Mat& src, cv::Mat& dst);

    int lowResPatchSize;
    int highResPatchSize;
    double stdDevThresh;
    cv::Ptr<cv::DescriptorMatcher> matcher;

private:
    void buildPatchLists(const cv::Mat& src, cv::Mat& lowResPatches, cv::Mat& highResPatches, int step);

    cv::Mat lowResPatches;
    cv::Mat highResPatches;
};

template <class Iter>
void SuperResolution::train(Iter begin, Iter end, int step)
{
    std::vector<cv::Mat> images(begin, end);
    train(images, step);
}

#endif // __SUPER_RESOLUTION_H__
