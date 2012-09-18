#pragma once

#ifndef __EXAMPLED_BASED_HPP__
#define __EXAMPLED_BASED_HPP__

#include <opencv2/features2d/features2d.hpp>
#include "super_resolution.hpp"

// W. T. Freeman, T. R. Jones, and E. C. Pasztor. Example-based super-resolution. Comp. Graph. Appl., (2), 2002
class ExampledBased : public SuperResolution
{
public:
    static cv::Ptr<SuperResolution> create();

    cv::AlgorithmInfo* info() const;

    ExampledBased();

    void train(const std::vector<cv::Mat>& images);
    void train(const cv::Mat& image);

    void clear();

    void process(const cv::Mat& src, cv::Mat& dst);

protected:
    void buildPatchLists(const cv::Mat& src, cv::Mat& lowResPatches, cv::Mat& highResPatches);

private:
    double patchStep;

    int lowResPatchSize;
    int highResPatchSize;

    double stdDevThresh;

    cv::Ptr<cv::DescriptorMatcher> matcher;

    cv::Mat lowResPatches;
    cv::Mat highResPatches;
};

#endif
