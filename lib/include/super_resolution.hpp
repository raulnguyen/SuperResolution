#pragma once

#ifndef __SUPER_RESOLUTION_HPP__
#define __SUPER_RESOLUTION_HPP__

#include <vector>
#include <opencv2/core/core.hpp>

#if defined WIN32 || defined _WIN32 || defined WINCE
#   if defined SUPER_RESOLUTION_SHARED
#       define SUPER_RESOLUTION_EXPORTS __declspec(dllexport)
#   else
#       define SUPER_RESOLUTION_EXPORTS __declspec(dllimport)
#   endif
#else
#   define SUPER_RESOLUTION_EXPORTS
#endif

class SUPER_RESOLUTION_EXPORTS SuperResolution : public cv::Algorithm
{
public:
    enum Method
    {
        EXAMPLE_BASED,
        METHOD_MAX
    };

    static cv::Ptr<SuperResolution> create(Method method);

    virtual ~SuperResolution();

    virtual void train(const std::vector<cv::Mat>& images) = 0;
    virtual void train(const cv::Mat& image);
    template <class Iter> void train(Iter begin, Iter end);

    virtual void clear() = 0;

    virtual void process(const cv::Mat& src, cv::Mat& dst) = 0;
};

template <class Iter>
void SuperResolution::train(Iter begin, Iter end)
{
    std::vector<cv::Mat> images(begin, end);
    train(images);
}

#endif // __SUPER_RESOLUTION_HPP__
