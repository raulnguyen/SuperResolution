#pragma once

#ifndef __EXTRACT_PATCH_HPP__
#define __EXTRACT_PATCH_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include "interpolation.hpp"

namespace detail
{
    template <typename T, typename D, template <typename, typename> class Interpolator>
    void extractPatch(const cv::Mat& src, cv::Point2d p, std::vector<D>& patch, int patchSize)
    {
        CV_DbgAssert(patchSize % 2 != 0);

        const int cn = src.channels();
        const int rad = patchSize / 2;

        patch.resize(patchSize * patchSize * cn);

        typename std::vector<D>::iterator patchIt = patch.begin();

        for (int dy = -rad; dy <= rad; ++dy)
        {
            for (int dx = -rad; dx <= rad; ++dx)
            {
                for (int c = 0; c < cn; ++c)
                    *patchIt++ = Interpolator<T, D>::getValue(src, p.y + dy, p.x + dx, c);
            }
        }
    }
}

template <typename T>
void extractPatch(const cv::Mat& src, cv::Point2d p, std::vector<T>& patch, int patchSize, int interpolation = cv::INTER_NEAREST)
{
    typedef void (*func_t)(const cv::Mat& src, cv::Point2d p, std::vector<T>& patch, int patchSize);
    static const func_t funcs[7][3] =
    {
        {detail::extractPatch<uchar , T, NearestInterpolator>, detail::extractPatch<uchar , T, LinearInterpolator>, detail::extractPatch<uchar , T, CubicInterpolator>},
        {detail::extractPatch<schar , T, NearestInterpolator>, detail::extractPatch<schar , T, LinearInterpolator>, detail::extractPatch<schar , T, CubicInterpolator>},
        {detail::extractPatch<ushort, T, NearestInterpolator>, detail::extractPatch<ushort, T, LinearInterpolator>, detail::extractPatch<ushort, T, CubicInterpolator>},
        {detail::extractPatch<short , T, NearestInterpolator>, detail::extractPatch<short , T, LinearInterpolator>, detail::extractPatch<short , T, CubicInterpolator>},
        {detail::extractPatch<int   , T, NearestInterpolator>, detail::extractPatch<int   , T, LinearInterpolator>, detail::extractPatch<int   , T, CubicInterpolator>},
        {detail::extractPatch<float , T, NearestInterpolator>, detail::extractPatch<float , T, LinearInterpolator>, detail::extractPatch<float , T, CubicInterpolator>},
        {detail::extractPatch<double, T, NearestInterpolator>, detail::extractPatch<double, T, LinearInterpolator>, detail::extractPatch<double, T, CubicInterpolator>}
    };

    CV_DbgAssert(src.depth() >= 0 && src.depth() <= 7);
    CV_DbgAssert(interpolation >= 0 && interpolation < 3);

    const func_t func = funcs[src.depth()][interpolation];

    func(src, p, patch, patchSize);
}

template <typename T>
void scalePatch(std::vector<T>& patch, int patchSize, int cn, cv::Scalar scale)
{
    CV_DbgAssert(patch.size() == patchSize * patchSize * cn);

    for (int i = 0, ind = 0; i < patchSize * patchSize; ++i)
        for (int c = 0; c < cn; ++c, ++ind)
            patch[ind] /= scale[c];
}

#endif // __EXTRACT_PATCH_HPP__
