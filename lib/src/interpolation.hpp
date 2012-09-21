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

#pragma once

#ifndef __INTERPOLATION_HPP__
#define __INTERPOLATION_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "super_resolution_export.h"

namespace detail
{
    template <typename T, typename D>
    SUPER_RESOLUTION_NO_EXPORT T readVal(const cv::Mat& src, int y, int x, int c, int borderMode, cv::Scalar borderVal)
    {
        if (borderMode == cv::BORDER_CONSTANT)
            return (y >= 0 && y < src.rows && x >= 0 && x < src.cols) ? cv::saturate_cast<D>(src.at<T>(y, x * src.channels() + c)) : cv::saturate_cast<D>(borderVal.val[c]);

        return cv::saturate_cast<D>(src.at<T>(cv::borderInterpolate(y, src.rows, borderMode), cv::borderInterpolate(x, src.cols, borderMode) * src.channels() + c));
    }
}

template <typename T, typename D>
struct SUPER_RESOLUTION_NO_EXPORT NearestInterpolator
{
    static D getValue(const cv::Mat& src, double y, double x, int c = 0, int borderMode = cv::BORDER_REFLECT_101, cv::Scalar borderVal = cv::Scalar())
    {
        return detail::readVal<T, D>(src, int(y), int(x), c, borderMode, borderVal);
    }
};

template <typename T, typename D>
struct SUPER_RESOLUTION_NO_EXPORT LinearInterpolator
{
    static T getValue(const cv::Mat& src, double y, double x, int c = 0, int borderMode = cv::BORDER_REFLECT_101, cv::Scalar borderVal = cv::Scalar())
    {
        const int x1 = cvFloor(x);
        const int y1 = cvFloor(y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;

        double res = 0;

        res += detail::readVal<T, double>(src, y1, x1, c, borderMode, borderVal) * ((x2 - x) * (y2 - y));
        res += detail::readVal<T, double>(src, y1, x2, c, borderMode, borderVal) * ((x - x1) * (y2 - y));
        res += detail::readVal<T, double>(src, y2, x1, c, borderMode, borderVal) * ((x2 - x) * (y - y1));
        res += detail::readVal<T, double>(src, y2, x2, c, borderMode, borderVal) * ((x - x1) * (y - y1));

        return cv::saturate_cast<D>(res);
    }
};

template <typename T, typename D>
struct SUPER_RESOLUTION_NO_EXPORT CubicInterpolator
{
    static T getValue(const cv::Mat& src, double y, double x, int c = 0, int borderMode = cv::BORDER_REFLECT_101, cv::Scalar borderVal = cv::Scalar())
    {
        const int xmin = cvCeil(x - 2);
        const int xmax = cvFloor(x + 2);

        const int ymin = cvCeil(y - 2);
        const int ymax = cvFloor(y + 2);

        double sum  = 0;
        double wsum = 0;

        for (int cy = ymin; cy <= ymax; ++cy)
        {
            for (int cx = xmin; cx <= xmax; ++cx)
            {
                const double w = bicubicCoeff(x - cx) * bicubicCoeff(y - cy);

                sum += w * detail::readVal<T, double>(src, cvFloor(cy), cvFloor(cx), c, borderMode, borderVal);
                wsum += w;
            }
        }

        double res = wsum == 0 ? 0 : sum / wsum;

        return cv::saturate_cast<D>(res);
    }

private:
    static double bicubicCoeff(double x)
    {
        x = std::fabs(x);

        if (x <= 1)
        {
            return x * x * (1.5 * x - 2.5) + 1;
        }
        else if (x < 2)
        {
            return x * (x * (-0.5 * x + 2.5) - 4) + 2;
        }

        return 0;
    }
};

#endif // __INTERPOLATION_HPP__
