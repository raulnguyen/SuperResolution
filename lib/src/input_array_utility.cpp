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

#include "input_array_utility.hpp"
#include <limits>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

Mat getMat(InputArray arr, Mat& buf)
{
    if (arr.kind() == _InputArray::GPU_MAT)
    {
        arr.getGpuMat().download(buf);
        return buf;
    }

    return arr.getMat();
}

GpuMat getGpuMat(InputArray arr, GpuMat& buf)
{
    if (arr.kind() != _InputArray::GPU_MAT)
    {
        buf.upload(arr.getMat());
        return buf;
    }

    return arr.getGpuMat();
}

void copy(OutputArray dst, const Mat& src)
{
    if (dst.kind() == _InputArray::GPU_MAT)
        dst.getGpuMatRef().upload(src);
    else
        src.copyTo(dst);
}

void copy(OutputArray dst, const GpuMat& src)
{
    if (dst.kind() == _InputArray::GPU_MAT)
        src.copyTo(dst.getGpuMatRef());
    else
    {
        dst.create(src.size(), src.type());
        Mat h_dst = dst.getMat();
        src.download(h_dst);
    }
}

Mat convertToType(const Mat& src, int depth, int cn, Mat& buf0, Mat& buf1)
{
    CV_DbgAssert( src.depth() <= CV_64F );
    CV_DbgAssert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );
    CV_DbgAssert( depth == CV_8U || depth == CV_32F );
    CV_DbgAssert( cn == 1 || cn == 3 || cn == 4 );

    Mat result;

    if (src.depth() == depth)
        result = src;
    else
    {
        static const double maxVals[] =
        {
            numeric_limits<uchar>::max(),
            numeric_limits<schar>::max(),
            numeric_limits<ushort>::max(),
            numeric_limits<short>::max(),
            numeric_limits<int>::max(),
            1.0,
            1.0,
        };

        const double scale = maxVals[depth] / maxVals[src.depth()];

        src.convertTo(buf0, depth, scale);
        result = buf0;
    }

    if (result.channels() == cn)
        return result;

    static const int codes[5][5] =
    {
        {-1, -1, -1, -1, -1},
        {-1, -1, -1, COLOR_GRAY2BGR, COLOR_GRAY2BGRA},
        {-1, -1, -1, -1, -1},
        {-1, COLOR_BGR2GRAY, -1, -1, COLOR_BGR2BGRA},
        {-1, COLOR_BGRA2GRAY, -1, COLOR_BGRA2BGR, -1},
    };

    const int code = codes[src.channels()][cn];
    CV_DbgAssert( code >= 0 );

    cvtColor(result, buf1, code, cn);
    return buf1;
}

GpuMat convertToType(const GpuMat& src, int depth, int cn, GpuMat& buf0, GpuMat& buf1)
{
    CV_DbgAssert( src.depth() <= CV_64F );
    CV_DbgAssert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );
    CV_DbgAssert( depth == CV_8U || depth == CV_32F );
    CV_DbgAssert( cn == 1 || cn == 3 || cn == 4 );

    GpuMat result;

    if (src.depth() == depth)
        result = src;
    else
    {
        static const double maxVals[] =
        {
            numeric_limits<uchar>::max(),
            numeric_limits<schar>::max(),
            numeric_limits<ushort>::max(),
            numeric_limits<short>::max(),
            numeric_limits<int>::max(),
            1.0,
            1.0,
        };

        const double scale = maxVals[depth] / maxVals[src.depth()];

        src.convertTo(buf0, depth, scale);
        result = buf0;
    }

    if (result.channels() == cn)
        return result;

    static const int codes[5][5] =
    {
        {-1, -1, -1, -1, -1},
        {-1, -1, -1, COLOR_GRAY2BGR, COLOR_GRAY2BGRA},
        {-1, -1, -1, -1, -1},
        {-1, COLOR_BGR2GRAY, -1, -1, COLOR_BGR2BGRA},
        {-1, COLOR_BGRA2GRAY, -1, COLOR_BGRA2BGR, -1},
    };

    const int code = codes[src.channels()][cn];
    CV_DbgAssert( code >= 0 );

    gpu::cvtColor(result, buf1, code, cn);
    return buf1;
}
