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

#include "interpolation.hpp"

#ifdef WITH_TESTS
#include <opencv2/ts/ts_gtest.h>
#endif

using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////
// Tests

#ifdef WITH_TESTS

TEST(ReadVal, Inner)
{
    Mat_<Vec3b> src(3, 3);
    src << Vec3b(0,1,2), Vec3b(3,4,5), Vec3b(6,7,8), Vec3b(9,10,11), Vec3b(12,13,14), Vec3b(15,16,17), Vec3b(18,19,20), Vec3b(21,22,23), Vec3b(24,25,26);

    for (int y = 0; y < src.rows; ++y)
    {
        for (int x = 0; x < src.cols; ++x)
        {
            for (int c = 0; c < src.channels(); ++c)
            {
                int gold = src(y, x)[c];
                int val = readVal<uchar, uchar>(src, y, x, c);
                EXPECT_EQ(gold, val);
            }
        }
    }
}

TEST(ReadVal, OutOfBorder)
{
    Mat_<Vec3b> src(3, 3);
    src << Vec3b(0,1,2), Vec3b(3,4,5), Vec3b(6,7,8), Vec3b(9,10,11), Vec3b(12,13,14), Vec3b(15,16,17), Vec3b(18,19,20), Vec3b(21,22,23), Vec3b(24,25,26);

    for (int y = -src.rows; y < 0; ++y)
    {
        for (int x = -src.cols; x < 0; ++x)
        {
            for (int c = 0; c < src.channels(); ++c)
            {
                int gold = src(borderInterpolate(y, src.rows, BORDER_REFLECT_101), borderInterpolate(x, src.cols, BORDER_REFLECT_101))[c];
                int val = readVal<uchar, uchar>(src, y, x, c, BORDER_REFLECT_101);
                EXPECT_EQ(gold, val);
            }
        }
    }
}

TEST(NearestInterpolator, GetValue)
{
    Mat_<int> src(3, 3);
    src << 0, 1, 2, 3, 4, 5, 6, 7, 8;

    int gold = src(1, 1);
    int val = NearestInterpolator<int, int>::getValue(src, 1.5, 1.5);
    EXPECT_EQ(gold, val);
}

TEST(LinearInterpolator, GetValue)
{
    Mat_<int> src(3, 3);
    src << 0, 1, 2, 3, 4, 5, 6, 7, 8;

    double gold = 0.25 * src(1, 1) + 0.25 * src(1, 2) + 0.25 * src(2, 1) + 0.25 * src(2, 2);
    double val = LinearInterpolator<int, double>::getValue(src, 1.5, 1.5);
    EXPECT_EQ(gold, val);
}

#endif // WITH_TESTS
