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

#include "extract_patch.hpp"

#ifdef WITH_TESTS
#include <opencv2/ts/ts_gtest.h>
#endif

using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////
// Tests

#ifdef WITH_TESTS

TEST(ExtractPatch, Identical)
{
    Mat src(100, 100, CV_8UC3);

    theRNG().fill(src, RNG::UNIFORM, 0, 255);

    const Point loc(src.cols / 2, src.rows / 2);
    const int patchSize = 21;

    vector<uchar> patch1Vec;
    extractPatch(src, loc, patch1Vec, patchSize, INTER_NEAREST);

    ASSERT_EQ(patchSize * patchSize * 3, patch1Vec.size());

    Mat_<Vec3b> patch1(patchSize, patchSize, (Vec3b*) &patch1Vec[0]);

    Mat_<Vec3b> patch2 = extractPatch(src, loc, patchSize / 2);

    double diff = norm(patch1, patch2, NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

#endif // WITH_TESTS
