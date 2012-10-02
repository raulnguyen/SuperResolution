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

#ifndef __EXAMPLED_BASED_HPP__
#define __EXAMPLED_BASED_HPP__

#include <vector>
#include <opencv2/features2d/features2d.hpp>
#ifdef WITH_TESTS
    #include <opencv2/ts/ts_gtest.h>
#endif
#include "image_super_resolution.hpp"
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        // W. T. Freeman, T. R. Jones, and E. C. Pasztor. Example-based super-resolution.
        class SUPER_RESOLUTION_NO_EXPORT ExampledBased : public ImageSuperResolution
        {
        public:
            static bool init();
            static Ptr<ImageSuperResolution> create();

            AlgorithmInfo* info() const;

            ExampledBased();

            void train(InputArrayOfArrays images);

            bool empty() const;
            void clear();

            void process(InputArray src, OutputArray dst);

            void write(FileStorage& fs) const;
            void read(const FileNode& fn);

        protected:
            void trainImpl(const std::vector<Mat>& images);
            void buildPatchList(const Mat& src, Mat& lowResPatches, Mat& highResPatches);

        private:
            double scale;

            double patchStep;
            int trainInterpolation;

            int lowResPatchSize;
            int highResPatchSize;

            double stdDevThresh;

            bool saveTrainBase;

            Ptr<DescriptorMatcher> matcher;

            std::vector<Mat> lowResPatches;
            std::vector<Mat> highResPatches;

            #ifdef WITH_TESTS
                FRIEND_TEST(ExampledBased, BuildPatchList);
                FRIEND_TEST(ExampledBased, ReadWriteConsistency);
            #endif
        };
    }
}

#endif
