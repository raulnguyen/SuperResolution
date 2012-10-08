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

#ifndef __BTV_HPP__
#define __BTV_HPP__

#include <vector>
#include "image_super_resolution.hpp"
#include "video_super_resolution.hpp"
#include "motion_estimation.hpp"
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        class SUPER_RESOLUTION_NO_EXPORT BilateralTotalVariation
        {
        protected:
            BilateralTotalVariation();

            void setMotionModel(int motionModel);

            void process(const std::vector<Mat>& y, const std::vector<Mat>& DHF, int count, OutputArray dst);

            int scale;
            int iterations;
            double beta;
            double lambda;
            double alpha;
            int btvKernelSize;
            int workDepth;
            int motionModel;
            int blurModel;
            int blurKernelSize;

            Ptr<MotionEstimator> motionEstimator;

        private:
            Mat X;
            std::vector<Mat> diffTerms;
            std::vector<Mat> bufs;
            Mat regTerm;
            Mat blurWeights;
            int curBlurModel;
        };

        class SUPER_RESOLUTION_NO_EXPORT BTV_Image : public ImageSuperResolution, private BilateralTotalVariation
        {
        public:
            static bool init();
            static Ptr<ImageSuperResolution> create();

            AlgorithmInfo* info() const;

            void train(InputArrayOfArrays images);

            bool empty() const;
            void clear();

            void process(InputArray src, OutputArray dst);

        private:
            void trainImpl(const std::vector<Mat>& images);

            std::vector<Mat> images;

            std::vector<Mat> y;
            std::vector<Mat> DHF;

            Mat m1, m2;
        };

        class SUPER_RESOLUTION_NO_EXPORT BTV_Video : public VideoSuperResolution, private BilateralTotalVariation
        {
        public:
            static bool init();
            static Ptr<VideoSuperResolution> create();

            AlgorithmInfo* info() const;

            BTV_Video();

        protected:
            void initImpl(Ptr<IFrameSource>& frameSource);
            Mat processImpl(const Mat& frame);

        private:
            void addNewFrame(const Mat& frame);
            void processFrame(int idx);

            int temporalAreaRadius;

            std::vector<Mat> frames;
            std::vector<Mat> results;

            std::vector<Mat> y;
            std::vector<Mat> DHF;

            Mat m1, m2;

            int storePos;
            int procPos;
            int outPos;
        };
    }
}

#endif // __BTV_HPP__
