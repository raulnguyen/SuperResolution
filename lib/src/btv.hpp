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
#include "super_resolution.hpp"
#include "motion_estimation.hpp"
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        class SUPER_RESOLUTION_NO_EXPORT BTV_Base
        {
        public:
            struct DHF_Val
            {
                Point coord;
                float weight;
                DHF_Val(Point coord, float weight) : coord(coord), weight(weight) {}
            };

            BTV_Base();

            void process(const Mat& src, Mat& dst, const std::vector<Mat>& y, const std::vector<Mat>& DHF, int count);

            static void calcBlurWeights(BlurModel blurModel, int blurKernelSize, std::vector<float>& blurWeights);
            static void calcDhf(Size lowResSize, int scale, int blurKernelSize, const std::vector<float>& blurWeights,
                                MotionModel motionModel, const Mat& m1, const Mat& m2, Mat& DHF);

            int scale;
            int iterations;
            double beta;
            double lambda;
            double alpha;
            int btvKernelSize;

        private:
            Mat X;

            std::vector<Mat> diffTerms;
            std::vector<Mat> bufs;

            Mat regTerm;

            std::vector<float> btvWeights;
        };

        class SUPER_RESOLUTION_NO_EXPORT BTV : public SuperResolution, private BTV_Base
        {
        public:
            AlgorithmInfo* info() const;

            static bool init();
            static Ptr<SuperResolution> create();

            BTV();

        protected:
            void initImpl(Ptr<IFrameSource>& frameSource);
            Mat processImpl(Ptr<IFrameSource>& frameSource);

        private:
            void setMotionModel(int motionModel);
            void addNewFrame(const Mat& frame);
            void processFrame(int idx);

            int motionModel;
            int blurModel;
            int blurKernelSize;
            int temporalAreaRadius;

            Ptr<MotionEstimator> motionEstimator;
            Mat m1, m2;

            std::vector<Mat> frames;
            std::vector<Mat> results;

            int storePos;
            int procPos;
            int outPos;

            std::vector<float> blurWeights;
            int curBlurModel;

            Mat src_f;
            std::vector<Mat> y;
            std::vector<Mat> DHF;
        };
    }
}

#endif // __BTV_HPP__
