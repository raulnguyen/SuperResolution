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

#ifndef __OPTICAL_FLOW_HPP__
#define __OPTICAL_FLOW_HPP__

#include <opencv2/core/core.hpp>
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        class SUPER_RESOLUTION_NO_EXPORT DenseOpticalFlow : public Algorithm
        {
        public:
            virtual void calc(InputArray frame0, InputArray frame1, OutputArray flow) = 0;
        };

        class SUPER_RESOLUTION_NO_EXPORT Farneback : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            Farneback();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow);

            double pyrScale;
            int numLevels;
            int winSize;
            int numIters;
            int polyN;
            double polySigma;
            int flags;

        private:
            Mat buf0, buf1;
        };

        class SUPER_RESOLUTION_NO_EXPORT SimpleFlow : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            SimpleFlow();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow);

            int layers;
            int averagingBlockSize;
            int maxFlow;
            double sigmaDist;
            double sigmaColor;
            int postProcessWindow;
            double sigmaDistFix;
            double sigmaColorFix;
            double occThr;
            int upscaleAveragingRadius;
            double upscaleSigmaDist;
            double upscaleSigmaColor;
            double speedUpThr;

        private:
            Mat buf0, buf1;
        };
    }
}

#endif // __OPTICAL_FLOW_HPP__
