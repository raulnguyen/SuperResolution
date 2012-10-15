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

#ifndef __SR_OPTICAL_FLOW_HPP__
#define __SR_OPTICAL_FLOW_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        class SUPER_RESOLUTION_EXPORT DenseOpticalFlow : public Algorithm
        {
        public:
            virtual void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2 = noArray()) = 0;
        };

        class SUPER_RESOLUTION_EXPORT FarnebackOpticalFlow : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            FarnebackOpticalFlow();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);

            double pyrScale;
            int numLevels;
            int winSize;
            int numIters;
            int polyN;
            double polySigma;
            int flags;

        private:
            Mat buf0, buf1, buf2, buf3, buf4, buf5;
            Mat flow;
            std::vector<Mat> flows;
        };

        class SUPER_RESOLUTION_EXPORT SimpleOpticalFlow : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            SimpleOpticalFlow();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);

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
            Mat buf0, buf1, buf2, buf3, buf4, buf5;
            Mat flow;
            std::vector<Mat> flows;
        };

        class SUPER_RESOLUTION_EXPORT BroxOpticalFlow_GPU : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            BroxOpticalFlow_GPU();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);

            double alpha;
            double gamma;
            double scaleFactor;
            int innerIterations;
            int outerIterations;
            int solverIterations;

        private:
            gpu::BroxOpticalFlow alg;
            gpu::GpuMat buf0, buf1, buf2, buf3, buf4, buf5;
            gpu::GpuMat u, v, flow;
        };

        class SUPER_RESOLUTION_EXPORT PyrLKOpticalFlow_GPU : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            PyrLKOpticalFlow_GPU();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);

            int winSize;
            int maxLevel;
            int iterations;

        private:
            gpu::PyrLKOpticalFlow alg;
            gpu::GpuMat buf0, buf1, buf2, buf3, buf4, buf5;
            gpu::GpuMat u, v, flow;
        };

        class SUPER_RESOLUTION_EXPORT FarnebackOpticalFlow_GPU : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            FarnebackOpticalFlow_GPU();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);

            double pyrScale;
            int numLevels;
            int winSize;
            int numIters;
            int polyN;
            double polySigma;
            int flags;

        private:
            gpu::FarnebackOpticalFlow alg;
            gpu::GpuMat buf0, buf1, buf2, buf3, buf4, buf5;
            gpu::GpuMat u, v, flow;
        };
    }
}

#endif // __SR_OPTICAL_FLOW_HPP__
