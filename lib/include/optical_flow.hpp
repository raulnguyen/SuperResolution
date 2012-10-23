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

#ifndef __OPENCV_SR_OPTICAL_FLOW_HPP__
#define __OPENCV_SR_OPTICAL_FLOW_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
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
            virtual void collectGarbage();
        };

        class SUPER_RESOLUTION_EXPORT Farneback : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            Farneback();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
            void collectGarbage();

            double pyrScale;
            int numLevels;
            int winSize;
            int numIters;
            int polyN;
            double polySigma;
            int flags;

        private:
            void call(const Mat& input0, const Mat& input1, OutputArray dst);

            Mat buf[6];
            Mat flow;
            std::vector<Mat> flows;
        };

        class SUPER_RESOLUTION_EXPORT Simple : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            Simple();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
            void collectGarbage();

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
            void call(Mat input0, Mat input1, Mat& dst);

            Mat buf[6];
            Mat flow;
            std::vector<Mat> flows;
        };

        class SUPER_RESOLUTION_EXPORT Dual_TVL1 : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            Dual_TVL1();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
            void collectGarbage();

            double tau;
            double lambda;
            double theta;
            int    nscales;
            int    warps;
            double epsilon;
            int iterations;
            bool useInitialFlow;

        private:
            void call(const Mat& input0, const Mat& input1, OutputArray dst);

            Mat buf[6];
            Mat flow;
            std::vector<Mat> flows;
            OpticalFlowDual_TVL1 alg;
        };

        class SUPER_RESOLUTION_EXPORT Brox_GPU : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            Brox_GPU();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
            void collectGarbage();

            double alpha;
            double gamma;
            double scaleFactor;
            int innerIterations;
            int outerIterations;
            int solverIterations;

        private:
            void call(const gpu::GpuMat& input0, const gpu::GpuMat& input1, gpu::GpuMat& dst1, gpu::GpuMat& dst2);

            gpu::BroxOpticalFlow alg;
            gpu::GpuMat buf[6];
            gpu::GpuMat u, v, flow;
        };

        class SUPER_RESOLUTION_EXPORT PyrLK_GPU : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            PyrLK_GPU();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
            void collectGarbage();

            int winSize;
            int maxLevel;
            int iterations;

        private:
            void call(const gpu::GpuMat& input0, const gpu::GpuMat& input1, gpu::GpuMat& dst1, gpu::GpuMat& dst2);

            gpu::PyrLKOpticalFlow alg;
            gpu::GpuMat buf[6];
            gpu::GpuMat u, v, flow;
        };

        class SUPER_RESOLUTION_EXPORT Farneback_GPU : public DenseOpticalFlow
        {
        public:
            AlgorithmInfo* info() const;

            Farneback_GPU();

            void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
            void collectGarbage();

            double pyrScale;
            int numLevels;
            int winSize;
            int numIters;
            int polyN;
            double polySigma;
            int flags;

        private:
            void call(const gpu::GpuMat& input0, const gpu::GpuMat& input1, gpu::GpuMat& dst1, gpu::GpuMat& dst2);

            gpu::FarnebackOpticalFlow alg;
            gpu::GpuMat buf[6];
            gpu::GpuMat u, v, flow;
        };
    }
}

#endif // __OPENCV_SR_OPTICAL_FLOW_HPP__
