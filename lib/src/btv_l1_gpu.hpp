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

#ifndef __BTV_L1_GPU_HPP__
#define __BTV_L1_GPU_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "super_resolution.hpp"
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        using std::vector;
        using std::pair;
        using cv::gpu::GpuMat;

        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        // Dennis Mitzel, Thomas Pock, Thomas Schoenemann, Daniel Cremers. Video Super Resolution using Duality Based TV-L1 Optical Flow.
        class SUPER_RESOLUTION_NO_EXPORT BTV_L1_GPU_Base
        {
        public:
            BTV_L1_GPU_Base();

            void process(const vector<GpuMat>& src, GpuMat& dst, int startIdx, int procIdx, int endIdx);

            int scale;
            int iterations;
            double lambda;
            double tau;
            double alpha;
            int btvKernelSize;
            int blurKernelSize;
            double blurSigma;

        private:
            vector<float> btvWeights;

            GpuMat gray0, gray1;
            gpu::FarnebackOpticalFlow opticalFlow;

            vector<pair<GpuMat, GpuMat> > lowResMotions;
            vector<pair<GpuMat, GpuMat> > highResMotions;
            vector<pair<GpuMat, GpuMat> > forward, backward;

            vector<GpuMat> src_f;

            Ptr<gpu::FilterEngine_GPU> filter;
            int curBlurKernelSize;
            double curBlurSigma;
            int curSrcType;

            GpuMat highRes;

            GpuMat diffTerm, regTerm;
            GpuMat diff;
            GpuMat a, b, c, d;
        };

        class SUPER_RESOLUTION_NO_EXPORT BTV_L1_GPU : public SuperResolution, private BTV_L1_GPU_Base
        {
        public:
            AlgorithmInfo* info() const;

            static bool init();
            static Ptr<SuperResolution> create();

            BTV_L1_GPU();

        protected:
            void initImpl(Ptr<IFrameSource>& frameSource);
            Mat processImpl(Ptr<IFrameSource>& frameSource);

        private:
            void addNewFrame(const Mat& frame);
            void processFrame(int idx);

            int temporalAreaRadius;

            vector<GpuMat> frames;
            vector<GpuMat> results;
            Mat h_dst;

            int storePos;
            int procPos;
            int outPos;
        };
    }
}

#endif // __TV_L1_GPU_HPP__
