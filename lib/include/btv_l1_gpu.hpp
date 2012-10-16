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

#ifndef __SR_BTV_L1_GPU_HPP__
#define __SR_BTV_L1_GPU_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        // Dennis Mitzel, Thomas Pock, Thomas Schoenemann, Daniel Cremers. Video Super Resolution using Duality Based TV-L1 Optical Flow.
        class SUPER_RESOLUTION_EXPORT BTV_L1_GPU_Base
        {
        public:
            BTV_L1_GPU_Base();

            void process(const std::vector<gpu::GpuMat>& src, gpu::GpuMat& dst,
                         const std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> >& motions,
                         int baseIdx = 0);

            int scale;
            int iterations;
            double lambda;
            double tau;
            double alpha;
            int btvKernelSize;
            int blurKernelSize;
            double blurSigma;

        private:
            std::vector<gpu::GpuMat> src_f;

            std::vector<float> btvWeights;
            int curBtvKernelSize;
            double curAlpha;

            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > lowResMotions;
            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > highResMotions;
            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > forward;
            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > backward;

            Ptr<gpu::FilterEngine_GPU> filter;
            int curBlurKernelSize;
            double curBlurSigma;
            int curSrcType;

            gpu::GpuMat highRes;

            gpu::GpuMat diffTerm, regTerm;
            gpu::GpuMat diff;
            gpu::GpuMat a, b, c, d;
        };
    }
}

#endif // __SR_BTV_L1_GPU_HPP__
