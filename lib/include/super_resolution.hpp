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

#ifndef __SUPER_RESOLUTION_HPP__
#define __SUPER_RESOLUTION_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/videostab/frame_source.hpp>
#include "optical_flow.hpp"
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        SUPER_RESOLUTION_EXPORT bool initModule_superres();

        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        // Dennis Mitzel, Thomas Pock, Thomas Schoenemann, Daniel Cremers. Video Super Resolution using Duality Based TV-L1 Optical Flow.
        class SUPER_RESOLUTION_EXPORT BTV_L1_Base
        {
        public:
            BTV_L1_Base();

            void process(const std::vector<Mat>& src, OutputArray dst, int baseIdx = 0);
            void process(const std::vector<Mat>& src, OutputArray dst, const std::vector<Mat>& forwardMotions, int baseIdx = 0);

            int scale;
            int iterations;
            double tau;
            double lambda;
            double alpha;
            int btvKernelSize;
            int blurKernelSize;
            double blurSigma;
            Ptr<DenseOpticalFlow> opticalFlow;

        protected:
            void run(const std::vector<Mat>& src, OutputArray dst, const std::vector<Mat>& relativeMotions, int baseIdx);

        private:
            std::vector<Mat> src_f;

            std::vector<float> btvWeights;
            int curBtvKernelSize;
            double curAlpha;

            std::vector<Mat> lowResMotions;
            std::vector<Mat> highResMotions;
            std::vector<Mat> forward;
            std::vector<Mat> backward;

            Ptr<FilterEngine> filter;
            int curBlurKernelSize;
            double curBlurSigma;
            int curSrcType;

            Mat highRes;

            Mat diffTerm, regTerm;
            Mat diff;
            Mat a, b, c, d;
        };

        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        // Dennis Mitzel, Thomas Pock, Thomas Schoenemann, Daniel Cremers. Video Super Resolution using Duality Based TV-L1 Optical Flow.
        class SUPER_RESOLUTION_EXPORT BTV_L1_GPU_Base
        {
        public:
            BTV_L1_GPU_Base();

            void process(const std::vector<gpu::GpuMat>& src, gpu::GpuMat& dst, int baseIdx = 0);
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
            Ptr<DenseOpticalFlow> opticalFlow;

        protected:
            void run(const std::vector<gpu::GpuMat>& src, gpu::GpuMat& dst,
                     const std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> >& relativeMotions,
                     int baseIdx);

        private:
            std::vector<gpu::GpuMat> src_f;

            std::vector<float> btvWeights;
            int curBtvKernelSize;
            double curAlpha;

            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > lowResMotions;
            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > highResMotions;
            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > forward;
            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > backward;

            std::vector<Ptr<gpu::FilterEngine_GPU> > filters;
            int curBlurKernelSize;
            double curBlurSigma;
            int curSrcType;

            gpu::GpuMat highRes;

            std::vector<gpu::GpuMat> diffTerms;
            gpu::GpuMat regTerm;
            gpu::GpuMat diff;
            gpu::GpuMat a, b, c, d;

            std::vector<gpu::Stream> streams;
        };

        using videostab::IFrameSource;
        using videostab::NullFrameSource;
        using videostab::VideoFileSource;

        class SUPER_RESOLUTION_EXPORT SuperResolution : public Algorithm, public IFrameSource
        {
        public:
            void setFrameSource(const Ptr<IFrameSource>& frameSource);

            void reset();
            Mat nextFrame();

        protected:
            SuperResolution();

            virtual void initImpl(Ptr<IFrameSource>& frameSource) = 0;
            virtual Mat processImpl(Ptr<IFrameSource>& frameSource) = 0;

        private:
            Ptr<IFrameSource> frameSource;
            bool firstCall;
        };

        class SUPER_RESOLUTION_EXPORT BTV_L1 : public SuperResolution, private BTV_L1_Base
        {
        public:
            AlgorithmInfo* info() const;

            BTV_L1();

            int temporalAreaRadius;

        protected:
            void initImpl(Ptr<IFrameSource>& frameSource);
            Mat processImpl(Ptr<IFrameSource>& frameSource);

        private:
            void addNewFrame(const Mat& frame);
            void processFrame(int idx);

            std::vector<Mat> frames;
            std::vector<Mat> results;

            std::vector<Mat> motions;
            Mat prevFrame;

            int storePos;
            int procPos;
            int outPos;

            std::vector<Mat> src;
            std::vector<Mat> relMotions;
            Mat dst;
        };

        class SUPER_RESOLUTION_EXPORT BTV_L1_GPU : public SuperResolution, private BTV_L1_GPU_Base
        {
        public:
            AlgorithmInfo* info() const;

            BTV_L1_GPU();

            int temporalAreaRadius;

        protected:
            void initImpl(Ptr<IFrameSource>& frameSource);
            Mat processImpl(Ptr<IFrameSource>& frameSource);

        private:
            void addNewFrame(const Mat& frame);
            void processFrame(int idx);

            std::vector<gpu::GpuMat> frames;
            std::vector<gpu::GpuMat> results;

            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > motions;
            gpu::GpuMat d_frame;
            gpu::GpuMat prevFrame;

            int storePos;
            int procPos;
            int outPos;

            std::vector<gpu::GpuMat> src;
            std::vector<std::pair<gpu::GpuMat, gpu::GpuMat> > relMotions;
            gpu::GpuMat dst;
            Mat h_dst;
        };
    }
}

#endif // __SUPER_RESOLUTION_HPP__
