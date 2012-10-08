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

#ifndef __VIDEO_SUPER_RESOLUTION_HPP__
#define __VIDEO_SUPER_RESOLUTION_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/videostab/frame_source.hpp>
#include "super_resolution_common.hpp"
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        using videostab::IFrameSource;

        enum VideoSRMethod
        {
            VIDEO_SR_NLM,
            VIDEO_SR_BILATERAL_TOTAL_VARIATION,
            VIDEO_SR_METHOD_MAX
        };

        class SUPER_RESOLUTION_EXPORT VideoSuperResolution : public Algorithm, public IFrameSource
        {
        public:
            static Ptr<VideoSuperResolution> create(VideoSRMethod method, bool useGpu = false);

            virtual ~VideoSuperResolution();

            void setFrameSource(const Ptr<IFrameSource>& frameSource);

            void reset();
            Mat nextFrame();

        protected:
            VideoSuperResolution();

            virtual void initImpl(Ptr<IFrameSource>& frameSource) = 0;
            virtual Mat processImpl(const Mat& frame) = 0;

        private:
            Ptr<IFrameSource> frameSource;
            bool firstCall;
        };
    }
}

#endif // __VIDEO_SUPER_RESOLUTION_HPP__
