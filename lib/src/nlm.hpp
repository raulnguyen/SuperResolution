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

#ifndef __NLM_HPP__
#define __NLM_HPP__

#include <vector>
#include "video_super_resolution.hpp"
#include "super_resolution_export.h"

// M. Protter, M. Elad, H. Takeda, and P. Milanfar. Generalizing the nonlocal-means to super-resolution reconstruction.
class SUPER_RESOLUTION_NO_EXPORT Nlm : public cv::superres::VideoSuperResolution
{
public:
    static bool init();
    static cv::Ptr<VideoSuperResolution> create();

    cv::AlgorithmInfo* info() const;

    Nlm();

protected:
    void initImpl(cv::Ptr<IFrameSource>& frameSource);
    cv::Mat processImpl(const cv::Mat& frame);

private:
    void addNewFrame(const cv::Mat& frame);
    void processFrame(int idx);

    int scale;
    int searchWindowRadius;
    int temporalAreaRadius;
    int patchRadius;
    double sigma;
    bool doDeblurring;

    int storePos;
    int procPos;
    int outPos;

    std::vector<cv::Mat> y; // input set of low resolution and noisy images
    std::vector<cv::Mat> Y; // An initial estimate of the super-resolved sequence.

    cv::Mat_<cv::Point3d> V;
    cv::Mat_<cv::Point3d> W;

    cv::Mat Z;

    cv::Mat buf;

    cv::Ptr<cv::videostab::ImageMotionEstimatorBase> motionEstimator;
    cv::Ptr<cv::videostab::DeblurerBase> deblurer;
    std::vector<cv::Mat> outFrames;
    std::vector<cv::Mat> motions;
    std::vector<float> blurrinessRates;
};

#endif // __NLM_HPP__