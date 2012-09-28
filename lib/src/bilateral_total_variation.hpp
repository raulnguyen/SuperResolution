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

#ifndef __BILATERAL_TOTAL_VARIATION_HPP__
#define __BILATERAL_TOTAL_VARIATION_HPP__

#include <opencv2/videostab/global_motion.hpp>
#include "image_super_resolution.hpp"
#include "super_resolution_export.h"

// S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
// Thanks to https://github.com/Palethorn/SuperResolution implementation.
class SUPER_RESOLUTION_NO_EXPORT BilateralTotalVariation : public cv::superres::ImageSuperResolution
{
public:
    static bool init();
    static cv::Ptr<ImageSuperResolution> create();

    cv::AlgorithmInfo* info() const;

    BilateralTotalVariation();

    void train(const std::vector<cv::Mat>& images);

    bool empty() const;
    void clear();

    void process(const cv::Mat& src, cv::Mat& dst);

protected:
    void calcDHF(cv::Size srcSize, const cv::Mat_<float>& M, cv::SparseMat_<double>& DHF);
    void btvRegularization(cv::Size highResSize);

private:
    int scale;
    int iterations;
    double beta;
    double lambda;
    double alpha;
    int btvKernelSize;
    int normType;

    std::vector<cv::Mat_<cv::Vec3b> > degImages;
    std::vector<cv::SparseMat_<double> > DHFs;

    cv::Ptr<cv::videostab::ImageMotionEstimatorBase> motionEstimator;

    cv::Mat_<cv::Point3d> dstVec;

    std::vector<cv::Mat_<cv::Point3d> > dstVecTemp;
    std::vector<cv::Mat_<cv::Point3d> > svec;
    std::vector<cv::Mat_<cv::Point3d> > svec2;

    cv::Mat_<cv::Point3d> regVec;
};

#endif // __BILATERAL_TOTAL_VARIATION_HPP__
