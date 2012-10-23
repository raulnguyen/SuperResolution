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

#include "super_resolution.hpp"
#include <opencv2/core/internal.hpp>

using namespace std;
using namespace cv;
using namespace cv::superres;

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(Farneback, "DenseOpticalFlow.Farneback",
                          obj.info()->addParam(obj, "pyrScale", obj.pyrScale);
                          obj.info()->addParam(obj, "numLevels", obj.numLevels);
                          obj.info()->addParam(obj, "winSize", obj.winSize);
                          obj.info()->addParam(obj, "numIters", obj.numIters);
                          obj.info()->addParam(obj, "polyN", obj.polyN);
                          obj.info()->addParam(obj, "polySigma", obj.polySigma);
                          obj.info()->addParam(obj, "flags", obj.flags));

        CV_INIT_ALGORITHM(Simple, "DenseOpticalFlow.Simple",
                          obj.info()->addParam(obj, "layers", obj.layers);
                          obj.info()->addParam(obj, "averagingBlockSize", obj.averagingBlockSize);
                          obj.info()->addParam(obj, "maxFlow", obj.maxFlow);
                          obj.info()->addParam(obj, "sigmaDist", obj.sigmaDist);
                          obj.info()->addParam(obj, "sigmaColor", obj.sigmaColor);
                          obj.info()->addParam(obj, "postProcessWindow", obj.postProcessWindow);
                          obj.info()->addParam(obj, "sigmaDistFix", obj.sigmaDistFix);
                          obj.info()->addParam(obj, "sigmaColorFix", obj.sigmaColorFix);
                          obj.info()->addParam(obj, "occThr", obj.occThr);
                          obj.info()->addParam(obj, "upscaleAveragingRadius", obj.upscaleAveragingRadius);
                          obj.info()->addParam(obj, "upscaleSigmaDist", obj.upscaleSigmaDist);
                          obj.info()->addParam(obj, "upscaleSigmaColor", obj.upscaleSigmaColor);
                          obj.info()->addParam(obj, "speedUpThr", obj.speedUpThr));

        CV_INIT_ALGORITHM(Brox_GPU, "DenseOpticalFlow.Brox_GPU",
                          obj.info()->addParam(obj, "alpha", obj.alpha, false, 0, 0, "Flow smoothness");
                          obj.info()->addParam(obj, "gamma", obj.gamma, false, 0, 0, "Gradient constancy importance");
                          obj.info()->addParam(obj, "scaleFactor", obj.scaleFactor, false, 0, 0, "Pyramid scale factor");
                          obj.info()->addParam(obj, "innerIterations", obj.innerIterations, false, 0, 0, "Number of lagged non-linearity iterations (inner loop)");
                          obj.info()->addParam(obj, "outerIterations", obj.outerIterations, false, 0, 0, "Number of warping iterations (number of pyramid levels)");
                          obj.info()->addParam(obj, "solverIterations", obj.solverIterations, false, 0, 0, "Number of linear system solver iterations"));

        CV_INIT_ALGORITHM(PyrLK_GPU, "DenseOpticalFlow.PyrLK_GPU",
                          obj.info()->addParam(obj, "winSize", obj.winSize);
                          obj.info()->addParam(obj, "maxLevel", obj.maxLevel);
                          obj.info()->addParam(obj, "iterations", obj.iterations));

        CV_INIT_ALGORITHM(Farneback_GPU, "DenseOpticalFlow.Farneback_GPU",
                          obj.info()->addParam(obj, "pyrScale", obj.pyrScale);
                          obj.info()->addParam(obj, "numLevels", obj.numLevels);
                          obj.info()->addParam(obj, "winSize", obj.winSize);
                          obj.info()->addParam(obj, "numIters", obj.numIters);
                          obj.info()->addParam(obj, "polyN", obj.polyN);
                          obj.info()->addParam(obj, "polySigma", obj.polySigma);
                          obj.info()->addParam(obj, "flags", obj.flags));

        CV_INIT_ALGORITHM(BTV_L1, "SuperResolution.BTV_L1",
                          obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0, "Scale factor.");
                          obj.info()->addParam(obj, "iterations", obj.iterations, false, 0, 0, "Iteration count.");
                          obj.info()->addParam(obj, "tau", obj.tau, false, 0, 0, "Asymptotic value of steepest descent method.");
                          obj.info()->addParam(obj, "lambda", obj.lambda, false, 0, 0, "Weight parameter to balance data term and smoothness term.");
                          obj.info()->addParam(obj, "alpha", obj.alpha, false, 0, 0, "Parameter of spacial distribution in btv.");
                          obj.info()->addParam(obj, "btvKernelSize", obj.btvKernelSize, false, 0, 0, "Kernel size of btv filter.");
                          obj.info()->addParam(obj, "blurKernelSize", obj.blurKernelSize, false, 0, 0, "Gaussian blur kernel size.");
                          obj.info()->addParam(obj, "blurSigma", obj.blurSigma, false, 0, 0, "Gaussian blur sigma.");
                          obj.info()->addParam(obj, "temporalAreaRadius", obj.temporalAreaRadius, false, 0, 0, "Radius of the temporal search area.");
                          obj.info()->addParam<DenseOpticalFlow>(obj, "opticalFlow", obj.opticalFlow, false, 0, 0, "Dense optical flow algorithm."));

        CV_INIT_ALGORITHM(BTV_L1_GPU, "SuperResolution.BTV_L1_GPU",
                          obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0, "Scale factor.");
                          obj.info()->addParam(obj, "iterations", obj.iterations, false, 0, 0, "Iteration count.");
                          obj.info()->addParam(obj, "tau", obj.tau, false, 0, 0, "Asymptotic value of steepest descent method.");
                          obj.info()->addParam(obj, "lambda", obj.lambda, false, 0, 0, "Weight parameter to balance data term and smoothness term.");
                          obj.info()->addParam(obj, "alpha", obj.alpha, false, 0, 0, "Parameter of spacial distribution in btv.");
                          obj.info()->addParam(obj, "btvKernelSize", obj.btvKernelSize, false, 0, 0, "Kernel size of btv filter.");
                          obj.info()->addParam(obj, "blurKernelSize", obj.blurKernelSize, false, 0, 0, "Gaussian blur kernel size.");
                          obj.info()->addParam(obj, "blurSigma", obj.blurSigma, false, 0, 0, "Gaussian blur sigma.");
                          obj.info()->addParam(obj, "temporalAreaRadius", obj.temporalAreaRadius, false, 0, 0, "Radius of the temporal search area.");
                          obj.info()->addParam<DenseOpticalFlow>(obj, "opticalFlow", obj.opticalFlow, false, 0, 0, "Dense optical flow algorithm."));
    }
}

bool cv::superres::initModule_superres()
{
    bool all = true;

    all &= !Farneback_info_auto.name().empty();
    all &= !Simple_info_auto.name().empty();
    all &= !Brox_GPU_info_auto.name().empty();
    all &= !PyrLK_GPU_info_auto.name().empty();
    all &= !Farneback_GPU_info_auto.name().empty();

    all &= !BTV_L1_info_auto.name().empty();
    all &= !BTV_L1_GPU_info_auto.name().empty();

    return all;
}

cv::superres::SuperResolution::SuperResolution()
{
    frameSource = new NullFrameSource();
    firstCall = true;
}

void cv::superres::SuperResolution::setFrameSource(const Ptr<IFrameSource>& frameSource)
{
    this->frameSource = frameSource;
    firstCall = true;
}

void cv::superres::SuperResolution::reset()
{
    this->frameSource->reset();
    firstCall = true;
}

Mat cv::superres::SuperResolution::nextFrame()
{
    if (firstCall)
    {
        initImpl(frameSource);
        firstCall = false;
    }

    return processImpl(frameSource);
}
