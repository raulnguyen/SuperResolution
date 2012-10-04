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
#include "exampled_based.hpp"
#include "btv.hpp"
#include "nlm.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;

bool cv::superres::initModule_superres()
{
    bool all = true;

    all &= ExampledBased::init();
    all &= BTV_Image::init();

    all &= Nlm::init();

    return all;
}

////////////////////////////////////////////////////
// ImageSuperResolution

Ptr<ImageSuperResolution> cv::superres::ImageSuperResolution::create(ImageSRMethod method)
{
    typedef Ptr<ImageSuperResolution> (*func_t)();
    static const func_t funcs[] =
    {
        ExampledBased::create,
        BTV_Image::create
    };

    CV_DbgAssert(method >= IMAGE_SR_EXAMPLE_BASED && method < IMAGE_SR_METHOD_MAX);

    return funcs[method]();
}

cv::superres::ImageSuperResolution::~ImageSuperResolution()
{
}

////////////////////////////////////////////////////
// VideoSuperResolution

Ptr<VideoSuperResolution> cv::superres::VideoSuperResolution::create(VideoSRMethod method)
{
    typedef Ptr<VideoSuperResolution> (*func_t)();
    static const func_t funcs[] =
    {
        Nlm::create,
        BTV_Video::create
    };

    CV_DbgAssert(method >= VIDEO_SR_NLM && method < VIDEO_SR_METHOD_MAX);

    return funcs[method]();
}

cv::superres::VideoSuperResolution::~VideoSuperResolution()
{
}

cv::superres::VideoSuperResolution::VideoSuperResolution()
{
    firstCall = true;
    frameSource = Ptr<IFrameSource>(new NullFrameSource());
}

void cv::superres::VideoSuperResolution::setFrameSource(const Ptr<IFrameSource>& frameSource)
{
    this->frameSource = frameSource;
    reset();
}

Mat cv::superres::VideoSuperResolution::nextFrame()
{
    if (firstCall)
    {
        initImpl(frameSource);
        firstCall = false;
    }

    Mat frame = frameSource->nextFrame();

    if (frame.empty())
        return Mat();

    return processImpl(frame);
}

void cv::superres::VideoSuperResolution::reset()
{
    firstCall = true;
}
