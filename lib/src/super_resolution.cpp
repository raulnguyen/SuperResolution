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
#include "btv.hpp"
#include "btv_gpu.hpp"
#include "tv-l1.hpp"
#include "tv-l1_gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

bool cv::superres::initModule_superres()
{
    bool all = true;

    all &= BTV::init();
    all &= BTV_GPU::init();

    all &= TV_L1::init();
    all &= TV_L1_GPU::init();

    return all;
}

Ptr<SuperResolution> cv::superres::SuperResolution::create(SRMethod method, bool useGpu)
{
    typedef Ptr<SuperResolution> (*func_t)();
    static const func_t cpu_funcs[] =
    {
        BTV::create,
        TV_L1::create
    };
    static const func_t gpu_funcs[] =
    {
        BTV_GPU::create,
        TV_L1_GPU::create
    };

    CV_DbgAssert(method >= SR_BILATERAL_TOTAL_VARIATION && method < SR_METHOD_MAX);

    const func_t func = (useGpu ? gpu_funcs : cpu_funcs)[method];
    CV_Assert( func != 0 );

    return func();
}

cv::superres::SuperResolution::~SuperResolution()
{
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
