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

#include "btv_l1_gpu.hpp"
#include "super_resolution.hpp"
#include <opencv2/videostab/ring_buffer.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::videostab;
using namespace cv::superres;

namespace btv_l1_device
{
    void buildMotionMaps(PtrStepSzf motionx, PtrStepSzf motiony,
                         PtrStepSzf forwardx, PtrStepSzf forwardy,
                         PtrStepSzf backwardx, PtrStepSzf backwardy);

    template <int cn>
    void upscale(const PtrStepSzb src, PtrStepSzb dst, int scale);

    void diffSign(PtrStepSzf src1, PtrStepSzf src2, PtrStepSzf dst);

    void loadBtvWeights(const float* weights, size_t count);
    template <int cn> void calcBtvRegularization(PtrStepSzb src, PtrStepSzb dst, int ksize);
}

namespace
{
    void calcMotions(const vector<pair<GpuMat, GpuMat> >& relMotions, vector<pair<GpuMat, GpuMat> >& motions, int baseIdx, Size size)
    {
        CV_DbgAssert( baseIdx >= 0 && baseIdx <= relMotions.size() );
        #ifdef _DEBUG
        for (size_t i = 0; i < relMotions.size(); ++i)
        {
            CV_DbgAssert( relMotions[i].first.size() == size );
            CV_DbgAssert( relMotions[i].second.size() == size );
            CV_DbgAssert( relMotions[i].first.type() == CV_32FC1 );
            CV_DbgAssert( relMotions[i].second.type() == CV_32FC1 );
        }
        #endif

        motions.resize(relMotions.size() + 1);

        motions[baseIdx].first.create(size, CV_32FC1);
        motions[baseIdx].first.setTo(Scalar::all(0));

        motions[baseIdx].second.create(size, CV_32FC1);
        motions[baseIdx].second.setTo(Scalar::all(0));

        for (int i = baseIdx - 1; i >= 0; --i)
        {
            add(motions[i + 1].first, relMotions[i].first, motions[i].first);
            add(motions[i + 1].second, relMotions[i].second, motions[i].second);
        }

        for (size_t i = baseIdx + 1; i < motions.size(); ++i)
        {
            subtract(motions[i - 1].first, relMotions[i - 1].first, motions[i].first);
            subtract(motions[i - 1].second, relMotions[i - 1].second, motions[i].second);
        }
    }

    void upscaleMotions(const vector<pair<GpuMat, GpuMat> >& lowResMotions, vector<pair<GpuMat, GpuMat> >& highResMotions, int scale)
    {
        CV_DbgAssert( !lowResMotions.empty() );
        #ifdef _DEBUG
        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            CV_DbgAssert( lowResMotions[i].first.size() == lowResMotions[0].first.size() );
            CV_DbgAssert( lowResMotions[i].second.size() == lowResMotions[0].first.size() );
            CV_DbgAssert( lowResMotions[i].first.type() == CV_32FC1 );
            CV_DbgAssert( lowResMotions[i].second.type() == CV_32FC1 );
        }
        #endif
        CV_DbgAssert( scale > 1 );

        highResMotions.resize(lowResMotions.size());

        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            resize(lowResMotions[i].first, highResMotions[i].first, Size(), scale, scale, INTER_LINEAR);
            resize(lowResMotions[i].second, highResMotions[i].second, Size(), scale, scale, INTER_LINEAR);

            multiply(highResMotions[i].first, Scalar::all(scale), highResMotions[i].first);
            multiply(highResMotions[i].second, Scalar::all(scale), highResMotions[i].second);
        }
    }

    void buildMotionMaps(const pair<GpuMat, GpuMat>& motion, pair<GpuMat, GpuMat>& forward, pair<GpuMat, GpuMat>& backward)
    {
        CV_DbgAssert( motion.first.type() == CV_32FC1 );
        CV_DbgAssert( motion.second.type() == motion.first.type() );
        CV_DbgAssert( motion.second.size() == motion.first.size() );

        forward.first.create(motion.first.size(), motion.first.type());
        forward.second.create(motion.first.size(), motion.first.type());

        backward.first.create(motion.first.size(), motion.first.type());
        backward.second.create(motion.first.size(), motion.first.type());

        btv_l1_device::buildMotionMaps(motion.first, motion.second, forward.first, forward.second, backward.first, backward.second);
    }

    void upscale(const GpuMat& src, GpuMat& dst, int scale)
    {
        typedef void (*func_t)(const PtrStepSzb src, PtrStepSzb dst, int scale);
        static const func_t funcs[] =
        {
            0, btv_l1_device::upscale<1>, 0, btv_l1_device::upscale<3>, btv_l1_device::upscale<4>
        };

        CV_DbgAssert( src.depth() == CV_32F );
        CV_DbgAssert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );
        CV_DbgAssert( scale > 1 );

        dst.create(src.rows * scale, src.cols * scale, src.type());
        dst.setTo(Scalar::all(0));

        const func_t func = funcs[src.channels()];

        func(src, dst, scale);
    }

    void diffSign(const GpuMat& src1, const GpuMat& src2, GpuMat& dst)
    {
        CV_DbgAssert( src1.depth() == CV_32F );
        CV_DbgAssert( src1.type() == src2.type() );
        CV_DbgAssert( src1.size() == src2.size() );

        dst.create(src1.size(), src1.type());

        btv_l1_device::diffSign(src1.reshape(1), src2.reshape(1), dst.reshape(1));
    }

    void calcBtvWeights(int btvKernelSize, double alpha, vector<float>& btvWeights)
    {
        CV_DbgAssert( btvKernelSize > 0 && btvKernelSize <= 16 );
        CV_DbgAssert( alpha > 0 );

        const size_t size = btvKernelSize * btvKernelSize;

        btvWeights.resize(size);

        const int ksize = (btvKernelSize - 1) / 2;
        const float alpha_f = static_cast<float>(alpha);

        for (int m = 0, ind = 0; m <= ksize; ++m)
        {
            for (int l = ksize; l + m >= 0; --l, ++ind)
                btvWeights[ind] = pow(alpha_f, std::abs(m) + std::abs(l));
        }

        btv_l1_device::loadBtvWeights(&btvWeights[0], size);
    }

    void calcBtvRegularization(const GpuMat& src, GpuMat& dst, int btvKernelSize)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, int ksize);
        static const func_t funcs[] =
        {
            0,
            btv_l1_device::calcBtvRegularization<1>,
            0,
            btv_l1_device::calcBtvRegularization<3>,
            btv_l1_device::calcBtvRegularization<4>
        };

        CV_DbgAssert( src.depth() == CV_32F );
        CV_DbgAssert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );
        CV_DbgAssert( btvKernelSize > 0 && btvKernelSize <= 16 );

        dst.create(src.size(), src.type());
        dst.setTo(Scalar::all(0));

        const int ksize = (btvKernelSize - 1) / 2;

        funcs[src.channels()](src, dst, ksize);
    }
}

cv::superres::BTV_L1_GPU_Base::BTV_L1_GPU_Base()
{
    scale = 4;
    iterations = 180;
    lambda = 0.03;
    tau = 1.3;
    alpha = 0.7;
    btvKernelSize = 7;
    blurKernelSize = 5;
    blurSigma = 0.0;

    curBtvKernelSize = -1;
    curAlpha = -1.0;

    curBlurKernelSize = -1;
    curBlurSigma = -1.0;
    curSrcType = -1;
}

void cv::superres::BTV_L1_GPU_Base::process(const vector<GpuMat>& src, GpuMat& dst, const vector<pair<GpuMat, GpuMat> >& motions, int baseIdx)
{
    CV_DbgAssert( !src.empty() );
#ifdef _DEBUG
    for (size_t i = 1; i < src.size(); ++i)
    {
        CV_DbgAssert( src[i].size() == src[0].size() );
        CV_DbgAssert( src[i].type() == src[0].type() );
    }
#endif
    CV_DbgAssert( motions.size() == src.size() - 1 );
#ifdef _DEBUG
    for (size_t i = 1; i < motions.size(); ++i)
    {
        CV_DbgAssert( motions[i].first.size() == src[0].size() );
        CV_DbgAssert( motions[i].second.size() == src[0].size() );
    }
#endif
    CV_DbgAssert( baseIdx >= 0 && baseIdx < src.size() );
    CV_DbgAssert( scale > 1 );
    CV_DbgAssert( iterations > 0 );
    CV_DbgAssert( tau > 0.0 );
    CV_DbgAssert( alpha > 0.0 );
    CV_DbgAssert( btvKernelSize > 0 );
    CV_DbgAssert( blurKernelSize > 0 );
    CV_DbgAssert( blurSigma >= 0.0 );

    // convert sources to float

    const vector<GpuMat>* yPtr;
    if (src[0].depth() == CV_32F)
        yPtr = &src;
    else
    {
        src_f.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            src[i].convertTo(src_f[i], CV_32F);
        yPtr = &src_f;
    }
    const vector<GpuMat>& y = *yPtr;

    // calc motions between input frames

    calcMotions(motions, lowResMotions, baseIdx, src[0].size());
    upscaleMotions(lowResMotions, highResMotions, scale);

    forward.resize(highResMotions.size());
    backward.resize(highResMotions.size());
    for (size_t i = 0; i < highResMotions.size(); ++i)
        buildMotionMaps(highResMotions[i], forward[i], backward[i]);

    // update blur filter and btv weights

    if (filter.empty() || blurKernelSize != curBlurKernelSize || blurSigma != curBlurSigma || y[0].type() != curSrcType)
    {
        filter = createGaussianFilter_GPU(y[0].type(), Size(blurKernelSize, blurKernelSize), blurSigma);
        curBlurKernelSize = blurKernelSize;
        curBlurSigma = blurSigma;
        curSrcType = y[0].type();
    }

    if (btvWeights.empty() || btvKernelSize != curBtvKernelSize || alpha != curAlpha)
    {
        calcBtvWeights(btvKernelSize, alpha, btvWeights);
        curBtvKernelSize = btvKernelSize;
        curAlpha = alpha;
    }

    // initial estimation

    const Size lowResSize = y[0].size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

    resize(y[baseIdx], highRes, highResSize, 0, 0, INTER_CUBIC);

    // iterations

    diffTerm.create(highResSize, highRes.type());
    a.create(highResSize, highRes.type());
    b.create(highResSize, highRes.type());
    c.create(lowResSize, highRes.type());
    diff.create(lowResSize, highRes.type());
    d.create(highResSize, highRes.type());

    for (int i = 0; i < iterations; ++i)
    {
        diffTerm.setTo(Scalar::all(0));

        for (size_t k = 0; k < y.size(); ++k)
        {
            // a = M * Ih
            remap(highRes, a, backward[k].first, backward[k].second, INTER_NEAREST);
            // b = HM * Ih
            filter->apply(a, b);
            // c = DHF * Ih
            resize(b, c, lowResSize, 0, 0, INTER_NEAREST);

            diffSign(y[k], c, diff);

            // d = Dt * diff
            upscale(diff, d, scale);
            // b = HtDt * diff
            filter->apply(d, b);
            // a = MtHtDt * diff
            remap(b, a, forward[k].first, forward[k].second, INTER_NEAREST);

            add(diffTerm, a, diffTerm);
        }

        if (lambda > 0)
        {
            calcBtvRegularization(highRes, regTerm, btvKernelSize);
            addWeighted(diffTerm, 1.0, regTerm, -lambda, 0.0, diffTerm);
        }

        addWeighted(highRes, 1.0, diffTerm, tau, 0.0, highRes);
    }

    Rect inner(btvKernelSize, btvKernelSize, highRes.cols - 2 * btvKernelSize, highRes.rows - 2 * btvKernelSize);
    highRes(inner).convertTo(dst, src[0].depth());
}

cv::superres::BTV_L1_GPU::BTV_L1_GPU()
{
    temporalAreaRadius = 4;
    opticalFlow = new FarnebackOpticalFlow_GPU;
}

void cv::superres::BTV_L1_GPU::initImpl(Ptr<IFrameSource>& frameSource)
{
    const int cacheSize = 2 * temporalAreaRadius + 1;

    frames.resize(cacheSize);
    results.resize(cacheSize);
    motions.resize(cacheSize);

    storePos = -1;

    for (int t = -temporalAreaRadius; t <= temporalAreaRadius; ++t)
    {
        Mat frame = frameSource->nextFrame();
        CV_Assert( !frame.empty() );
        addNewFrame(frame);
    }

    for (int i = 0; i <= temporalAreaRadius; ++i)
        processFrame(i);

    procPos = temporalAreaRadius;
    outPos = -1;
}

Mat cv::superres::BTV_L1_GPU::processImpl(Ptr<IFrameSource>& frameSource)
{
    Mat frame = frameSource->nextFrame();
    addNewFrame(frame);

    if (procPos < storePos)
    {
        ++procPos;
        processFrame(procPos);
    }

    if (outPos < storePos)
    {
        ++outPos;
        at(outPos, results).convertTo(dst, CV_8U);
        dst.download(h_dst);
        return h_dst;
    }

    return Mat();
}

void cv::superres::BTV_L1_GPU::addNewFrame(const Mat& frame)
{
    if (frame.empty())
        return;

    CV_DbgAssert( storePos < 0 || frame.size() == at(storePos, frames).size() );

    d_frame.upload(frame);

    ++storePos;
    d_frame.convertTo(at(storePos, frames), CV_32F);

    if (storePos > 0)
        opticalFlow->calc(prevFrame, d_frame, at(storePos - 1, motions).first, at(storePos - 1, motions).second);

    d_frame.copyTo(prevFrame);
}

void cv::superres::BTV_L1_GPU::processFrame(int idx)
{
    const int startIdx = max(idx - temporalAreaRadius, 0);
    const int procIdx = idx;
    const int endIdx = min(startIdx + 2 * temporalAreaRadius, storePos);

    src.resize(endIdx - startIdx + 1);
    relMotions.resize(endIdx - startIdx);
    int baseIdx = -1;

    for (int i = startIdx, k = 0; i <= endIdx; ++i, ++k)
    {
        if (i == procIdx)
            baseIdx = k;

        src[k] = at(i, frames);

        if (i < endIdx)
            relMotions[k] = at(i, motions);
    }

    process(src, at(idx, results), relMotions, baseIdx);
}
