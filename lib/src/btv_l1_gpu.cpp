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
#include <opencv2/core/internal.hpp>
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

    curBlurKernelSize = -1;
    curBlurSigma = -1.0;
    curSrcType = -1;
}

namespace
{
    void calcOpticalFlow(FarnebackOpticalFlow& opticalFlow, const GpuMat& frame0, const GpuMat& frame1, GpuMat& flowx, GpuMat& flowy, GpuMat& gray0, GpuMat& gray1)
    {
        CV_DbgAssert( frame0.depth() == CV_8U );
        CV_DbgAssert( frame0.channels() == 1 || frame0.channels() == 3 || frame0.channels() == 4 );
        CV_DbgAssert( frame1.type() == frame0.type() );
        CV_DbgAssert( frame1.size() == frame0.size() );

        GpuMat input0, input1;

        if (frame0.channels() == 1)
        {
            input0 = frame0;
            input1 = frame1;
        }
        else if (frame0.channels() == 3)
        {
            cvtColor(frame0, gray0, COLOR_BGR2GRAY);
            cvtColor(frame1, gray1, COLOR_BGR2GRAY);
            input0 = gray0;
            input1 = gray1;
        }
        else
        {
            cvtColor(frame0, gray0, COLOR_BGRA2GRAY);
            cvtColor(frame1, gray1, COLOR_BGRA2GRAY);
            input0 = gray0;
            input1 = gray1;
        }

        opticalFlow(input0, input1, flowx, flowy);
    }

    void calcMotions(FarnebackOpticalFlow& opticalFlow, const vector<GpuMat>& src, int startIdx, int procIdx, int endIdx,
                     vector<pair<GpuMat, GpuMat> >& motions, GpuMat& gray0, GpuMat& gray1)
    {
        CV_DbgAssert( !src.empty() );
        CV_DbgAssert( startIdx <= procIdx && procIdx <= endIdx );

        motions.resize(src.size());

        for (int i = startIdx; i <= endIdx; ++i)
            calcOpticalFlow(opticalFlow, at(i, src), at(procIdx, src), at(i, motions).first, at(i, motions).second, gray0, gray1);
    }

    void upscaleMotions(const vector<pair<GpuMat, GpuMat> >& lowResMotions, vector<pair<GpuMat, GpuMat> >& highResMotions, int scale)
    {
        CV_DbgAssert( !lowResMotions.empty() );
        #ifdef _DEBUG
        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            CV_DbgAssert( lowResMotions[i].first.size() == lowResMotions[0].first.size() );
            CV_DbgAssert( lowResMotions[i].second.size() == lowResMotions[0].first.size() );
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
        CV_DbgAssert( btvKernelSize > 0 );
        CV_DbgAssert( alpha > 0 );
        CV_DbgAssert( btvKernelSize > 0 && btvKernelSize <= 16 );

        const size_t size = btvKernelSize * btvKernelSize;

        if (btvWeights.size() != size)
        {
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

void cv::superres::BTV_L1_GPU_Base::process(const vector<GpuMat>& src, GpuMat& dst, int startIdx, int procIdx, int endIdx)
{
    CV_DbgAssert( !src.empty() );
    CV_DbgAssert( procIdx >= startIdx && endIdx >= procIdx );
    CV_DbgAssert( scale > 1 );
#ifdef _DEBUG
    for (size_t i = 1; i < src.size(); ++i)
    {
        CV_DbgAssert( src[i].size() == src[0].size() );
        CV_DbgAssert( src[i].type() == src[0].type() );
    }
#endif
    CV_DbgAssert( blurModel == BLUR_BOX || blurModel == BLUR_GAUSS );
    CV_DbgAssert( blurKernelSize > 0 );

    // convert sources to float

    src_f.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        src[i].convertTo(src_f[i], CV_32F);

    // calc motions between input frames

    calcMotions(opticalFlow, src, startIdx, procIdx, endIdx, lowResMotions, gray0, gray1);
    upscaleMotions(lowResMotions, highResMotions, scale);

    forward.resize(highResMotions.size());
    backward.resize(highResMotions.size());
    for (size_t i = 0; i < highResMotions.size(); ++i)
        buildMotionMaps(highResMotions[i], forward[i], backward[i]);

    // update blur filter and btv weights

    if (filter.empty() || blurKernelSize != curBlurKernelSize || blurSigma != curBlurSigma || src_f[0].type() != curSrcType)
    {
        filter = createGaussianFilter_GPU(src_f[0].type(), Size(blurKernelSize, blurKernelSize), blurSigma);
        curBlurKernelSize = blurKernelSize;
        curBlurSigma = blurSigma;
        curSrcType = src_f[0].type();
    }

    calcBtvWeights(btvKernelSize, alpha, btvWeights);

    // initial estimation

    const Size lowResSize = src[0].size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

    resize(at(procIdx, src_f), highRes, highResSize, 0, 0, INTER_CUBIC);

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

        for (size_t k = 0; k < src.size(); ++k)
        {
            // a = M * Ih
            remap(highRes, a, backward[k].first, backward[k].second, INTER_NEAREST);
            // b = HM * Ih
            filter->apply(a, b);
            // c = DHF * Ih
            resize(b, c, lowResSize, 0, 0, INTER_NEAREST);

            diffSign(src_f[k], c, diff);

            // d = Dt * diff
            upscale(diff, d, scale);
            // b = HtDt * diff
            filter->apply(d, b);
            // a = MtHtDt * diff
            remap(b, a, forward[k].first, forward[k].second, INTER_NEAREST);

            add(diffTerm, a, diffTerm);
        }

        addWeighted(highRes, 1.0, diffTerm, tau, 0.0, highRes);

        if (lambda > 0)
        {
            calcBtvRegularization(highRes, regTerm, btvKernelSize);
            addWeighted(highRes, 1.0, regTerm, -tau * lambda, 0.0, highRes);
        }
    }

    Rect inner(btvKernelSize, btvKernelSize, highRes.cols - 2 * btvKernelSize, highRes.rows - 2 * btvKernelSize);
    highRes(inner).convertTo(dst, src[0].depth());
}

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(BTV_L1_GPU, "SuperResolution.BTV_L1_GPU",
                          obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0,
                                               "Scale factor.");
                          obj.info()->addParam(obj, "iterations", obj.iterations, false, 0, 0,
                                               "Iteration count.");
                          obj.info()->addParam(obj, "tau", obj.tau, false, 0, 0,
                                               "Asymptotic value of steepest descent method.");
                          obj.info()->addParam(obj, "lambda", obj.lambda, false, 0, 0,
                                               "Weight parameter to balance data term and smoothness term.");
                          obj.info()->addParam(obj, "alpha", obj.alpha, false, 0, 0,
                                               "Parameter of spacial distribution in btv.");
                          obj.info()->addParam(obj, "btvKernelSize", obj.btvKernelSize, false, 0, 0,
                                               "Kernel size of btv filter.");
                          obj.info()->addParam(obj, "blurKernelSize", obj.blurKernelSize, false, 0, 0,
                                               "Gaussian blur kernel size.");
                          obj.info()->addParam(obj, "blurSigma", obj.blurSigma, false, 0, 0,
                                               "Gaussian blur sigma.");
                          obj.info()->addParam(obj, "temporalAreaRadius", obj.temporalAreaRadius, false, 0, 0,
                                               "Radius of the temporal search area."));
    }
}

bool cv::superres::BTV_L1_GPU::init()
{
    return !BTV_L1_GPU_info_auto.name().empty();
}

Ptr<SuperResolution> cv::superres::BTV_L1_GPU::create()
{
    Ptr<SuperResolution> alg(new BTV_L1_GPU);
    return alg;
}

cv::superres::BTV_L1_GPU::BTV_L1_GPU()
{
    temporalAreaRadius = 4;
}

void cv::superres::BTV_L1_GPU::initImpl(Ptr<IFrameSource>& frameSource)
{
    const int cacheSize = 2 * temporalAreaRadius + 1;

    frames.resize(cacheSize);
    results.resize(cacheSize);

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
        at(outPos, results).download(h_dst);
        return h_dst;
    }

    return Mat();
}

void cv::superres::BTV_L1_GPU::addNewFrame(const Mat& frame)
{
    if (frame.empty())
        return;

    CV_DbgAssert( storePos < 0 || frame.size() == at(storePos, frames).size() );

    ++storePos;
    at(storePos, frames).upload(frame);
}

void cv::superres::BTV_L1_GPU::processFrame(int idx)
{
    process(frames, at(idx, results), idx - temporalAreaRadius, idx, idx + temporalAreaRadius);
}
