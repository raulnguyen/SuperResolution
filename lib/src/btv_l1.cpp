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

#include "btv_l1.hpp"
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videostab/ring_buffer.hpp>

using namespace std;
using namespace cv;
using namespace cv::videostab;
using namespace cv::superres;

namespace
{
    void calcMotions(const vector<Mat_<Point2f> >& relMotions, vector<Mat_<Point2f> >& motions, int baseIdx, Size size)
    {
        CV_DbgAssert( baseIdx >= 0 && baseIdx <= relMotions.size() );
        #ifdef _DEBUG
        for (size_t i = 0; i < relMotions.size(); ++i)
            CV_DbgAssert( relMotions[i].size() == size );
        #endif

        motions.resize(relMotions.size() + 1);

        motions[baseIdx].create(size);
        motions[baseIdx].setTo(Scalar::all(0));

        for (int i = baseIdx - 1; i >= 0; --i)
            add(motions[i + 1], relMotions[i], motions[i]);

        for (size_t i = baseIdx + 1; i < motions.size(); ++i)
            subtract(motions[i - 1], relMotions[i - 1], motions[i]);
    }

    void upscaleMotions(const vector<Mat_<Point2f> >& lowResMotions, vector<Mat_<Point2f> >& highResMotions, int scale)
    {
        CV_DbgAssert( !lowResMotions.empty() );
        #ifdef _DEBUG
        for (size_t i = 1; i < lowResMotions.size(); ++i)
            CV_DbgAssert( lowResMotions[i].size() == lowResMotions[0].size() );
        #endif
        CV_DbgAssert( scale > 1 );

        highResMotions.resize(lowResMotions.size());

        for (size_t i = 0; i < lowResMotions.size(); ++i)
        {
            resize(lowResMotions[i], highResMotions[i], Size(), scale, scale, INTER_LINEAR);
            multiply(highResMotions[i], Scalar::all(scale), highResMotions[i]);
        }
    }

    void buildMotionMaps(const Mat_<Point2f>& motion, Mat_<Point2f>& forward, Mat_<Point2f>& backward)
    {
        forward.create(motion.size());
        backward.create(motion.size());

        for (int y = 0; y < motion.rows; ++y)
        {
            const Point2f* motionRow = motion[y];
            Point2f* forwardRow = forward[y];
            Point2f* backwardRow = backward[y];

            for (int x = 0; x < motion.cols; ++x)
            {
                Point2f base(x, y);

                forwardRow[x] = base - motionRow[x];
                backwardRow[x] = base + motionRow[x];
            }
        }
    }

    template <typename T>
    void upscaleImpl(const Mat& src, Mat& dst, int scale)
    {
        CV_DbgAssert( src.type() == DataType<T>::type );
        CV_DbgAssert( scale > 1 );

        dst.create(src.rows * scale, src.cols * scale, src.type());
        dst.setTo(Scalar::all(0));

        for (int y = 0, Y = 0; y < src.rows; ++y, Y += scale)
        {
            const T* srcRow = src.ptr<T>(y);
            T* dstRow = dst.ptr<T>(Y);

            for (int x = 0, X = 0; x < src.cols; ++x, X += scale)
                dstRow[X] = srcRow[x];
        }
    }

    void upscale(const Mat& src, Mat& dst, int scale)
    {
        typedef void (*func_t)(const Mat& src, Mat& dst, int scale);
        static const func_t funcs[] =
        {
            0, upscaleImpl<float>, 0, upscaleImpl<Point3f>
        };

        CV_DbgAssert( src.depth() == CV_32F );
        CV_DbgAssert( src.channels() == 1 || src.channels() == 3 );

        const func_t func = funcs[src.channels()];

        func(src, dst, scale);
    }

    float diffSign(float a, float b)
    {
        return a > b ? 1.0f : a < b ? -1.0f : 0.0f;
    }
    Point3f diffSign(Point3f a, Point3f b)
    {
        return Point3f(
            a.x > b.x ? 1.0f : a.x < b.x ? -1.0f : 0.0f,
            a.y > b.y ? 1.0f : a.y < b.y ? -1.0f : 0.0f,
            a.z > b.z ? 1.0f : a.z < b.z ? -1.0f : 0.0f
        );
    }

    void diffSign(const Mat& src1, const Mat& src2, Mat& dst)
    {
        CV_DbgAssert( src1.depth() == CV_32F );
        CV_DbgAssert( src2.size() == src1.size() );
        CV_DbgAssert( src2.type() == src1.type() );

        const int count = src1.cols * src1.channels();

        dst.create(src1.size(), src1.type());

        for (int y = 0; y < src1.rows; ++y)
        {
            const float* src1Ptr = src1.ptr<float>(y);
            const float* src2Ptr = src2.ptr<float>(y);
            float* dstPtr = dst.ptr<float>(y);

            for (int x = 0; x < count; ++x)
                dstPtr[x] = diffSign(src1Ptr[x], src2Ptr[x]);
        }
    }

    void calcBtvWeights(int btvKernelSize, double alpha, vector<float>& btvWeights)
    {
        CV_DbgAssert( btvKernelSize > 0 );
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
    }

    template <typename T>
    struct BtvRegularizationBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        Mat src;
        mutable Mat dst;
        int ksize;
        const float* btvWeights;
    };

    template <typename T>
    void BtvRegularizationBody<T>::operator ()(const Range& range) const
    {
        CV_DbgAssert( src.type() == DataType<T>::type );
        CV_DbgAssert( dst.size() == src.size() );
        CV_DbgAssert( dst.type() == src.type() );
        CV_DbgAssert( ksize > 0 );
        CV_DbgAssert( range.start >= ksize );
        CV_DbgAssert( range.end <= src.rows - ksize );

        for (int i = range.start; i < range.end; ++i)
        {
            const T* srcRow = src.ptr<T>(i);
            T* dstRow = dst.ptr<T>(i);

            for(int j = ksize; j < src.cols - ksize; ++j)
            {
                const T srcVal = srcRow[j];

                for (int m = 0, count = 0; m <= ksize; ++m)
                {
                    const T* srcRow2 = src.ptr<T>(i - m);
                    const T* srcRow3 = src.ptr<T>(i + m);

                    for (int l = ksize; l + m >= 0; --l, ++count)
                    {
                        CV_DbgAssert( j + l >= 0 && j + l < src.cols );
                        CV_DbgAssert( j - l >= 0 && j - l < src.cols );

                        dstRow[j] += btvWeights[count] * (diffSign(srcVal, srcRow3[j + l]) - diffSign(srcRow2[j - l], srcVal));
                    }
                }
            }
        }
    }

    template <typename T>
    void calcBtvRegularizationImpl(const Mat& src, Mat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        CV_DbgAssert( src.type() == DataType<T>::type );
        CV_DbgAssert( btvKernelSize > 0 );
        CV_DbgAssert( btvWeights.size() == btvKernelSize * btvKernelSize );

        dst.create(src.size(), src.type());
        dst.setTo(Scalar::all(0));

        const int ksize = (btvKernelSize - 1) / 2;

        BtvRegularizationBody<T> body;

        body.src = src;
        body.dst = dst;
        body.ksize = ksize;
        body.btvWeights = &btvWeights[0];

        parallel_for_(Range(ksize, src.rows - ksize), body);
    }

    void calcBtvRegularization(const Mat& src, Mat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        typedef void (*func_t)(const Mat& src, Mat& dst, int btvKernelSize, const vector<float>& btvWeights);
        static const func_t funcs[] =
        {
            0, calcBtvRegularizationImpl<float>, 0, calcBtvRegularizationImpl<Point3f>
        };

        CV_DbgAssert( src.depth() == CV_32F );
        CV_DbgAssert( src.channels() == 1 || src.channels() == 3 );

        const func_t func = funcs[src.channels()];

        func(src, dst, btvKernelSize, btvWeights);
    }
}

cv::superres::BTV_L1_Base::BTV_L1_Base()
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

void cv::superres::BTV_L1_Base::process(const vector<Mat>& src, Mat& dst, const vector<Mat_<Point2f> >& relMotions, int baseIdx)
{
    CV_DbgAssert( !src.empty() );
#ifdef _DEBUG
    for (size_t i = 1; i < src.size(); ++i)
    {
        CV_DbgAssert( src[i].size() == src[0].size() );
        CV_DbgAssert( src[i].type() == src[0].type() );
    }
#endif
    CV_DbgAssert( relMotions.size() == src.size() - 1 );
#ifdef _DEBUG
    for (size_t i = 1; i < relMotions.size(); ++i)
        CV_DbgAssert( relMotions[i].size() == src[0].size() );
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

    const vector<Mat>* yPtr;
    if (src[0].depth() == CV_32F)
        yPtr = &src;
    else
    {
        src_f.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            src[i].convertTo(src_f[i], CV_32F);
        yPtr = &src_f;
    }
    const vector<Mat>& y = *yPtr;

    // calc motions between input frames

    calcMotions(relMotions, lowResMotions, baseIdx, src[0].size());
    upscaleMotions(lowResMotions, highResMotions, scale);

    forward.resize(highResMotions.size());
    backward.resize(highResMotions.size());
    for (size_t i = 0; i < highResMotions.size(); ++i)
        buildMotionMaps(highResMotions[i], forward[i], backward[i]);

    // update blur filter and btv weights

    if (filter.empty() || blurKernelSize != curBlurKernelSize || blurSigma != curBlurSigma || y[0].type() != curSrcType)
    {
        filter = createGaussianFilter(y[0].type(), Size(blurKernelSize, blurKernelSize), blurSigma);
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
            remap(highRes, a, backward[k], noArray(), INTER_NEAREST);
            // b = HM * Ih
            filter->apply(a, b);
            // c = DHM * Ih
            resize(b, c, lowResSize, 0, 0, INTER_NEAREST);

            diffSign(y[k], c, diff);

            // d = Dt * diff
            upscale(diff, d, scale);
            // b = HtDt * diff
            filter->apply(d, b);
            // a = MtHtDt * diff
            remap(b, a, forward[k], noArray(), INTER_NEAREST);

            add(diffTerm, a, diffTerm);
        }

        if (lambda > 0)
        {
            calcBtvRegularization(highRes, regTerm, btvKernelSize, btvWeights);
            addWeighted(diffTerm, 1.0, regTerm, -lambda, 0.0, diffTerm);
        }

        addWeighted(highRes, 1.0, diffTerm, tau, 0.0, highRes);
    }

    Rect inner(btvKernelSize, btvKernelSize, highRes.cols - 2 * btvKernelSize, highRes.rows - 2 * btvKernelSize);
    highRes(inner).convertTo(dst, src[0].depth());
}

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(BTV_L1, "SuperResolution.BTV_L1",
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
                                               "Radius of the temporal search area.");
                          obj.info()->addParam<DenseOpticalFlow>(obj, "opticalFlow", obj.opticalFlow, false, 0, 0,
                                               "Dense optical flow algorithm."));
    }
}

bool cv::superres::BTV_L1::init()
{
    return !BTV_L1_info_auto.name().empty();
}

Ptr<SuperResolution> cv::superres::BTV_L1::create()
{
    Ptr<SuperResolution> alg(new BTV_L1);
    return alg;
}

cv::superres::BTV_L1::BTV_L1()
{
    temporalAreaRadius = 4;
    opticalFlow = new Farneback;
}

void cv::superres::BTV_L1::initImpl(Ptr<IFrameSource>& frameSource)
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

Mat cv::superres::BTV_L1::processImpl(Ptr<IFrameSource>& frameSource)
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
        return dst;
    }

    return Mat();
}

void cv::superres::BTV_L1::addNewFrame(const Mat& frame)
{
    if (frame.empty())
        return;

    CV_DbgAssert( storePos < 0 || frame.size() == at(storePos, frames).size() );

    ++storePos;
    frame.convertTo(at(storePos, frames), CV_32F);

    if (storePos > 0)
        opticalFlow->calc(prevFrame, frame, at(storePos - 1, motions));

    frame.copyTo(prevFrame);
}

void cv::superres::BTV_L1::processFrame(int idx)
{
    const int startIdx = std::max(idx - temporalAreaRadius, 0);
    const int procIdx = idx;
    const int endIdx = std::min(startIdx + 2 * temporalAreaRadius, storePos);

    src.resize(endIdx - startIdx + 1);
    relMotions.resize(endIdx - startIdx );
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
