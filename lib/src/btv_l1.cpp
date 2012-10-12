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

    curBlurKernelSize = -1;
    curBlurSigma = -1.0;
    curSrcType = -1;
}

namespace
{
    void calcOpticalFlow(const Mat& frame0, const Mat& frame1, Mat_<Point2f>& flow, Mat& gray0, Mat& gray1)
    {
        CV_DbgAssert( frame0.depth() == CV_8U );
        CV_DbgAssert( frame0.channels() == 1 || frame0.channels() == 3 || frame0.channels() == 4 );
        CV_DbgAssert( frame1.type() == frame0.type() );
        CV_DbgAssert( frame1.size() == frame0.size() );

        Mat input0, input1;

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

        calcOpticalFlowFarneback(input0, input1, flow,
                                 /*pyrScale =*/ 0.5,
                                 /*numLevels =*/ 5,
                                 /*winSize =*/ 13,
                                 /*numIters =*/ 10,
                                 /*polyN =*/ 5,
                                 /*polySigma =*/ 1.1,
                                 /*flags =*/ 0);
    }

    void calcMotions(const vector<Mat>& src, int startIdx, int procIdx, int endIdx, vector<Mat_<Point2f> >& motions, Mat& gray0, Mat& gray1)
    {
        CV_DbgAssert( !src.empty() );
        CV_DbgAssert( startIdx <= procIdx && procIdx <= endIdx );

        motions.resize(src.size());

        for (int i = startIdx; i <= endIdx; ++i)
            calcOpticalFlow(at(i, src), at(procIdx, src), at(i, motions), gray0, gray1);
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
    void calcBtvRegularizationImpl(const Mat& X, Mat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        CV_DbgAssert( X.type() == DataType<T>::type );
        CV_DbgAssert( btvKernelSize > 0 );
        CV_DbgAssert( btvWeights.size() == btvKernelSize * btvKernelSize );

        dst.create(X.size(), X.type());
        dst.setTo(Scalar::all(0));

        const int ksize = (btvKernelSize - 1) / 2;

        BtvRegularizationBody<T> body;

        body.src = X;
        body.dst = dst;
        body.ksize = ksize;
        body.btvWeights = &btvWeights[0];

        parallel_for_(Range(ksize, X.rows - ksize), body);
    }

    void calcBtvRegularization(const Mat& X, Mat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        typedef void (*func_t)(const Mat& X, Mat& dst, int btvKernelSize, const vector<float>& btvWeights);
        static const func_t funcs[] =
        {
            0, calcBtvRegularizationImpl<float>, 0, calcBtvRegularizationImpl<Point3f>
        };

        CV_DbgAssert( X.depth() == CV_32F );
        CV_DbgAssert( X.channels() == 1 || X.channels() == 3 );

        const func_t func = funcs[X.channels()];

        func(X, dst, btvKernelSize, btvWeights);
    }
}

void cv::superres::BTV_L1_Base::process(const vector<Mat>& src, Mat& dst, int startIdx, int procIdx, int endIdx)
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
    CV_DbgAssert( blurKernelSize > 0 );

    // convert sources to float

    src_f.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        src[i].convertTo(src_f[i], CV_32F);

    // calc motions between input frames

    calcMotions(src, startIdx, procIdx, endIdx, lowResMotions, gray0, gray1);
    upscaleMotions(lowResMotions, highResMotions, scale);

    forward.resize(highResMotions.size());
    backward.resize(highResMotions.size());
    for (size_t i = 0; i < highResMotions.size(); ++i)
        buildMotionMaps(highResMotions[i], forward[i], backward[i]);

    // update blur filter and btv weights

    if (filter.empty() || blurKernelSize != curBlurKernelSize || blurSigma != curBlurSigma || src_f[0].type() != curSrcType)
    {
        filter = createGaussianFilter(src_f[0].type(), Size(blurKernelSize, blurKernelSize), blurSigma);
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
            remap(highRes, a, backward[k], noArray(), INTER_NEAREST);
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
            remap(b, a, forward[k], noArray(), INTER_NEAREST);

            add(diffTerm, a, diffTerm);
        }

        addWeighted(highRes, 1.0, diffTerm, tau, 0.0, highRes);

        if (lambda > 0)
        {
            calcBtvRegularization(highRes, regTerm, btvKernelSize, btvWeights);
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
                                               "Radius of the temporal search area."));
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
}

void cv::superres::BTV_L1::initImpl(Ptr<IFrameSource>& frameSource)
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
        return at(outPos, results);
    }

    return Mat();
}

void cv::superres::BTV_L1::addNewFrame(const Mat& frame)
{
    if (frame.empty())
        return;

    CV_DbgAssert( storePos < 0 || frame.size() == at(storePos, frames).size() );

    ++storePos;
    frame.copyTo(at(storePos, frames));
}

void cv::superres::BTV_L1::processFrame(int idx)
{
    process(frames, at(idx, results), idx - temporalAreaRadius, idx, idx + temporalAreaRadius);
}
