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

#include "btv.hpp"
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videostab/ring_buffer.hpp>
#ifdef WITH_TESTS
    #include <opencv2/ts/ts_gtest.h>
#endif

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;

/////////////////////////////////////////////////////////////
// BTV_Base

cv::superres::BTV_Base::BTV_Base()
{
    scale = 4;
    iterations = 180;
    beta = 1.3;
    lambda = 0.03;
    alpha = 0.7;
    btvKernelSize = 7;
}

namespace
{
    void calcBtvWeights(int btvKernelSize, double alpha, vector<float>& btvWeights)
    {
        CV_DbgAssert(btvKernelSize > 0);
        CV_DbgAssert(alpha > 0);

        const size_t size = btvKernelSize * btvKernelSize;

        if (btvWeights.size() != size)
        {
            btvWeights.resize(size);

            const int ksize = (btvKernelSize - 1) / 2;

            for (int m = 0, ind = 0; m <= ksize; ++m)
            {
                for (int l = ksize; l + m >= 0; --l, ++ind)
                    btvWeights[ind] = static_cast<float>(pow(alpha, std::abs(m) + std::abs(l)));
            }
        }
    }

    template <typename T>
    void mulDhfMatImpl(const Mat& DHF, const Mat& src, Mat& dst, Size dstSize, bool isTranspose)
    {
        CV_DbgAssert(DHF.type() == CV_32SC3);
        CV_DbgAssert(src.type() == DataType<T>::type);

        dst.create(dstSize, src.type());
        dst.setTo(Scalar::all(0));

        if (isTranspose)
        {
            CV_DbgAssert(DHF.rows == src.size().area());

            for (int y = 0, lowResInd = 0; y < src.rows; ++y)
            {
                const T* srcPtr = src.ptr<T>(y);

                for (int x = 0; x < src.cols; ++x, ++lowResInd)
                {
                    T srcVal = srcPtr[x];

                    const BTV_Base::DHF_Val* DHFPtr = DHF.ptr<BTV_Base::DHF_Val>(lowResInd);

                    for (int i = 0; i < DHF.cols; ++i)
                    {
                        if (DHFPtr[i].coord.x >= 0 && DHFPtr[i].coord.x < dst.cols && DHFPtr[i].coord.y >= 0 && DHFPtr[i].coord.y < dst.rows)
                            dst.at<T>(DHFPtr[i].coord) += DHFPtr[i].weight * srcVal;
                    }
                }
            }
        }
        else
        {
            CV_DbgAssert(DHF.rows == dstSize.area());

            for (int y = 0, lowResInd = 0; y < dstSize.height; ++y)
            {
                T* dstPtr = dst.ptr<T>(y);

                for (int x = 0; x < dstSize.width; ++x, ++lowResInd)
                {
                    const BTV_Base::DHF_Val* DHFPtr = DHF.ptr<BTV_Base::DHF_Val>(lowResInd);

                    for (int i = 0; i < DHF.cols; ++i)
                    {
                        if (DHFPtr[i].coord.x >= 0 && DHFPtr[i].coord.x < src.cols && DHFPtr[i].coord.y >= 0 && DHFPtr[i].coord.y < src.rows)
                            dstPtr[x] += DHFPtr[i].weight * src.at<T>(DHFPtr[i].coord);
                    }
                }
            }
        }
    }

    void mulDhfMat(const Mat& DHF, const Mat& src, Mat& dst, Size dstSize, bool isTranspose = false)
    {
        typedef void (*func_t)(const Mat& DHF, const Mat& src, Mat& dst, Size dstSize, bool isTranspose);
        static const func_t funcs[] =
        {
            0, mulDhfMatImpl<float>, 0, mulDhfMatImpl<Point3f>
        };

        CV_DbgAssert(src.depth() == CV_32F);
        CV_DbgAssert(src.channels() == 1 || src.channels() == 3);

        const func_t func = funcs[src.channels()];

        func(DHF, src, dst, dstSize, isTranspose);
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
        CV_DbgAssert(src1.type() == CV_32FC1);
        CV_DbgAssert(src2.size() == src1.size());
        CV_DbgAssert(src2.type() == src1.type());

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

    void calcBtvDiffTerm(const Mat& y, const Mat& DHF, const Mat& X, Mat& diffTerm, Mat& buf)
    {
        mulDhfMat(DHF, X, buf, y.size());

        diffSign(buf, y, buf);

        mulDhfMat(DHF, buf, diffTerm, X.size(), true);
    }

    struct BtvDiffTermBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        const vector<Mat>* y;
        const vector<Mat>* DHF;
        int count;

        Mat X;

        vector<Mat>* diffTerms;
        vector<Mat>* bufs;
    };

    void BtvDiffTermBody::operator ()(const Range& range) const
    {
        CV_DbgAssert(y && y->size() >= count);
        CV_DbgAssert(DHF && DHF->size() >= count);
        CV_DbgAssert(diffTerms && diffTerms->size() == count);
        CV_DbgAssert(bufs && bufs->size() == count);
        CV_DbgAssert(range.start >= 0);
        CV_DbgAssert(range.end <= count);

        Mat& buf = (*bufs)[range.start];

        for (int i = range.start; i < range.end; ++i)
            calcBtvDiffTerm((*y)[i], (*DHF)[i], X, (*diffTerms)[i], buf);
    }

    void calcBtvDiffTerms(const vector<Mat>& y, const vector<Mat>& DHF, int count, const Mat& X, vector<Mat>& diffTerms, vector<Mat>& bufs)
    {
        diffTerms.resize(count);
        bufs.resize(count);

        BtvDiffTermBody body;

        body.y = &y;
        body.DHF = &DHF;
        body.count = count;
        body.X = X;
        body.diffTerms = &diffTerms;
        body.bufs = &bufs;

        parallel_for_(Range(0, count), body);
    }

    template <typename T>
    struct BtvRegularizationBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        Mat src;
        mutable Mat dst;
        int ksize;
        const vector<float>* btvWeights;
    };

    template <typename T>
    void BtvRegularizationBody<T>::operator ()(const Range& range) const
    {
        CV_DbgAssert(src.type() == DataType<T>::type);
        CV_DbgAssert(dst.size() == src.size());
        CV_DbgAssert(dst.type() == src.type());
        CV_DbgAssert(ksize > 0);
        CV_DbgAssert(range.start >= ksize);
        CV_DbgAssert(range.end <= src.rows - ksize);

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
                        CV_DbgAssert(j + l >= 0 && j + l < src.cols);
                        CV_DbgAssert(j - l >= 0 && j - l < src.cols);

                        dstRow[j] += (*btvWeights)[count] * (diffSign(srcVal, srcRow3[j + l]) - diffSign(srcRow2[j - l], srcVal));
                    }
                }
            }
        }
    }

    template <typename T>
    void calcBtvRegularizationImpl(const Mat& X, Mat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        CV_DbgAssert(X.type() == DataType<T>::type);
        CV_DbgAssert(btvKernelSize > 0);
        CV_DbgAssert(btvWeights.size() == btvKernelSize * btvKernelSize);

        dst.create(X.size(), X.type());
        dst.setTo(Scalar::all(0));

        const int ksize = (btvKernelSize - 1) / 2;

        BtvRegularizationBody<T> body;

        body.src = X;
        body.dst = dst;
        body.ksize = ksize;
        body.btvWeights = &btvWeights;

        parallel_for_(Range(ksize, X.rows - ksize), body);
    }

    void calcBtvRegularization(const Mat& X, Mat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        typedef void (*func_t)(const Mat& X, Mat& dst, int btvKernelSize, const vector<float>& btvWeights);
        static const func_t funcs[] =
        {
            0, calcBtvRegularizationImpl<float>, 0, calcBtvRegularizationImpl<Point3f>
        };

        CV_DbgAssert(X.depth() == CV_32F);
        CV_DbgAssert(X.channels() == 1 || X.channels() == 3);

        const func_t func = funcs[X.channels()];

        func(X, dst, btvKernelSize, btvWeights);
    }
}

void cv::superres::BTV_Base::process(const Mat& src, Mat& dst, const vector<Mat>& y, const vector<Mat>& DHF, int count)
{
    CV_DbgAssert(count > 0);
    CV_DbgAssert(y.size() >= count);
    CV_DbgAssert(DHF.size() >= count);

    calcBtvWeights(btvKernelSize, alpha, btvWeights);

    Size lowResSize = src.size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

#ifdef _DEBUG
    for (int i = 0; i < count; ++i)
    {
        CV_DbgAssert(y[i].size() == lowResSize);
        CV_DbgAssert(DHF[i].rows == lowResSize.area());
    }
#endif

    resize(src, X, highResSize, 0, 0, INTER_CUBIC);

    for (int i = 0; i < iterations; ++i)
    {
        calcBtvDiffTerms(y, DHF, count, X, diffTerms, bufs);

        if (lambda > 0)
            calcBtvRegularization(X, regTerm, btvKernelSize, btvWeights);

        for (size_t n = 0; n < count; ++n)
            addWeighted(X, 1.0, diffTerms[n], -beta, 0.0, X);

        if (lambda > 0.0)
            addWeighted(X, 1.0, regTerm, -beta * lambda, 0.0, X);
    }

    X.convertTo(dst, CV_8U);
}

void cv::superres::BTV_Base::calcBlurWeights(BlurModel blurModel, int blurKernelSize, std::vector<float>& blurWeights)
{
    CV_DbgAssert(blurModel == BLUR_BOX || blurModel == BLUR_GAUSS);
    CV_DbgAssert(blurKernelSize > 0);

    const size_t size = blurKernelSize * blurKernelSize;

    switch (blurModel)
    {
    case BLUR_BOX:
        blurWeights.clear();
        blurWeights.resize(size, 1.0f / size);
        break;

    case BLUR_GAUSS:
        Mat_<float> ker = getGaussianKernel(blurKernelSize, 0, CV_32F);

        blurWeights.resize(size);

        for (int i = 0, ind = 0; i < blurKernelSize; ++i)
            for (int j = 0; j < blurKernelSize; ++j, ++ind)
                blurWeights[ind] = ker(i, 0) * ker(j, 0);
    };
}

namespace
{
    class AffineMotion
    {
    public:
        explicit AffineMotion(const Mat& M) : M(M) {}

        Point2f calcCoord(Point base) const
        {
            Point2f res;
            res.x = M(0, 0) * base.x + M(0, 1) * base.y + M(0, 2);
            res.y = M(1, 0) * base.x + M(1, 1) * base.y + M(1, 2);

            return res;
        }

    private:
        Mat_<float> M;
    };

    class PerspectiveMotion
    {
    public:
        explicit PerspectiveMotion(const Mat& M) : M(M) {}

        Point2f calcCoord(Point base) const
        {
            const float w = 1.0f / (M(2, 0) * base.x + M(2, 1) * base.y + M(2, 2));

            Point2f res;
            res.x = (M(0, 0) * base.x + M(0, 1) * base.y + M(0, 2)) * w;
            res.y = (M(1, 0) * base.x + M(1, 1) * base.y + M(1, 2)) * w;

            return res;
        }

    private:
        Mat_<float> M;
    };

    class GeneralMotion
    {
    public:
        GeneralMotion(const Mat& dx, const Mat& dy) : dx(dx), dy(dy) {}

        Point2f calcCoord(Point base) const
        {
            Point2f res;
            res.x = base.x + dx(base);
            res.y = base.y + dy(base);

            return res;
        }

    private:
        Mat_<float> dx;
        Mat_<float> dy;
    };

    int clamp(int val, int minVal, int maxVal)
    {
        return max(min(val, maxVal), minVal);
    }

    template <class Motion>
    void calcDhfImpl(Size lowResSize, int scale, int blurKernelSize, const std::vector<float>& blurWeights, const Motion& motion, Mat& DHF)
    {
        CV_DbgAssert(scale > 1);
        CV_DbgAssert(blurKernelSize > 0);
        CV_DbgAssert(blurWeights.size() == blurKernelSize * blurKernelSize);

        DHF.create(lowResSize.area(), blurKernelSize * blurKernelSize, CV_32SC3);
        DHF.setTo(Scalar::all(-1));

        for (int y = 0, lowResInd = 0; y < lowResSize.height; ++y)
        {
            for (int x = 0; x < lowResSize.width; ++x, ++lowResInd)
            {
                Point2f lowOrigCoord = motion.calcCoord(Point(x, y));

                BTV_Base::DHF_Val* DHFPtr = DHF.ptr<BTV_Base::DHF_Val>(lowResInd);

                for (int i = 0, ind = 0; i < blurKernelSize; ++i)
                {
                    for (int j = 0; j < blurKernelSize; ++j, ++ind)
                    {
                        DHFPtr[ind].coord.x = cvFloor(lowOrigCoord.x * scale + j - blurKernelSize / 2);
                        DHFPtr[ind].coord.y = cvFloor(lowOrigCoord.y * scale + i - blurKernelSize / 2);
                        DHFPtr[ind].weight = blurWeights[ind];
                    }
                }
            }
        }
    }
}

void cv::superres::BTV_Base::calcDhf(Size lowResSize, int scale, int blurKernelSize, const vector<float>& blurWeights, MotionModel motionModel, const Mat& m1, const Mat& m2, Mat& DHF)
{
    CV_DbgAssert(motionModel <= MM_UNKNOWN);

    if (motionModel == MM_UNKNOWN)
    {
        CV_DbgAssert(m1.type() == CV_32FC1);
        CV_DbgAssert(m1.size() == lowResSize);
        CV_DbgAssert(m1.size() == m2.size());
        CV_DbgAssert(m1.type() == m2.type());

        GeneralMotion motion(m1, m2);
        calcDhfImpl(lowResSize, scale, blurKernelSize, blurWeights, motion, DHF);
    }
    else if (motionModel < MM_HOMOGRAPHY)
    {
        CV_DbgAssert(m1.type() == CV_32FC1);
        CV_DbgAssert(m1.rows == 2 || m1.rows == 3);
        CV_DbgAssert(m1.cols == 3);

        AffineMotion motion(m1);
        calcDhfImpl(lowResSize, scale, blurKernelSize, blurWeights, motion, DHF);
    }
    else
    {
        CV_DbgAssert(m1.type() == CV_32FC1);
        CV_DbgAssert(m1.rows == 3);
        CV_DbgAssert(m1.cols == 3);

        PerspectiveMotion motion(m1);
        calcDhfImpl(lowResSize, scale, blurKernelSize, blurWeights, motion, DHF);
    }
}

/////////////////////////////////////////////////////////////
// BTV

namespace cv
{
    namespace superres
    {
        typedef void (Algorithm::*IntSetter)(int);

        CV_INIT_ALGORITHM(BTV, "SuperResolution.BTV",
                          obj.info()->addParam(obj, "scale", obj.scale, false, 0, 0,
                                               "Scale factor.");
                          obj.info()->addParam(obj, "iterations", obj.iterations, false, 0, 0,
                                               "Iteration count.");
                          obj.info()->addParam(obj, "beta", obj.beta, false, 0, 0,
                                               "Asymptotic value of steepest descent method.");
                          obj.info()->addParam(obj, "lambda", obj.lambda, false, 0, 0,
                                               "Weight parameter to balance data term and smoothness term.");
                          obj.info()->addParam(obj, "alpha", obj.alpha, false, 0, 0,
                                               "Parameter of spacial distribution in btv.");
                          obj.info()->addParam(obj, "btvKernelSize", obj.btvKernelSize, false, 0, 0,
                                               "Kernel size of btv filter.");
                          obj.info()->addParam(obj, "motionModel", obj.motionModel, false, 0, (IntSetter) &BTV::setMotionModel,
                                              "Motion model between frames.");
                          obj.info()->addParam(obj, "blurModel", obj.blurModel, false, 0, 0,
                                               "Blur model.");
                          obj.info()->addParam(obj, "blurKernelSize", obj.blurKernelSize, false, 0, 0,
                                               "Blur kernel size (if -1, than it will be equal scale factor).");
                          obj.info()->addParam(obj, "temporalAreaRadius", obj.temporalAreaRadius, false, 0, 0,
                                               "Radius of the temporal search area."));
    }
}

bool cv::superres::BTV::init()
{
    return !BTV_info_auto.name().empty();
}

Ptr<SuperResolution> cv::superres::BTV::create()
{
    Ptr<SuperResolution> alg(new BTV);
    return alg;
}

cv::superres::BTV::BTV()
{
    setMotionModel(MM_AFFINE);
    blurModel = BLUR_GAUSS;
    blurKernelSize = 5;
    temporalAreaRadius = 4;

    curBlurModel = -1;
}

void cv::superres::BTV::setMotionModel(int motionModel)
{
    CV_DbgAssert(motionModel >= MM_TRANSLATION && motionModel <= MM_UNKNOWN);

    motionEstimator = MotionEstimator::create(static_cast<MotionModel>(motionModel));
    this->motionModel = motionModel;
}

void cv::superres::BTV::initImpl(Ptr<IFrameSource>& frameSource)
{
    const int cacheSize = 2 * temporalAreaRadius + 1;

    frames.resize(cacheSize);
    results.resize(cacheSize);

    storePos = -1;

    for (int t = -temporalAreaRadius; t <= temporalAreaRadius; ++t)
    {
        Mat frame = frameSource->nextFrame();
        CV_Assert(!frame.empty());
        addNewFrame(frame);
    }

    for (int i = 0; i <= temporalAreaRadius; ++i)
        processFrame(i);

    procPos = temporalAreaRadius;
    outPos = -1;
}

Mat cv::superres::BTV::processImpl(Ptr<IFrameSource>& frameSource)
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

void cv::superres::BTV::addNewFrame(const Mat& frame)
{
    if (frame.empty())
        return;

    CV_DbgAssert(frame.type() == CV_8UC1 || frame.type() == CV_8UC3);
    CV_DbgAssert(storePos < 0 || frame.size() == at(storePos, frames).size());

    ++storePos;
    frame.copyTo(at(storePos, frames));
}

void cv::superres::BTV::processFrame(int idx)
{
    CV_DbgAssert(scale > 1);
    CV_DbgAssert(blurModel == BLUR_BOX || blurModel == BLUR_GAUSS);

    if (blurKernelSize < 0)
        blurKernelSize = scale;

    if (blurWeights.size() != blurKernelSize * blurKernelSize || curBlurModel != blurModel)
    {
        calcBlurWeights(static_cast<BlurModel>(blurModel), blurKernelSize, blurWeights);
        curBlurModel = blurModel;
    }

    y.resize(frames.size());
    DHF.resize(frames.size());

    int count = 0;

    Mat src = at(idx, frames);
    src.convertTo(src_f, CV_32F);

    for (size_t k = 0; k < frames.size(); ++k)
    {
        Mat curImage = frames[k];

        bool ok = motionEstimator->estimate(curImage, src, m1, m2);

        if (ok)
        {
            curImage.convertTo(y[count], CV_32F);
            calcDhf(src.size(), scale, blurKernelSize, blurWeights, static_cast<MotionModel>(motionModel), m1, m2, DHF[count]);
            ++count;
        }
    }

    process(src_f, at(idx, results), y, DHF, count);
}

///////////////////////////////////////////////////////////////
// Tests

#ifdef WITH_TESTS

namespace
{
    TEST(MulDhfMat, Identity)
    {
        Mat_<float> src(10, 10);
        theRNG().fill(src, RNG::UNIFORM, 0, 255);

        Mat_<Point3f> DHF(src.size().area(), 1);
        for (int y = 0; y < src.rows; ++y)
            for (int x = 0; x < src.cols; ++x)
                DHF(y * src.rows + x, 0) = Point3f(x, y, 1.0);

        Mat_<float> dst;
        mulDhfMat(DHF, src, dst, src.size());

        const double diff = norm(src, dst, NORM_INF);
        EXPECT_EQ(0, diff);
    }

    TEST(MulDhfMat, PairSum)
    {
        Mat_<float> src(10, 10);
        theRNG().fill(src, RNG::UNIFORM, 0, 255);

        Mat_<Point3f> DHF(src.size().area(), 2);
        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                DHF(y * src.rows + x, 0) = Point3f(x, y, 0.5f);
                DHF(y * src.rows + x, 1) = Point3f(min(x + 1, src.cols - 1), y, 0.5f);
            }
        }

        Mat_<float> dst;
        mulDhfMat(DHF, src, dst, src.size());

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                const float gold = src(y, x) * 0.5f + src(y, min(x + 1, src.cols - 1)) * 0.5f;
                EXPECT_EQ(gold, dst(y, x));
            }
        }
    }

    TEST(DHF, Calc)
    {
        Size lowResSize(5, 5);
        int scale = 2;

        int blurKernelRadius = 2;
        vector<float> blurWeights(blurKernelRadius * blurKernelRadius, 1.0f / (blurKernelRadius * blurKernelRadius));

        Mat_<Point3f> DHF;
        BTV_Base::calcDhf(lowResSize, scale, blurKernelRadius, blurWeights, MM_AFFINE, Mat_<float>::eye(2, 3), Mat(), DHF);

        EXPECT_EQ(lowResSize.area(), DHF.rows);
        EXPECT_EQ(blurKernelRadius * blurKernelRadius, DHF.cols);

        for (int y = 0; y < lowResSize.height; ++y)
        {
            for (int x = 0; x < lowResSize.width; ++x)
            {
                for (int i = 0; i < blurKernelRadius; ++i)
                {
                    for (int j = 0; j < blurKernelRadius; ++j)
                    {
                        Point3f gold(x * scale + j - blurKernelRadius / 2, y * scale + i - blurKernelRadius / 2, 1.0f / (blurKernelRadius * blurKernelRadius));

                        EXPECT_EQ(gold, DHF.at<Point3f>(y * lowResSize.width + x, i * blurKernelRadius + j));
                    }
                }
            }
        }
    }

    TEST(DiffSign, Accuracy)
    {
        Mat_<float> src1(1, 3);
        src1 << 1, 2, 3;

        Mat_<float> src2(1, 3);
        src2 << 3, 2, 1;

        Mat_<float> gold(1, 3);
        gold << -1, 0, 1;

        Mat_<float> dst(1, 3);
        diffSign(src1, src2, dst);

        const double diff = norm(gold, dst, NORM_INF);
        EXPECT_EQ(0, diff);
    }

    TEST(CalcBtvDiffTerm, Accuracy)
    {
        Mat_<float> X(3, 3, 2.0f);

        Mat_<Point3f> DHF(X.size().area(), 1);
        for (int y = 0; y < X.rows; ++y)
            for (int x = 0; x < X.cols; ++x)
                DHF(y * X.rows + x, 0) = Point3f(x, y, 1.0f);

        Mat_<float> y(3, 3);
        y << 1,1,1,2,2,2,3,3,3;

        Mat_<float> gold(3, 3);
        gold << 1,1,1,0,0,0,-1,-1,-1;

        Mat dst, buf;
        calcBtvDiffTerm(y, DHF, X, dst, buf);

        const double diff = norm(gold, dst, NORM_INF);
        EXPECT_EQ(0, diff);
    }
}

#endif // WITH_TESTS
