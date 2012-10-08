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

///////////////////////////////////////////////////////////////
// BilateralTotalVariation

namespace
{
    void calcBlurWeights(BlurModel blurModel, int blurKernelSize, int depth, Mat& weights)
    {
        CV_DbgAssert(depth == CV_32F || depth == CV_64F);

        switch (blurModel)
        {
        case BLUR_BOX:
            weights.create(1, blurKernelSize * blurKernelSize, depth);
            weights.setTo(Scalar::all(1.0 / (blurKernelSize * blurKernelSize)));
            break;

        case BLUR_GAUSS:
            if (depth == CV_32F)
            {
                Mat_<float> ker = getGaussianKernel(blurKernelSize, 0, CV_32F);

                weights.create(1, blurKernelSize * blurKernelSize, CV_32F);
                float* weightsPtr = weights.ptr<float>();

                for (int i = 0, ind = 0; i < blurKernelSize; ++i)
                    for (int j = 0; j < blurKernelSize; ++j, ++ind)
                        weightsPtr[ind] = ker(i, 0) * ker(j, 0);
            }
            else
            {
                Mat_<double> ker = getGaussianKernel(blurKernelSize, 0, CV_64F);

                weights.create(1, blurKernelSize * blurKernelSize, CV_64F);
                double* weightsPtr = weights.ptr<double>();

                for (int i = 0, ind = 0; i < blurKernelSize; ++i)
                    for (int j = 0; j < blurKernelSize; ++j, ++ind)
                        weightsPtr[ind] = ker(i, 0) * ker(j, 0);
            }
        };
    }

    template <typename T>
    class AffineMotion
    {
    public:
        explicit AffineMotion(const Mat& M) : M(M) {}

        Point2d calcCoord(Point base) const
        {
            Point2d res;
            res.x = M(0, 0) * base.x + M(0, 1) * base.y + M(0, 2);
            res.y = M(1, 0) * base.x + M(1, 1) * base.y + M(1, 2);

            return res;
        }

    private:
        Mat_<T> M;
    };

    template <typename T>
    class PerspectiveMotion
    {
    public:
        explicit PerspectiveMotion(const Mat& M) : M(M) {}

        Point2d calcCoord(Point base) const
        {
            double w = 1.0 / (M(2, 0) * base.x + M(2, 1) * base.y + M(2, 2));

            Point2d res;
            res.x = (M(0, 0) * base.x + M(0, 1) * base.y + M(0, 2)) * w;
            res.y = (M(1, 0) * base.x + M(1, 1) * base.y + M(1, 2)) * w;

            return res;
        }

    private:
        Mat_<T> M;
    };

    template <typename T>
    class GeneralMotion
    {
    public:
        GeneralMotion(const Mat& dx, const Mat& dy) : dx(dx), dy(dy) {}

        Point2d calcCoord(Point base) const
        {
            Point2d res;
            res.x = base.x + dx(base);
            res.y = base.y + dy(base);

            return res;
        }

    private:
        Mat_<T> dx;
        Mat_<T> dy;
    };

    int clamp(int val, int minVal, int maxVal)
    {
        return max(min(val, maxVal), minVal);
    }

    template <class Motion>
    void calcDhfImpl(Size lowResSize, Size highResSize, int scale, int blurKernelSize, const Motion& motion, Mat& DHF)
    {
        CV_DbgAssert(scale > 1);
        CV_DbgAssert(blurKernelSize > 0);

        DHF.create(lowResSize.area(), blurKernelSize * blurKernelSize, CV_32SC2);

        for (int y = 0, lowResInd = 0; y < lowResSize.height; ++y)
        {
            for (int x = 0; x < lowResSize.width; ++x, ++lowResInd)
            {
                Point2d lowOrigCoord = motion.calcCoord(Point(x, y));

                Point* dhfRow = DHF.ptr<Point>(lowResInd);

                for (int i = 0, highResInd = 0; i < blurKernelSize; ++i)
                {
                    for (int j = 0; j < blurKernelSize; ++j, ++highResInd)
                    {
                        int X = cvFloor(lowOrigCoord.x * scale + j - blurKernelSize / 2);
                        int Y = cvFloor(lowOrigCoord.y * scale + i - blurKernelSize / 2);

                        X = clamp(X, 0, highResSize.width - 1);
                        Y = clamp(Y, 0, highResSize.height - 1);

                        dhfRow[highResInd] = Point(X, Y);
                    }
                }
            }
        }
    }

    void calcDhf(Size lowResSize, int scale, int blurKernelSize, const Mat& m1, const Mat& m2, MotionModel motionModel, Mat& DHF)
    {
        CV_DbgAssert(scale > 1);

        Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

        if (motionModel == MM_UNKNOWN)
        {
            CV_DbgAssert(m1.type() == CV_32FC1 || m1.type() == CV_64FC1);
            CV_DbgAssert(m1.size() == lowResSize);
            CV_DbgAssert(m1.size() == m2.size());
            CV_DbgAssert(m1.type() == m2.type());

            if (m1.type() == CV_32FC1)
            {
                GeneralMotion<float> motion(m1, m2);
                calcDhfImpl(lowResSize, highResSize, scale, blurKernelSize, motion, DHF);
            }
            else
            {
                GeneralMotion<double> motion(m1, m2);
                calcDhfImpl(lowResSize, highResSize, scale, blurKernelSize, motion, DHF);
            }
        }
        else if (motionModel < MM_HOMOGRAPHY)
        {
            CV_DbgAssert(m1.rows == 2 || m1.rows == 3);
            CV_DbgAssert(m1.cols == 3);
            CV_DbgAssert(m1.type() == CV_32FC1 || m1.type() == CV_64FC1);

            if (m1.type() == CV_32FC1)
            {
                AffineMotion<float> motion(m1);
                calcDhfImpl(lowResSize, highResSize, scale, blurKernelSize, motion, DHF);
            }
            else
            {
                AffineMotion<double> motion(m1);
                calcDhfImpl(lowResSize, highResSize, scale, blurKernelSize, motion, DHF);
            }
        }
        else
        {
            CV_DbgAssert(m1.rows == 3);
            CV_DbgAssert(m1.cols == 3);
            CV_DbgAssert(m1.type() == CV_32FC1 || m1.type() == CV_64FC1);

            if (m1.type() == CV_32FC1)
            {
                PerspectiveMotion<float> motion(m1);
                calcDhfImpl(lowResSize, highResSize, scale, blurKernelSize, motion, DHF);
            }
            else
            {
                PerspectiveMotion<double> motion(m1);
                calcDhfImpl(lowResSize, highResSize, scale, blurKernelSize, motion, DHF);
            }
        }
    }
}

cv::superres::BilateralTotalVariation::BilateralTotalVariation()
{
    scale = 4;
    iterations = 180;
    beta = 1.3;
    lambda = 0.03;
    alpha = 0.7;
    btvKernelSize = 7;
    workDepth = CV_64F;
    blurModel = BLUR_GAUSS;
    blurKernelSize = 5;
    setMotionModel(MM_AFFINE);

    curBlurModel = -1;
}

void cv::superres::BilateralTotalVariation::setMotionModel(int motionModel)
{
    CV_DbgAssert(motionModel >= MM_TRANSLATION && motionModel <= MM_UNKNOWN);

    motionEstimator = MotionEstimator::create(static_cast<MotionModel>(motionModel));
    this->motionModel = motionModel;
}

namespace
{
    template <typename T, typename VT>
    void mulDhfMatImpl(const Mat& DHF, const Mat& blurWeights, const Mat& src, Mat& dst, Size dstSize, bool isTranspose)
    {
        CV_DbgAssert(DHF.type() == CV_32SC2);
        CV_DbgAssert(DHF.cols == blurWeights.cols);
        CV_DbgAssert(blurWeights.type() == DataType<T>::type);
        CV_DbgAssert(src.type() == DataType<VT>::type);

        const T* weights = blurWeights.ptr<T>();

        dst.create(dstSize, src.type());
        dst.setTo(Scalar::all(0));

        if (isTranspose)
        {
            CV_DbgAssert(DHF.rows == src.size().area());

            for (int y = 0, lowResInd = 0; y < src.rows; ++y)
            {
                const VT* srcPtr = src.ptr<VT>(y);

                for (int x = 0; x < src.cols; ++x, ++lowResInd)
                {
                    const Point* coordPtr = DHF.ptr<Point>(lowResInd);

                    for (int i = 0; i < DHF.cols; ++i)
                    {
                        const Point highResCoord = coordPtr[i];
                        const double w = weights[i];

                        CV_DbgAssert(highResCoord.x >= 0 && highResCoord.x < dst.cols);
                        CV_DbgAssert(highResCoord.y >= 0 && highResCoord.y < dst.rows);

                        dst.at<VT>(highResCoord) += w * srcPtr[x];
                    }
                }
            }
        }
        else
        {
            CV_DbgAssert(DHF.rows == dstSize.area());

            for (int y = 0, lowResInd = 0; y < dstSize.height; ++y)
            {
                VT* dstPtr = dst.ptr<VT>(y);

                for (int x = 0; x < dstSize.width; ++x, ++lowResInd)
                {
                    const Point* coordPtr = DHF.ptr<Point>(lowResInd);

                    for (int i = 0; i < DHF.cols; ++i)
                    {
                        const Point highResCoord = coordPtr[i];
                        const double w = weights[i];

                        CV_DbgAssert(highResCoord.x >= 0 && highResCoord.x < src.cols);
                        CV_DbgAssert(highResCoord.y >= 0 && highResCoord.y < src.rows);

                        dstPtr[x] += w * src.at<VT>(highResCoord);
                    }
                }
            }
        }
    }

    void mulDhfMat(const Mat& DHF, const Mat& blurWeights, const Mat& src, Mat& dst, Size dstSize, bool isTranspose = false)
    {
        typedef void (*func_t)(const Mat& DHF, const Mat& blurWeights, const Mat& src, Mat& dst, Size dstSize, bool isTranspose);
        static const func_t funcs[2][2] =
        {
            {mulDhfMatImpl<float, float>, mulDhfMatImpl<float, Point3f>},
            {mulDhfMatImpl<double, double>, mulDhfMatImpl<double, Point3d>}
        };

        CV_DbgAssert(src.depth() == CV_32F || src.depth() == CV_64F);
        CV_DbgAssert(src.channels() == 1 || src.channels() == 3);

        const func_t func = funcs[src.depth() == CV_64F][src.channels() == 3];

        func(DHF, blurWeights, src, dst, dstSize, isTranspose);
    }

    template <typename T>
    T diffSign(T a, T b)
    {
        return a > b ? 1 : a < b ? -1 : 0;
    }
    template <typename T>
    Point3_<T> diffSign(Point3_<T> a, Point3_<T> b)
    {
        return Point3_<T>(
            a.x > b.x ? 1 : a.x < b.x ? -1 : 0,
            a.y > b.y ? 1 : a.y < b.y ? -1 : 0,
            a.z > b.z ? 1 : a.z < b.z ? -1 : 0
        );
    }

    template <typename T>
    void diffSignImpl(const Mat& src1, const Mat& src2, Mat& dst)
    {
        CV_DbgAssert(src1.type() == DataType<T>::type);
        CV_DbgAssert(src2.size() == src1.size());
        CV_DbgAssert(src2.type() == src1.type());
        CV_DbgAssert(dst.size() == src1.size());
        CV_DbgAssert(dst.type() == src1.type());

        for (int y = 0; y < src1.rows; ++y)
        {
            const T* src1Ptr = src1.ptr<T>(y);
            const T* src2Ptr = src2.ptr<T>(y);
            T* dstPtr = dst.ptr<T>(y);

            for (int x = 0; x < src1.cols; ++x)
                dstPtr[x] = diffSign(src1Ptr[x], src2Ptr[x]);
        }
    }

    void diffSign(const Mat& src1, const Mat& src2, Mat& dst)
    {
        CV_DbgAssert(src1.channels() == src2.channels());
        CV_DbgAssert(src1.depth() == CV_32F || src1.depth() == CV_64F);

        typedef void (*func_t)(const Mat& src1, const Mat& src2, Mat& dst);
        static const func_t funcs[] =
        {
            diffSignImpl<float>,
            diffSignImpl<double>
        };

        dst.create(src1.size(), src1.type());

        const func_t func = funcs[src1.depth() == CV_64F];

        Mat dst1cn = dst.reshape(1);
        func(src1.reshape(1), src2.reshape(1), dst1cn);
    }

    void calcBtvDiffTerm(const Mat& y, const Mat& DHF, const Mat& blurWeights, const Mat& X, Mat& diffTerm, Mat& buf)
    {
        // degrade current estimated image
        mulDhfMat(DHF, blurWeights, X, buf, y.size());

        // compere input and degraded image
        diffSign(buf, y, buf);

        // blur the subtructed vector with transposed matrix
        mulDhfMat(DHF, blurWeights, buf, diffTerm, X.size(), true);
    }

    struct BtvDiffTermBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        const vector<Mat>* y;
        const vector<Mat>* DHF;
        int count;

        Mat blurWeights;

        Mat X;

        vector<Mat>* diffTerms;
        vector<Mat>* bufs;
    };

    void BtvDiffTermBody::operator ()(const Range& range) const
    {
        CV_DbgAssert(y && !y->empty() && y->size() >= count);
        CV_DbgAssert(DHF && DHF->size() >= count);
        CV_DbgAssert(diffTerms && diffTerms->size() == count);
        CV_DbgAssert(bufs && bufs->size() == count);
        CV_DbgAssert(range.start >= 0);
        CV_DbgAssert(range.end <= count);

        Mat& buf = (*bufs)[range.start];

        for (int i = range.start; i < range.end; ++i)
            calcBtvDiffTerm((*y)[i], (*DHF)[i], blurWeights, X, (*diffTerms)[i], buf);
    }

    void calcBtvDiffTerms(const vector<Mat>& y, const vector<Mat>& DHF, const Mat& blurWeights, int count, const Mat& X, vector<Mat>& diffTerms, vector<Mat>& bufs)
    {
        diffTerms.resize(count);
        bufs.resize(count);

        BtvDiffTermBody body;

        body.y = &y;
        body.DHF = &DHF;
        body.count = count;
        body.blurWeights = blurWeights;
        body.X = X;
        body.diffTerms = &diffTerms;
        body.bufs = &bufs;

        parallel_for_(Range(0, count), body);
    }

    template <typename T, typename VT>
    struct BtvRegularizationBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        Mat src;
        mutable Mat dst;
        int ksize;
        Mat_<T> _weight;
    };

    template <typename T, typename VT>
    void BtvRegularizationBody<T, VT>::operator ()(const Range& range) const
    {
        CV_DbgAssert(src.type() == DataType<VT>::type);
        CV_DbgAssert(dst.size() == src.size());
        CV_DbgAssert(dst.type() == src.type());
        CV_DbgAssert(ksize > 0);
        CV_DbgAssert(range.start >= 0);
        CV_DbgAssert(range.end <= src.rows);

        const T* weight = _weight[0];

        for (int i = range.start; i < range.end; ++i)
        {
            const VT* srcRow = src.ptr<VT>(i);
            VT* dstRow = dst.ptr<VT>(i);

            for(int j = ksize; j < src.cols - ksize; ++j)
            {
                const VT srcVal = srcRow[j];

                for (int m = 0, count = 0; m <= ksize; ++m)
                {
                    const VT* srcRow2 = src.ptr<VT>(i - m);
                    const VT* srcRow3 = src.ptr<VT>(i + m);

                    for (int l = ksize; l + m >= 0; --l, ++count)
                    {
                        CV_DbgAssert(j + l >= 0 && j + l < src.cols);
                        CV_DbgAssert(j - l >= 0 && j - l < src.cols);
                        CV_DbgAssert(count < _weight.cols);

                        dstRow[j] += weight[count] * (diffSign(srcVal, srcRow3[j + l]) - diffSign(srcRow2[j - l], srcVal));
                    }
                }
            }
        }
    }

    template <typename T, typename VT>
    void calcBtvRegularizationImpl(const Mat& X, Mat& dst, int btvKernelSize, double alpha)
    {
        CV_DbgAssert(btvKernelSize > 0);
        CV_DbgAssert(alpha > 0);

        dst.create(X.size(), X.type());
        dst.setTo(Scalar::all(0));

        const int ksize = (btvKernelSize - 1) / 2;

        const int weightSize = btvKernelSize * btvKernelSize;
        AutoBuffer<T> weight_(weightSize);
        T* weight = weight_;
        for (int m = 0, count = 0; m <= ksize; ++m)
        {
            for (int l = ksize; l + m >= 0; --l, ++count)
            {
                CV_DbgAssert(count < weightSize);
                weight[count] = pow(static_cast<T>(alpha), std::abs(m) + std::abs(l));
            }
        }

        BtvRegularizationBody<T, VT> body;

        body.src = X;
        body.dst = dst;
        body.ksize = ksize;
        body._weight = Mat_<T>(1, weightSize, weight);

        parallel_for_(Range(ksize, X.rows - ksize), body);
    }

    void calcBtvRegularization(const Mat& X, Mat& dst, int btvKernelSize, double alpha)
    {
        typedef void (*func_t)(const Mat& X, Mat& dst, int btvKernelSize, double alpha);
        static const func_t funcs[2][2] =
        {
            {calcBtvRegularizationImpl<float, float>, calcBtvRegularizationImpl<float, Point3f>},
            {calcBtvRegularizationImpl<double, double>, calcBtvRegularizationImpl<double, Point3d>},
        };

        CV_DbgAssert(X.depth() == CV_32F || X.depth() == CV_64F);
        CV_DbgAssert(X.channels() == 1 || X.channels() == 3);

        const func_t func = funcs[X.depth() == CV_64F][X.channels() == 3];

        func(X, dst, btvKernelSize, alpha);
    }
}

void cv::superres::BilateralTotalVariation::process(const vector<Mat>& y, const vector<Mat>& DHF, int count, OutputArray dst)
{
    CV_DbgAssert(count > 0);
    CV_DbgAssert(y.size() >= count);
    CV_DbgAssert(DHF.size() >= count);
    CV_DbgAssert(blurModel == BLUR_BOX || blurModel == BLUR_GAUSS);
    CV_DbgAssert(blurKernelSize > 0);

    if (blurWeights.cols != blurKernelSize * blurKernelSize || blurWeights.depth() != workDepth || blurModel != curBlurModel)
    {
        calcBlurWeights(static_cast<BlurModel>(blurModel), blurKernelSize, workDepth, blurWeights);
        curBlurModel = blurModel;
    }

    Size lowResSize = y.front().size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

#ifdef _DEBUG
    for (size_t i = 0; i < count; ++i)
    {
        CV_DbgAssert(y[i].size() == lowResSize);
        CV_DbgAssert(DHF[i].rows == lowResSize.area());
        CV_DbgAssert(DHF[i].cols == blurWeights.cols);
    }
#endif

    // create initial image by simple bi-cubic interpolation

    resize(y.front(), X, highResSize, 0, 0, INTER_CUBIC);

    // steepest descent method for L1 norm minimization

    for (int i = 0; i < iterations; ++i)
    {
        // diff terms
        calcBtvDiffTerms(y, DHF, blurWeights, count, X, diffTerms, bufs);

        // regularization term

        if (lambda > 0)
            calcBtvRegularization(X, regTerm, btvKernelSize, alpha);

        // creep ideal image, beta is parameter of the creeping speed.

        for (size_t n = 0; n < count; ++n)
            addWeighted(X, 1.0, diffTerms[n], -beta, 0.0, X);

        // add smoothness term

        if (lambda > 0.0)
            addWeighted(X, 1.0, regTerm, -beta * lambda, 0.0, X);
    }

    X.convertTo(dst, CV_8U);
}

///////////////////////////////////////////////////////////////
// BTV_Image

namespace cv
{
    namespace superres
    {
        typedef void (Algorithm::*IntSetter)(int);

        CV_INIT_ALGORITHM(BTV_Image, "ImageSuperResolution.BilateralTotalVariation",
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
                          obj.info()->addParam(obj, "workDepth", obj.workDepth, false, 0, 0,
                                               "Depth for inner operations (CV_32F or CV_64F).");
                          obj.info()->addParam(obj, "motionModel", obj.motionModel, false, 0, (IntSetter) &BTV_Image::setMotionModel,
                                               "Motion model between frames.");
                          obj.info()->addParam(obj, "blurModel", obj.blurModel, false, 0, 0,
                                               "Blur model.");
                          obj.info()->addParam(obj, "blurKernelSize", obj.blurKernelSize, false, 0, 0,
                                               "Blur kernel size (if -1, than it will be equal scale factor)."));
    }
}

bool cv::superres::BTV_Image::init()
{
    return !BTV_Image_info_auto.name().empty();
}

Ptr<ImageSuperResolution> cv::superres::BTV_Image::create()
{
    return Ptr<ImageSuperResolution>(new BTV_Image);
}

void cv::superres::BTV_Image::train(InputArrayOfArrays _images)
{
    vector<Mat> images;

    if (_images.kind() == _InputArray::STD_VECTOR_MAT)
        _images.getMatVector(images);
    else
    {
        Mat image = _images.getMat();
        images.push_back(image);
    }

    trainImpl(images);
}

void cv::superres::BTV_Image::trainImpl(const vector<Mat>& images)
{
#ifdef _DEBUG
    CV_DbgAssert(!images.empty());
    CV_DbgAssert(images[0].type() == CV_8UC1 || images[0].type() == CV_8UC3);

    for (size_t i = 1; i < images.size(); ++i)
    {
        CV_DbgAssert(images[i].size() == images[0].size());
        CV_DbgAssert(images[i].type() == images[0].type());
    }

    if (!this->images.empty())
    {
        CV_DbgAssert(images[0].size() == this->images[0].size());
    }
#endif

    this->images.insert(this->images.end(), images.begin(), images.end());
}

bool cv::superres::BTV_Image::empty() const
{
    return images.empty();
}

void cv::superres::BTV_Image::clear()
{
    images.clear();
}

void cv::superres::BTV_Image::process(InputArray _src, OutputArray dst)
{
    Mat src = _src.getMat();

    CV_DbgAssert(workDepth == CV_32F || workDepth == CV_64F);
    CV_DbgAssert(empty() || src.size() == images[0].size());
    CV_DbgAssert(empty() || src.type() == images[0].type());

    if (blurKernelSize < 0)
        blurKernelSize = scale;

    // calc DHF for all low-res images

    y.resize(images.size() + 1);
    DHF.resize(images.size() + 1);

    int count = 1;
    src.convertTo(y[0], workDepth);
    calcDhf(src.size(), scale, blurKernelSize, Mat_<float>::eye(2, 3), Mat(), MM_AFFINE, DHF[0]);

    for (size_t i = 0; i < images.size(); ++i)
    {
        const Mat& curImage = images[i];

        bool ok = motionEstimator->estimate(curImage, src, m1, m2);

        if (ok)
        {
            curImage.convertTo(y[count], workDepth);
            calcDhf(src.size(), scale, blurKernelSize, m1, m2, static_cast<MotionModel>(motionModel), DHF[count]);
            ++count;
        }
    }

    BilateralTotalVariation::process(y, DHF, count, dst);
}

///////////////////////////////////////////////////////////////
// BTV_Video

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(BTV_Video, "VideoSuperResolution.BilateralTotalVariation",
                          obj.info()->addParam(obj, "temporalAreaRadius", obj.temporalAreaRadius, false, 0, 0,
                                               "Radius of the temporal search area.");
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
                          obj.info()->addParam(obj, "workDepth", obj.workDepth, false, 0, 0,
                                               "Depth for inner operations (CV_32F or CV_64F).");
                          obj.info()->addParam(obj, "motionModel", obj.motionModel, false, 0, (IntSetter) &BTV_Video::setMotionModel,
                                              "Motion model between frames.");
                          obj.info()->addParam(obj, "blurModel", obj.blurModel, false, 0, 0,
                                               "Blur model.");
                          obj.info()->addParam(obj, "blurKernelSize", obj.blurKernelSize, false, 0, 0,
                                               "Blur kernel size (if -1, than it will be equal scale factor)."));
    }
}

bool cv::superres::BTV_Video::init()
{
    return !BTV_Video_info_auto.name().empty();
}

Ptr<VideoSuperResolution> cv::superres::BTV_Video::create()
{
    return Ptr<VideoSuperResolution>(new BTV_Video);
}

cv::superres::BTV_Video::BTV_Video()
{
    temporalAreaRadius = 4;
}

void cv::superres::BTV_Video::initImpl(Ptr<IFrameSource>& frameSource)
{
    const int cacheSize = 2 * temporalAreaRadius + 1;

    frames.resize(cacheSize);
    results.resize(cacheSize);
    y.reserve(cacheSize);
    DHF.reserve(cacheSize);

    storePos = -1;
    procPos = storePos - temporalAreaRadius;
    outPos = procPos - temporalAreaRadius - 1;

    for (int t = -temporalAreaRadius; t <= temporalAreaRadius; ++t)
    {
        Mat frame = frameSource->nextFrame();

        CV_Assert(!frame.empty());

        addNewFrame(frame);
    }

    for (int i = 0; i <= procPos; ++i)
        processFrame(i);
}

Mat cv::superres::BTV_Video::processImpl(const Mat& frame)
{
    addNewFrame(frame);
    processFrame(procPos);
    return at(outPos, results);
}

void cv::superres::BTV_Video::processFrame(int idx)
{
    if (blurKernelSize < 0)
        blurKernelSize = scale;

    y.resize(frames.size());
    DHF.resize(frames.size());

    int count = 0;

    Mat src = at(idx, frames);

    for (size_t k = 0; k < frames.size(); ++k)
    {
        Mat curImage = frames[k];

        bool ok = motionEstimator->estimate(curImage, src, m1, m2);

        if (ok)
        {
            curImage.convertTo(y[count], workDepth);
            calcDhf(src.size(), scale, blurKernelSize, m1, m2, static_cast<MotionModel>(motionModel), DHF[count]);
            ++count;
        }
    }

    BilateralTotalVariation::process(y, DHF, count, at(idx, results));
}

void cv::superres::BTV_Video::addNewFrame(const Mat& frame)
{
    CV_DbgAssert(frame.type() == CV_8UC1 || frame.type() == CV_8UC3);
    CV_DbgAssert(storePos < 0 || frame.size() == at(storePos, frames).size());

    ++storePos;
    ++procPos;
    ++outPos;

    frame.copyTo(at(storePos, frames));
}

///////////////////////////////////////////////////////////////
// Tests

#ifdef WITH_TESTS

namespace cv
{
    namespace superres
    {
        TEST(MulDhfMat, Identity)
        {
            Mat_<float> src(10, 10);
            theRNG().fill(src, RNG::UNIFORM, 0, 255);

            Mat_<Point> DHF(src.size().area(), 1);
            for (int y = 0; y < src.rows; ++y)
                for (int x = 0; x < src.cols; ++x)
                    DHF(y * src.rows + x, 0) = Point(x, y);

            Mat_<float> blurWeights(1, 1);
            blurWeights.setTo(Scalar::all(1));

            Mat_<float> dst;
            mulDhfMat(DHF, blurWeights, src, dst, src.size());

            const double diff = norm(src, dst, NORM_INF);
            EXPECT_EQ(0, diff);
        }

        TEST(MulDhfMat, PairSum)
        {
            Mat_<float> src(10, 10);
            theRNG().fill(src, RNG::UNIFORM, 0, 255);

            Mat_<Point> DHF(src.size().area(), 2);
            for (int y = 0; y < src.rows; ++y)
            {
                for (int x = 0; x < src.cols; ++x)
                {
                    DHF(y * src.rows + x, 0) = Point(x, y);
                    DHF(y * src.rows + x, 1) = Point(min(x + 1, src.cols - 1), y);
                }
            }

            Mat_<float> blurWeights(1, 2);
            blurWeights.setTo(Scalar::all(1));

            Mat_<float> dst;
            mulDhfMat(DHF, blurWeights, src, dst, src.size());

            for (int y = 0; y < src.rows; ++y)
            {
                for (int x = 0; x < src.cols; ++x)
                {
                    const float gold = src(y, x) + src(y, min(x + 1, src.cols - 1));
                    EXPECT_EQ(gold, dst(y, x));
                }
            }
        }

        TEST(DHF, Calc)
        {
            Size lowResSize(5, 5);
            int scale = 2;
            int blurKernelRadius = 2;

            Mat DHF;
            calcDhf(lowResSize, scale, blurKernelRadius, Mat_<float>::eye(2, 3), Mat(), MM_AFFINE, DHF);

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
                            Point gold(x * scale + j - blurKernelRadius / 2, y * scale + i - blurKernelRadius / 2);
                            gold.x = clamp(gold.x, 0, lowResSize.width * scale - 1);
                            gold.y = clamp(gold.y, 0, lowResSize.height * scale - 1);

                            EXPECT_EQ(gold, DHF.at<Point>(y * lowResSize.width + x, i * blurKernelRadius + j));
                        }
                    }
                }
            }
        }

        TEST(DiffSign, Accuracy)
        {
            Mat_<int> src1(1, 3);
            src1 << 1, 2, 3;

            Mat_<int> src2(1, 3);
            src2 << 3, 2, 1;

            Mat_<int> gold(1, 3);
            gold << -1, 0, 1;

            Mat_<int> dst(1, 3);
            diffSign(src1, src2, dst);

            const double diff = norm(gold, dst, NORM_INF);
            EXPECT_EQ(0, diff);
        }

        TEST(CalcBtvDiffTerm, Accuracy)
        {
            Mat_<float> X(3, 3, 2.0f);

            Mat_<Point> DHF(X.size().area(), 1);
            for (int y = 0; y < X.rows; ++y)
                for (int x = 0; x < X.cols; ++x)
                    DHF(y * X.rows + x, 0) = Point(x, y);

            Mat_<float> blurWeights(1, 1);
            blurWeights.setTo(Scalar::all(1));

            Mat_<float> y(3, 3);
            y << 1,1,1,2,2,2,3,3,3;

            Mat_<float> gold(3, 3);
            gold << 1,1,1,0,0,0,-1,-1,-1;

            Mat dst, buf;
            calcBtvDiffTerm(y, DHF, blurWeights, X, dst, buf);

            const double diff = norm(gold, dst, NORM_INF);
            EXPECT_EQ(0, diff);
        }
    }
}

#endif // WITH_TESTS
