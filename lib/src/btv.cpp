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
#ifdef WITH_TESTS
    #include <opencv2/ts/ts_gtest.h>
#endif


using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;

namespace cv
{
    namespace superres
    {
        CV_INIT_ALGORITHM(BilateralTotalVariation, "ImageSuperResolution.BilateralTotalVariation",
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
                                               "Kernel size of btv filter."));
    }
}

bool cv::superres::BilateralTotalVariation::init()
{
    return !BilateralTotalVariation_info_auto.name().empty();
}

Ptr<ImageSuperResolution> cv::superres::BilateralTotalVariation::create()
{
    return Ptr<ImageSuperResolution>(new BilateralTotalVariation);
}

cv::superres::BilateralTotalVariation::BilateralTotalVariation()
{
    scale = 4;
    iterations = 180;
    beta = 1.3;
    lambda = 0.03;
    alpha = 0.7;
    btvKernelSize = 7;

    Ptr<MotionEstimatorBase> baseEstimator(new MotionEstimatorRansacL2);
    motionEstimator = new KeypointBasedMotionEstimator(baseEstimator);
}

void cv::superres::BilateralTotalVariation::train(InputArrayOfArrays _images)
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

void cv::superres::BilateralTotalVariation::trainImpl(const vector<Mat>& images)
{
#ifdef _DEBUG
    CV_DbgAssert(!images.empty());
    CV_DbgAssert(images[0].type() == CV_8UC3);

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

bool cv::superres::BilateralTotalVariation::empty() const
{
    return images.empty();
}

void cv::superres::BilateralTotalVariation::clear()
{
    images.clear();
}

namespace
{
    template <typename T, int cn> struct VecTraits;
    template <typename T> struct VecTraits<T, 1>
    {
        typedef T vec_t;
        static T defaultValue()
        {
            return 0;
        }
    };
    template <typename T> struct VecTraits<T, 3>
    {
        typedef Point3_<T> vec_t;
        static Point3_<T> defaultValue()
        {
            return Point3_<T>(0,0,0);
        }
    };

    template <typename T, int cn>
    void mulSparseMatImpl(const SparseMat& smat, const Mat& src, Mat& dst, bool isTranspose)
    {
        typedef typename VecTraits<T, cn>::vec_t vec_t;

        const int srcInd = isTranspose ? 0 : 1;
        const int dstInd = isTranspose ? 1 : 0;

        CV_DbgAssert(src.rows == 1);
        CV_DbgAssert(src.cols == smat.size(srcInd));

        dst.create(1, smat.size(dstInd), src.type());
        dst.setTo(Scalar::all(0));

        const vec_t* srcPtr = src.ptr<vec_t>();
        vec_t* dstPtr = dst.ptr<vec_t>();

        for (SparseMatConstIterator it = smat.begin(), it_end = smat.end(); it != it_end; ++it)
        {
            const int i = it.node()->idx[srcInd];
            const int j = it.node()->idx[dstInd];

            CV_DbgAssert(i >= 0 && i < src.cols);
            CV_DbgAssert(j >= 0 && j < dst.cols);

            const double w = it.value<T>();

            dstPtr[j] += w * srcPtr[i];
        }
    }

    void mulSparseMat(const SparseMat& smat, const Mat& src, Mat& dst, bool isTranspose = false)
    {
        typedef void (*func_t)(const SparseMat& smat, const Mat& src, Mat& dst, bool isTranspose);
        static const func_t funcs[2][2] =
        {
            {mulSparseMatImpl<float, 1>, mulSparseMatImpl<float, 3>},
            {mulSparseMatImpl<double, 1>, mulSparseMatImpl<double, 3>}
        };

        CV_DbgAssert(smat.depth() == CV_32F || smat.depth() == CV_64F);
        CV_DbgAssert(src.depth() == smat.depth());
        CV_DbgAssert(src.channels() == 1 || src.channels() == 3);

        const func_t func = funcs[smat.depth() == CV_64F][src.channels() == 3];

        func(smat, src, dst, isTranspose);
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
        const T* src1Ptr = src1.ptr<T>();
        const T* src2Ptr = src2.ptr<T>();
        T* dstPtr = dst.ptr<T>();

        for (int i = 0; i < src1.cols; ++i)
            dstPtr[i] = diffSign(src1Ptr[i], src2Ptr[i]);
    }

    void diffSign(const Mat& src1, const Mat& src2, Mat& dst)
    {
        typedef void (*func_t)(const Mat& src1, const Mat& src2, Mat& dst);
        static const func_t funcs[] =
        {
            diffSignImpl<uchar>,
            diffSignImpl<schar>,
            diffSignImpl<ushort>,
            diffSignImpl<short>,
            diffSignImpl<int>,
            diffSignImpl<float>,
            diffSignImpl<double>
        };

        CV_DbgAssert(src1.rows == 1);
        CV_DbgAssert(src1.size() == src2.size());
        CV_DbgAssert(src1.type() == src2.type());

        dst.create(src1.size(), src1.type());

        const func_t func = funcs[src1.depth()];

        Mat dst1cn = dst.reshape(1);
        func(src1.reshape(1), src2.reshape(1), dst1cn);
    }

    void calcBtvDiffTerm(const Mat& X, const SparseMat& DHF, const Mat& y, Mat& dst, Mat& buf)
    {
        // degrade current estimated image
        mulSparseMat(DHF, X, buf);

        // compere input and degraded image
        diffSign(buf, y, buf);

        // blur the subtructed vector with transposed matrix
        mulSparseMat(DHF, buf, dst, true);
    }

    struct ProcessBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        vector<Mat_<Point3d> >* y;
        vector<SparseMat_<double> >* DHFs;

        Mat_<Point3d> X;

        vector<Mat_<Point3d> >* diffTerms;
        vector<Mat_<Point3d> >* temps;
    };

    void ProcessBody::operator ()(const Range& range) const
    {
        Mat_<Point3d> temp = (*temps)[range.start];

        for (int i = range.start; i < range.end; ++i)
        {
            // degrade current estimated image
            mulSparseMat((*DHFs)[i], X, temp);

            // compere input and degraded image
            diffSign(temp, (*y)[i], temp);

            // blur the subtructed vector with transposed matrix
            mulSparseMat((*DHFs)[i], temp, (*diffTerms)[i], true);
        }
    }

    template <typename T, int cn>
    struct BtvRegularizationBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        Mat src;
        mutable Mat dst;
        int ksize;
        const T* weight;
    };

    template <typename T, int cn>
    void BtvRegularizationBody<T, cn>::operator ()(const Range& range) const
    {
        typedef typename VecTraits<T, cn>::vec_t vec_t;

        for (int i = range.start; i < range.end; ++i)
        {
            const vec_t* srcRow = src.ptr<vec_t>(i);
            vec_t* dstRow = dst.ptr<vec_t>(i);

            for(int j = ksize; j < src.cols - ksize; ++j)
            {
                vec_t dstVal = VecTraits<T, cn>::defaultValue();

                const vec_t srcVal = srcRow[j];

                for (int m = 0, count = 0; m <= ksize; ++m)
                {
                    const vec_t* srcRow2 = src.ptr<vec_t>(i - m);
                    const vec_t* srcRow3 = src.ptr<vec_t>(i + m);

                    for (int l = ksize; l + m >= 0; --l, ++count)
                        dstVal += weight[count] * (diffSign(srcVal, srcRow3[j + l]) - diffSign(srcRow2[j - l], srcVal));
                }

                dstRow[j] = dstVal;
            }
        }
    }

    template <typename T, int cn>
    void calcBtvRegularizationImpl(Size highResSize, const Mat& X_, Mat& dst_, int btvKernelSize, double alpha)
    {
        CV_DbgAssert(X_.rows == 1 && X_.cols == highResSize.area());

        dst_.create(X_.size(), X_.type());

        Mat src = X_.reshape(X_.channels(), highResSize.height);
        Mat dst = dst_.reshape(dst_.channels(), highResSize.height);

        const int ksize = (btvKernelSize - 1) / 2;

        AutoBuffer<T> weight_(btvKernelSize * btvKernelSize);
        T* weight = weight_;
        for (int m = 0, count = 0; m <= ksize; ++m)
        {
            for (int l = ksize; l + m >= 0; --l, ++count)
                weight[count] = pow(static_cast<T>(alpha), std::abs(m) + std::abs(l));
        }

        BtvRegularizationBody<T, cn> body;

        body.src = src;
        body.dst = dst;
        body.ksize = ksize;
        body.weight = weight;

        parallel_for_(Range(ksize, src.rows - ksize), body);
    }

    void calcBtvRegularization(Size highResSize, const Mat& X_, Mat& dst_, int btvKernelSize, double alpha)
    {
        typedef void (*func_t)(Size highResSize, const Mat& X_, Mat& dst_, int btvKernelSize, double alpha);
        static const func_t funcs[2][2] =
        {
            {calcBtvRegularizationImpl<float, 1>, calcBtvRegularizationImpl<float, 3>},
            {calcBtvRegularizationImpl<double, 1>, calcBtvRegularizationImpl<double, 3>},
        };

        CV_DbgAssert(X_.depth() == CV_32F || X_.depth() == CV_64F);
        CV_DbgAssert(X_.channels() == 1 || X_.channels() == 3);

        const func_t func = funcs[X_.depth() == CV_64F][X_.channels() == 3];

        func(highResSize, X_, dst_, btvKernelSize, alpha);
    }
}

void cv::superres::BilateralTotalVariation::process(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();

    CV_DbgAssert(empty() || src.size() == images[0].size());
    CV_DbgAssert(empty() || src.type() == images[0].type());

    const Size lowResSize = src.size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

    // calc DHF for all low-res images

    vector<Mat_<Point3d> > y;
    vector<SparseMat_<double> > DHFs;

    y.reserve(images.size() + 1);
    DHFs.reserve(images.size() + 1);

    y.push_back(src.reshape(src.channels(), 1));
    DHFs.push_back(calcDHF(lowResSize, highResSize, Mat_<float>::eye(3, 3)));

    for (size_t i = 0; i < images.size(); ++i)
    {
        const Mat& curImage = images[i];

        bool ok;
        Mat_<float> M = motionEstimator->estimate(curImage, src, &ok);

        if (ok)
        {
            M(0, 2) *= scale;
            M(1, 2) *= scale;

            y.push_back(curImage.reshape(curImage.channels(), 1));
            DHFs.push_back(calcDHF(lowResSize, highResSize, M));
        }
    }

    Mat_<Point3d> X(1, highResSize.area());
    vector<Mat_<Point3d> > diffTerms(y.size());
    vector<Mat_<Point3d> > temps(y.size());
    Mat regTerm;

    // create initial image by simple bi-cubic interpolation

    {
        Mat_<Point3d> lowResImage(lowResSize.height, lowResSize.width, y.front()[0]);
        Mat_<Point3d> highResImage(highResSize.height, highResSize.width, X[0]);
        resize(lowResImage, highResImage, highResSize, 0, 0, INTER_CUBIC);
    }

    // steepest descent method for L1 norm minimization

    for (int i = 0; i < iterations; ++i)
    {
        // diff terms

        {
            ProcessBody body;

            body.y = &y;
            body.DHFs = &DHFs;
            body.X = X;
            body.diffTerms = &diffTerms;
            body.temps = &temps;

            parallel_for_(Range(0, y.size()), body);
        }

        // regularization term

        if (lambda > 0)
            calcBtvRegularization(highResSize, X, regTerm, btvKernelSize, alpha);

        // creep ideal image, beta is parameter of the creeping speed.

        for (size_t n = 0; n < y.size(); ++n)
            addWeighted(X, 1.0, diffTerms[n], -beta, 0.0, X);

        // add smoothness term

        if (lambda > 0.0)
            addWeighted(X, 1.0, regTerm, -beta * lambda, 0.0, X);
    }

    // re-convert 1D vecor structure to Mat image structure
    X.reshape(X.channels(), highResSize.height).convertTo(_dst, CV_8UC(X.channels()));
}

SparseMat_<double> cv::superres::BilateralTotalVariation::calcDHF(cv::Size lowResSize, cv::Size highResSize, const cv::Mat_<float>& M)
{
    // D - down sampling matrix.
    // H - blur matrix, in this case, we use only ccd sampling blur.
    // F - motion matrix, in this case, we use only affine motion.

    const int sizes[] = {lowResSize.area(), highResSize.area()};
    SparseMat_<double> DHF(2, sizes);

    const int ksize = scale;
    const int anchor = ksize / 2;
    const double div = 1.0 / (ksize * ksize);

    for (int y = 0; y < lowResSize.height; ++y)
    {
        for (int x = 0; x < lowResSize.width; ++x)
        {
            const int lowResInd = y * lowResSize.width + x;

            for (int i = 0; i < ksize; ++i)
            {
                for (int j = 0; j < ksize; ++j)
                {
                    const int Y = y * scale + i - anchor;
                    const int X = x * scale + j - anchor;

                    const double nX = M(0, 0) * X + M(0, 1) * Y + M(0, 2);
                    const double nY = M(1, 0) * X + M(1, 1) * Y + M(1, 2);

                    const int nX0 = cvFloor(nX);
                    const int nY0 = cvFloor(nY);

                    if (nX0 >= 0 && nX0 < highResSize.width && nY0 >= 0 && nY0 < highResSize.height)
                    {
                        DHF.ref(lowResInd, nY0 * highResSize.width + nX0) += div;
                    }
                }
            }
        }
    }

    return DHF;
}

///////////////////////////////////////////////////////////////
// Tests

#ifdef WITH_TESTS

namespace cv
{
    namespace superres
    {
        TEST(MulSparseMat, Identity)
        {
            Mat_<float> src(1, 10);
            for (int i = 0; i < src.cols; ++i)
                src(0, i) = i;

            const int sizes[] = {src.cols, src.cols};
            SparseMat_<float> smat(2, sizes);
            for (int i = 0; i < src.cols; ++i)
                smat.ref(i, i) = 1;

            Mat_<float> dst;
            mulSparseMat(smat, src, dst);

            EXPECT_EQ(src.size(), dst.size());

            const double diff = norm(src, dst, NORM_INF);
            EXPECT_EQ(0, diff);
        }

        TEST(MulSparseMat, PairSum)
        {
            Mat_<float> src(1, 10);
            for (int i = 0; i < src.cols; ++i)
                src(0, i) = i;

            const int sizes[] = {src.cols - 1, src.cols};
            SparseMat_<float> smat(2, sizes);
            for (int i = 0; i < src.cols - 1; ++i)
            {
                smat.ref(i, i) = 1;
                smat.ref(i, i + 1) = 1;
            }

            Mat_<float> dst;
            mulSparseMat(smat, src, dst);

            EXPECT_EQ(1, dst.rows);
            EXPECT_EQ(src.cols - 1, dst.cols);

            for (int i = 0; i < src.cols - 1; ++i)
            {
                const float gold = src(0, i) + src(0, i + 1);
                EXPECT_EQ(gold, dst(0, i));
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
            Mat_<float> X(1, 9, 2.0f);

            const int sizes[] = {X.cols, X.cols};
            SparseMat_<float> DHF(2, sizes);
            for (int i = 0; i < X.cols; ++i)
                DHF.ref(i, i) = 1;

            Mat_<float> y(1, 9);
            y << 1,1,1,2,2,2,3,3,3;

            Mat_<float> gold(1, 9);
            gold << 1,1,1,0,0,0,-1,-1,-1;

            Mat dst, buf;
            calcBtvDiffTerm(X, DHF, y, dst, buf);

            const double diff = norm(gold, dst, NORM_INF);
            EXPECT_EQ(0, diff);
        }
    }
}

#endif // WITH_TESTS
