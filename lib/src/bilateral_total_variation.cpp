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

#include "bilateral_total_variation.hpp"
#include <algorithm>
#include <opencv2/core/internal.hpp>

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;

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

bool BilateralTotalVariation::init()
{
    return !BilateralTotalVariation_info_auto.name().empty();
}

Ptr<ImageSuperResolution> BilateralTotalVariation::create()
{
    return Ptr<ImageSuperResolution>(new BilateralTotalVariation);
}

BilateralTotalVariation::BilateralTotalVariation()
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

void BilateralTotalVariation::train(const vector<Mat>& images)
{
#ifdef _DEBUG
    CV_DbgAssert(!images.empty());
    CV_DbgAssert(images[0].type() == CV_8UC3);
    for (size_t i = 1; i < images.size(); ++i)
    {
        CV_DbgAssert(images[i].size() == images[0].size());
        CV_DbgAssert(images[i].type() == images[0].type());
    }
#endif

    this->images.insert(this->images.end(), images.begin(), images.end());
}

bool BilateralTotalVariation::empty() const
{
    return images.empty();
}

void BilateralTotalVariation::clear()
{
    images.clear();
}

namespace
{
    void mulSparseMat(const SparseMat_<double>& smat, const Mat_<Point3d>& src, Mat_<Point3d>& dst, bool isTranspose = false)
    {
        const int srcInd = isTranspose ? 1 : 0;
        const int dstInd = isTranspose ? 0 : 1;

        CV_DbgAssert(src.rows == 1);
        CV_DbgAssert(src.cols == smat.size(srcInd));

        dst.create(1, smat.size(dstInd));
        dst.setTo(Scalar::all(0));

        const Point3d* srcPtr = src[0];
        Point3d* dstPtr = dst[0];

        for (SparseMatConstIterator_<double> it = smat.begin(), it_end = smat.end(); it != it_end; ++it)
        {
            const int i = it.node()->idx[srcInd];
            const int j = it.node()->idx[dstInd];

            CV_DbgAssert(i >= 0 && i < src.cols);
            CV_DbgAssert(j >= 0 && j < dst.cols);

            const double w = *it;

            dstPtr[j] += w * srcPtr[i];
        }
    }

    Point3d diffSign(Point3d a, Point3d b)
    {
        return Point3d(
            a.x > b.x ? 1.0 : (a.x < b.x ? -1.0 : 0.0),
            a.y > b.y ? 1.0 : (a.y < b.y ? -1.0 : 0.0),
            a.z > b.z ? 1.0 : (a.z < b.z ? -1.0 : 0.0)
        );
    }

    void diffSign(const Mat_<Point3d>& src1, const Mat_<Point3d>& src2, Mat_<Point3d>& dst)
    {
        CV_DbgAssert(src1.rows == 1);
        CV_DbgAssert(src1.size() == src2.size());

        dst.create(src1.size());

        const Point3d* src1Ptr = src1[0];
        const Point3d* src2Ptr = src2[0];
        Point3d* dstPtr = dst[0];

        for (int i = 0; i < src1.cols; ++i)
            dstPtr[i] = diffSign(src1Ptr[i], src2Ptr[i]);
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
}

void BilateralTotalVariation::process(const Mat& src, Mat& dst)
{
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
        Mat_<float> M = motionEstimator->estimate(src, curImage, &ok);

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
    Mat_<Point3d> regTerm;

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
            calcBtvRegularization(highResSize, X, regTerm);

        // creep ideal image, beta is parameter of the creeping speed.

        for (size_t n = 0; n < y.size(); ++n)
            addWeighted(X, 1.0, diffTerms[n], -beta, 0.0, X);

        // add smoothness term

        if (lambda > 0.0)
            addWeighted(X, 1.0, regTerm, -beta * lambda, 0.0, X);
    }

    // re-convert 1D vecor structure to Mat image structure
    X.reshape(X.channels(), highResSize.height).convertTo(dst, CV_8UC(X.channels()));
}

SparseMat_<double> BilateralTotalVariation::calcDHF(cv::Size lowResSize, cv::Size highResSize, const cv::Mat_<float>& M)
{
    // D down sampling matrix.
    // H blur matrix, in this case, we use only ccd sampling blur.
    // F motion matrix, in this case, threr is only global shift motion.

    const int sizes[] = {highResSize.area(), lowResSize.area()};
    SparseMat_<double> DHF(2, sizes);

    const Point2d move(M(0, 2), M(1, 2));

    const double div = 1.0 / (scale * scale);

    const int x1 = cvFloor(move.x + 1);
    const int x0 = cvFloor(move.x);
    const double a1 = move.x - x0;
    const double a0 = 1.0 - a1;

    const int y1 = cvFloor(move.y + 1);
    const int y0 = cvFloor(move.y);
    const double b1 = move.y - y0;
    const double b0 = 1.0 - b1;

    const int bsx = x1;
    const int bsy = y1;

    for (int I = 0, i = 0; I < highResSize.height; I += scale, ++i)
    {
        for (int J = 0, j = 0; J < highResSize.width; J += scale, ++j)
        {
            const int y = highResSize.width * I + J;
            const int s = lowResSize.width * i + j;

            if (J >= bsx && J < highResSize.width - bsx - scale && I >= bsy && I < highResSize.height - bsy - scale)
            {
                for (int l = 0; l < scale; ++l)
                {
                    for (int k = 0; k < scale; ++k)
                    {
                        DHF.ref(y + highResSize.width * (y0 + l) + x0 + k, s) += a0 * b0 * div;
                        DHF.ref(y + highResSize.width * (y0 + l) + x1 + k, s) += a1 * b0 * div;
                        DHF.ref(y + highResSize.width * (y1 + l) + x0 + k, s) += a0 * b1 * div;
                        DHF.ref(y + highResSize.width * (y1 + l) + x1 + k, s) += a1 * b1 * div;
                    }
                }
            }
        }
    }

    return DHF;
}

namespace
{
    struct BtvRegularizationBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        Mat_<Point3d> src;
        mutable Mat_<Point3d> dst;
        int kw;
        int kh;
        float* weight;
    };

    void BtvRegularizationBody::operator ()(const Range& range) const
    {
        for (int i = range.start; i < range.end; ++i)
        {
            const Point3d* srcRow = src[i];
            Point3d* dstRow = dst[i];

            for(int j = kw; j < src.cols - kw; ++j)
            {
                Point3d dstVal(0, 0, 0);

                const Point3d srcVal = srcRow[j];

                for (int m = 0, count = 0; m <= kh; ++m)
                {
                    const Point3d* srcRow2 = src[i - m];
                    const Point3d* srcRow3 = src[i + m];

                    for (int l = kw; l + m >= 0; --l, ++count)
                        dstVal += weight[count] * (diffSign(srcVal, srcRow3[j + l]) - diffSign(srcRow2[j - l], srcVal));
                }

                dstRow[j] = dstVal;
            }
        }
    }
}

void BilateralTotalVariation::calcBtvRegularization(Size highResSize, const Mat_<cv::Point3d>& X_, Mat_<cv::Point3d>& dst_)
{
    CV_DbgAssert(X_.rows == 1 && X_.cols == highResSize.area());

    dst_.create(X_.size());

    const Mat_<Point3d> src = X_.reshape(X_.channels(), highResSize.height);
    Mat_<Point3d> dst = dst_.reshape(dst_.channels(), highResSize.height);

    const int kw = (btvKernelSize - 1) / 2;
    const int kh = (btvKernelSize - 1) / 2;

    AutoBuffer<float> weight_(btvKernelSize * btvKernelSize);
    float* weight = weight_;
    for (int m = 0, count = 0; m <= kh; ++m)
    {
        for (int l = kw; l + m >= 0; --l, ++count)
            weight[count] = pow(alpha, abs(m) + abs(l));
    }

    BtvRegularizationBody body;

    body.src = src;
    body.dst = dst;
    body.kw = kw;
    body.kh = kh;
    body.weight = weight;

    parallel_for_(Range(kh, src.rows - kh), body);
}
