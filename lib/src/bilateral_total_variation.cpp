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
                                       "Kernel size of btv filter.");
                  obj.info()->addParam(obj, "normType", obj.normType, false, 0, 0,
                                       "NORM_L1 or NORM_L2."));

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
    normType = NORM_L1;

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

    degImages.assign(images.begin(), images.end());
}

bool BilateralTotalVariation::empty() const
{
    return degImages.empty();
}

void BilateralTotalVariation::clear()
{
    degImages.clear();
}

namespace
{
    void mulSparseMat32f(const SparseMat_<double>& smat, const Mat_<Point3d>& src, Mat_<Point3d>& dst, bool isTranspose = false)
    {
        CV_DbgAssert(src.cols == 1 && dst.cols == 1);

        dst.setTo(Scalar::all(0));

        SparseMatConstIterator_<double> it = smat.begin(), it_end = smat.end();

        const int srcIndIdx = isTranspose ? 1 : 0;
        const int dstIndIdx = isTranspose ? 0 : 1;

        for (; it != it_end; ++it)
        {
            const int i = it.node()->idx[srcIndIdx];
            const int j = it.node()->idx[dstIndIdx];

            CV_DbgAssert(i >= 0 && i < src.rows);
            CV_DbgAssert(j >= 0 && j < dst.rows);

            dst(j, 0) += *it * src(i, 0);
        }
    }

    Point3d sign(Point3d a, Point3d b)
    {
        return Point3d(
            a.x > b.x ? 1.0 : (a.x < b.x ? -1.0 : 0.0),
            a.y > b.y ? 1.0 : (a.y < b.y ? -1.0 : 0.0),
            a.z > b.z ? 1.0 : (a.z < b.z ? -1.0 : 0.0)
        );
    }

    void subtract_sign(const Mat_<Point3d>& src1, const Mat_<Point3d>& src2, Mat_<Point3d>& dst)
    {
        CV_DbgAssert(src1.size() == src2.size());
        CV_DbgAssert(dst.size() == src1.size());

        for (int y = 0; y < src1.rows; ++y)
        {
            const Point3d* src1Row = src1[y];
            const Point3d* src2Row = src2[y];
            Point3d* dstRow = dst[y];

            for (int x = 0; x < src1.cols; ++x)
                dstRow[x] = sign(src1Row[x], src2Row[x]);
        }
    }

    struct ProcessBody : ParallelLoopBody
    {
        void operator ()(const Range& range) const;

        int normType;
        Size lowResSize;

        vector<SparseMat_<double> >* DHFs;

        Mat_<Point3d> dstVec;

        vector<Mat_<Point3d> >* dstVecTemp;
        vector<Mat_<Point3d> >* svec;
        vector<Mat_<Point3d> >* svec2;
    };

    void ProcessBody::operator ()(const Range& range) const
    {
        Mat_<Point3d> temp(lowResSize.area(), 1);

        for (int i = range.start; i < range.end; ++i)
        {
            // degrade current estimated image
            mulSparseMat32f((*DHFs)[i], dstVec, (*svec2)[i]);

            // compere input and degraded image
            temp.setTo(Scalar::all(1));

            if (normType == NORM_L1)
                subtract_sign((*svec2)[i], (*svec)[i], temp);
            else
                subtract((*svec2)[i], (*svec)[i], temp);

            // blur the subtructed vector with transposed matrix
            mulSparseMat32f((*DHFs)[i], temp, (*dstVecTemp)[i], true);
        }
    }
}

void BilateralTotalVariation::process(const Mat& src, Mat& dst)
{
    CV_DbgAssert(!empty());
    CV_DbgAssert(src.size() == degImages[0].size());
    CV_DbgAssert(src.type() == degImages[0].type());
    CV_DbgAssert(normType == NORM_L1 || normType == NORM_L2);

    const Size lowResSize = src.size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

    // calc DHF for all low-res images

    DHFs.resize(degImages.size() + 1);
    for (size_t i = 0; i < degImages.size(); ++i)
    {
        Mat_<float> M = motionEstimator->estimate(src, degImages[i]);
        calcDHF(lowResSize, M, DHFs[i]);
    }

    degImages.push_back(src);
    calcDHF(lowResSize, Mat_<float>::eye(3, 3), DHFs.back());

    // create initial image by simple linear interpolation

    resize(src, dst, highResSize);

    // convert Mat image structure to 1D vecor structure

    dst.reshape(3, highResSize.area()).convertTo(dstVec, dstVec.depth());

    dstVecTemp.resize(degImages.size());
    svec.resize(degImages.size());
    svec2.resize(degImages.size());

    for (size_t i = 0; i < degImages.size(); ++i)
    {
        degImages[i].reshape(3, lowResSize.area()).convertTo(svec[i], svec[i].depth());
        degImages[i].reshape(3, lowResSize.area()).convertTo(svec2[i], svec2[i].depth());
        dstVec.copyTo(dstVecTemp[i]);
    }

    regVec.create(highResSize.area(), 1);
    regVec.setTo(Scalar::all(0));

    // steepest descent method for L1 norm minimization
    for (int i = 0; i < iterations; ++i)
    {
        // btv
        if (lambda > 0)
            btvRegularization(highResSize);

        {
            ProcessBody body;

            body.normType = normType;
            body.lowResSize = lowResSize;
            body.DHFs = &DHFs;
            body.dstVec = dstVec;
            body.dstVecTemp = &dstVecTemp;
            body.svec = &svec;
            body.svec2 = &svec2;

            parallel_for_(Range(0, degImages.size()), body);
        }

        // creep ideal image, beta is parameter of the creeping speed.
        // add transeposed difference vector.
        for (size_t n = 0; n < degImages.size(); ++n)
            addWeighted(dstVec, 1.0, dstVecTemp[n], -beta, 0.0, dstVec);

        // add smoothness term
        if (lambda > 0.0)
            addWeighted(dstVec, 1.0, regVec, -beta  *lambda, 0.0, dstVec);
    }

    // re-convert 1D vecor structure to Mat image structure
    dstVec.reshape(3, highResSize.height).convertTo(dst, CV_8UC3);
}

void BilateralTotalVariation::calcDHF(Size lowResSize, const Mat_<float>& M, SparseMat_<double>& DHF)
{
    // D down sampling matrix.
    // H blur matrix, in this case, we use only ccd sampling blur.
    // F motion matrix, in this case, threr is only global shift motion.

    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

    const int sizes[] = {highResSize.area(), lowResSize.area()};
    DHF.create(2, sizes);

    const Point2d move(M(0, 2) * scale, M(1, 2) * scale);

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
            Point3d* dstRow = dst[i];

            const Point3d* srcRow = src[i];

            for(int j = kw; j < src.cols - kw; ++j)
            {
                Point3d dstVal(0,0,0);

                const Point3d srcVal = srcRow[j];

                for (int m = 0, count = 0; m <= kh; ++m)
                {
                    const Point3d* srcRow2 = src[i - m];
                    const Point3d* srcRow3 = src[i + m];

                    for (int l = kw; l + m >= 0; --l, ++count)
                        dstVal += weight[count] * (sign(srcVal, srcRow3[j + l]) - sign(srcRow2[j - l], srcVal));
                }

                dstRow[j] = dstVal;
            }
        }
    }
}

// btvregularization(dstVec,reg_window,alpha,reg_vec,dest.size());
// btvregularization(Mat& srcVec, Size kernel, float alpha, Mat& dstVec, Size size)

void BilateralTotalVariation::btvRegularization(Size highResSize)
{
    const Mat_<Point3d> src = dstVec.reshape(3, highResSize.height);
    Mat_<Point3d> dst = regVec.reshape(3, highResSize.height);

    const int kw = (btvKernelSize - 1) / 2;
    const int kh = (btvKernelSize - 1) / 2;

    AutoBuffer<float> weight_(btvKernelSize * btvKernelSize);
    float* weight = weight_;
    for (int m = 0, count = 0; m <= kh; ++m)
    {
        for (int l = kw; l + m >= 0; --l, ++count)
            weight[count] = pow(alpha, abs(m) + abs(l));
    }

    // a part of under term of Eq (22) lambda*\sum\sum ...
    {
        BtvRegularizationBody body;

        body.src = src;
        body.dst = dst;
        body.kw = kw;
        body.kh = kh;
        body.weight = weight;

        parallel_for_(Range(kh, src.rows - kh), body);
    }
}
