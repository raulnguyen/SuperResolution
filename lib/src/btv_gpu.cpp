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

#include "btv_gpu.hpp"
#include <opencv2/core/internal.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#ifdef WITH_TESTS
    #include <cuda_runtime.h>
    #include <opencv2/ts/ts_gtest.h>
    #include <opencv2/gpu/device/common.hpp>
#endif

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;
using namespace cv::gpu;

#ifdef _DEBUG
    #define cusparseCall(op) CV_DbgAssert((op) == CUSPARSE_STATUS_SUCCESS)
#else
    #define cusparseCall(op) (op)
#endif

namespace btv_device
{
    template <typename T> void diffSign(const PtrStepSzb& src1, const PtrStepSzb& src2, const PtrStepSzb& dst, cudaStream_t stream);
}

///////////////////////////////////////////////////////////////
// GpuSparseMat_CSR

void cv::superres::GpuSparseMat_CSR::create(int rows, int cols, int nonZeroCount, int type)
{
    size_t bytesPerRow = std::max(nonZeroCount * std::max(sizeof(int), (size_t) CV_ELEM_SIZE(type)), (rows + 1) * sizeof(int));

    data_.create(3, bytesPerRow, CV_8U);

    rows_ = rows;
    cols_ = cols;
    nonZeroCount_ = nonZeroCount;
    type_ = type;
}

void cv::superres::GpuSparseMat_CSR::release()
{
    rows_ = 0;
    cols_ = 0;
    nonZeroCount_ = 0;
    type_ = 0;
    data_.release();
}

void cv::superres::GpuSparseMat_CSR::setDataImpl(const void* vals, const int* rowPtr, const int* colInd)
{
    cudaSafeCall( cudaMemcpy(data_.ptr(0), vals, nonZeroCount_ * CV_ELEM_SIZE(type_), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(data_.ptr(1), rowPtr, (rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(data_.ptr(2), colInd, nonZeroCount_ * sizeof(int), cudaMemcpyHostToDevice) );
}

///////////////////////////////////////////////////////////////
// BilateralTotalVariation_GPU

cv::superres::BilateralTotalVariation_GPU::BilateralTotalVariation_GPU()
{
    scale = 4;
    iterations = 180;
    beta = 1.3;
    lambda = 0.03;
    alpha = 0.7;
    btvKernelSize = 7;
    workDepth = CV_32F;
    blurModel = BLUR_GAUSS;
    blurKernelSize = 5;
    setMotionModel(MM_AFFINE);

    cusparseCall( cusparseCreate(&handle) );
    cusparseCall( cusparseCreateMatDescr(&descr) );
}

cv::superres::BilateralTotalVariation_GPU::~BilateralTotalVariation_GPU()
{
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

void cv::superres::BilateralTotalVariation_GPU::setMotionModel(int motionModel)
{
    CV_DbgAssert(motionModel >= MM_TRANSLATION && motionModel <= MM_UNKNOWN);

    motionEstimator = MotionEstimator::create(static_cast<MotionModel>(motionModel), true);
    this->motionModel = motionModel;
}

namespace
{
    void mulSparseMat(cusparseHandle_t handle, cusparseMatDescr_t descr, const GpuSparseMat_CSR& smat, const GpuMat& src, GpuMat& dst, Size dstSize, bool isTranspose = false)
    {
        CV_DbgAssert(smat.type() == CV_32FC1 || smat.type() == CV_64FC1);
        CV_DbgAssert(src.type() == smat.type());
        CV_DbgAssert(src.isContinuous());
        #ifdef _DEBUG
            if (isTranspose)
            {
                CV_DbgAssert(smat.rows() == src.size().area());
                CV_DbgAssert(smat.cols() == dstSize.area());
            }
            else
            {
                CV_DbgAssert(smat.rows() == dstSize.area());
                CV_DbgAssert(smat.cols() == src.size().area());
            }
        #endif

        createContinuous(dstSize, src.type(), dst);

        if (smat.type() == CV_32FC1)
        {
            float one = 1.0f;
            float zero = 0.0f;
            cusparseCall( cusparseScsrmv_v2(handle,
                                            isTranspose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            smat.rows(), smat.cols(), smat.nonZeroCount(), &one, descr,
                                            smat.vals<float>(), smat.rowPtr(), smat.colInd(), src.ptr<float>(), &zero, dst.ptr<float>()) );
            cudaSafeCall( cudaDeviceSynchronize() );
        }
        else
        {
            double one = 1.0;
            double zero = 0.0;
            cusparseCall( cusparseDcsrmv_v2(handle,
                                            isTranspose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            smat.rows(), smat.cols(), smat.nonZeroCount(), &one, descr,
                                            smat.vals<double>(), smat.rowPtr(), smat.colInd(), src.ptr<double>(), &zero, dst.ptr<double>()) );
            cudaSafeCall( cudaDeviceSynchronize() );
        }
    }

    void diffSign(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())
    {
        CV_DbgAssert(src1.channels() == src2.channels());
        CV_DbgAssert(src1.depth() == CV_32F || src1.depth() == CV_64F);

        typedef void (*func_t)(const PtrStepSzb& src1, const PtrStepSzb& src2, const PtrStepSzb& dst, cudaStream_t stream);
        static const func_t funcs[] =
        {
            btv_device::diffSign<float>,
            btv_device::diffSign<double>
        };

        dst.create(src1.size(), src1.type());

        const func_t func = funcs[src1.depth() == CV_64F];

        GpuMat dst1cn = dst.reshape(1);
        func(src1.reshape(1), src2.reshape(1), dst1cn, StreamAccessor::getStream(stream));
    }

    void calcBtvDiffTerm(const GpuMat& y, const GpuSparseMat_CSR& DHF, const GpuMat& X, GpuMat& diffTerm, GpuMat& buf,
                         cusparseHandle_t handle, cusparseMatDescr_t descr)
    {
        // degrade current estimated image
        mulSparseMat(handle, descr, DHF, X, buf, y.size());

        // compere input and degraded image
        diffSign(buf, y, buf);

        // blur the subtructed vector with transposed matrix
        mulSparseMat(handle, descr, DHF, buf, diffTerm, X.size(), true);
    }

    void calcBtvRegularization(const GpuMat& X, GpuMat& dst, int btvKernelSize, double alpha)
    {
    }
}

void cv::superres::BilateralTotalVariation_GPU::process(const vector<GpuMat>& y, const vector<GpuSparseMat_CSR>& DHF, int count, OutputArray dst)
{
    CV_DbgAssert(count > 0);
    CV_DbgAssert(y.size() >= count);
    CV_DbgAssert(DHF.size() >= count);

    Size lowResSize = y.front().size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

#ifdef _DEBUG
    for (size_t i = 0; i < count; ++i)
    {
        CV_DbgAssert(y[i].isContinuous());
        CV_DbgAssert(y[i].size() == lowResSize);
        CV_DbgAssert(DHF[i].rows() == lowResSize.area());
    }
#endif

    // create initial image by simple bi-cubic interpolation

    createContinuous(highResSize, y.front().type(), X);
    resize(y.front(), X, highResSize, 0, 0, INTER_CUBIC);

    // steepest descent method for L1 norm minimization

    for (int i = 0; i < iterations; ++i)
    {
        X.copyTo(Xout);

        // diff terms
        for (int i = 0; i < count; ++i)
        {
            calcBtvDiffTerm(y[i], DHF[i], X, diffTerm, buf, handle, descr);
            addWeighted(Xout, 1.0, diffTerm, -beta, 0.0, Xout);
        }

        // regularization term

        if (lambda > 0)
        {
            calcBtvRegularization(X, regTerm, btvKernelSize, alpha);
            addWeighted(Xout, 1.0, regTerm, -beta * lambda, 0.0, Xout);
        }
    }

    if (dst.kind() == _InputArray::GPU_MAT)
        Xout.convertTo(dst.getGpuMatRef(), CV_8U);
    else
    {
        Xout.convertTo(d_dst, CV_8U);
        dst.create(d_dst.size(), d_dst.type());
        Mat h_dst = dst.getMat();
        d_dst.download(h_dst);
    }
}

///////////////////////////////////////////////////////////////
// Tests

#ifdef WITH_TESTS

namespace cv
{
    namespace superres
    {
        TEST(MulSparseMat_GPU, Accuracy)
        {
            Mat_<float> src(1, 10);
            src << 1,2,3,4,5,6,7,8,9,10;

            Mat_<float> dst_gold(1, 5);
            dst_gold << 3,12,18,1,0;

            float smat_vals[8] = {1, 1, 1, 1, 1, 1, 1, 1};
            int smat_rowPtr[6] = {0, 2, 4, 7, 8, 8};
            int smat_colInd[8] = {0, 1, 2, 8, 4, 5, 6, 0};

            vector<float> vals(smat_vals, smat_vals + 8);
            vector<int> rowPtr(smat_rowPtr, smat_rowPtr + 6);
            vector<int> colInd(smat_colInd, smat_colInd + 8);

            GpuSparseMat_CSR smat(5, 10, 8, CV_32FC1);
            smat.setData(vals, rowPtr, colInd);

            cusparseHandle_t handle;
            cusparseMatDescr_t descr;
            cusparseCall( cusparseCreate(&handle) );
            cusparseCall( cusparseCreateMatDescr(&descr) );

            GpuMat d_src(src);
            GpuMat d_dst;
            mulSparseMat(handle, descr, smat, d_src, d_dst, Size(5, 1));

            cusparseCall( cusparseDestroy(handle) );
            cusparseCall( cusparseDestroyMatDescr(descr) );

            Mat dst(d_dst);

            const double diff = norm(dst_gold, dst, NORM_INF);
            EXPECT_EQ(0, diff);
        }

        TEST(DiffSign_GPU, Accuracy)
        {
            Mat_<float> src1(1, 3);
            src1 << 1, 2, 3;

            Mat_<float> src2(1, 3);
            src2 << 3, 2, 1;

            Mat_<float> gold(1, 3);
            gold << -1, 0, 1;

            GpuMat d_src1(src1);
            GpuMat d_src2(src2);
            GpuMat d_dst;
            diffSign(d_src1, d_src2, d_dst);

            Mat dst(d_dst);

            const double diff = norm(gold, dst, NORM_INF);
            EXPECT_EQ(0, diff);
        }

        TEST(CalcBtvDiffTerm_GPU, Accuracy)
        {
            Mat_<float> X(3, 3, 2.0f);

            float DHF_vals[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
            int DHF_rowPtr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            int DHF_colInd[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

            vector<float> vals(DHF_vals, DHF_vals + 9);
            vector<int> rowPtr(DHF_rowPtr, DHF_rowPtr + 10);
            vector<int> colInd(DHF_colInd, DHF_colInd + 9);

            GpuSparseMat_CSR DHF(9, 9, 9, CV_32FC1);
            DHF.setData(vals, rowPtr, colInd);

            Mat_<float> y(3, 3);
            y << 1,1,1,2,2,2,3,3,3;

            Mat_<float> gold(3, 3);
            gold << 1,1,1,0,0,0,-1,-1,-1;

            cusparseHandle_t handle;
            cusparseMatDescr_t descr;
            cusparseCall( cusparseCreate(&handle) );
            cusparseCall( cusparseCreateMatDescr(&descr) );

            GpuMat d_y;
            GpuMat d_X;
            GpuMat d_dst;
            GpuMat d_buf;

            createContinuous(y.size(), y.type(), d_y);
            d_y.upload(y);
            createContinuous(X.size(), X.type(), d_X);
            d_X.upload(X);

            calcBtvDiffTerm(d_y, DHF, d_X, d_dst, d_buf, handle, descr);

            cusparseCall( cusparseDestroy(handle) );
            cusparseCall( cusparseDestroyMatDescr(descr) );

            Mat dst(d_dst);

            const double diff = norm(gold, dst, NORM_INF);
            EXPECT_EQ(0, diff);
        }
    }
}

#endif // WITH_TESTS
