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
#include <opencv2/videostab/ring_buffer.hpp>
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
    void diffSign(PtrStepSzf src1, PtrStepSzf src2, PtrStepSzf dst, cudaStream_t stream);

    template <int cn>
    void calcBtvRegularization(PtrStepSzb src, PtrStepSzb dst, int ksize, const float* weights, int count, cudaStream_t stream);
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
// BTV_GPU_Base

cv::superres::BTV_GPU_Base::BTV_GPU_Base()
{
    scale = 4;
    iterations = 180;
    beta = 1.3;
    lambda = 0.03;
    alpha = 0.7;
    btvKernelSize = 7;

    cusparseCall( cusparseCreate(&handle) );
    cusparseCall( cusparseCreateMatDescr(&descr) );
}

cv::superres::BTV_GPU_Base::~BTV_GPU_Base()
{
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}

namespace
{
    void calcBtvWeights(int btvKernelSize, double alpha, vector<float>& btvWeights)
    {
        CV_Assert(btvKernelSize <= 16);
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

    void mulSparseMat(cusparseHandle_t handle, cusparseMatDescr_t descr, const GpuSparseMat_CSR& smat, const GpuMat& src, GpuMat& dst, Size dstSize, bool isTranspose = false)
    {
        CV_DbgAssert(smat.type() == CV_32FC1);
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

        float one = 1.0f;
        float zero = 0.0f;
        cusparseCall( cusparseScsrmv_v2(handle,
                                        isTranspose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        smat.rows(), smat.cols(), smat.nonZeroCount(), &one, descr,
                                        smat.vals<float>(), smat.rowPtr(), smat.colInd(), src.ptr<float>(), &zero, dst.ptr<float>()) );
        cudaSafeCall( cudaDeviceSynchronize() );
    }

    void diffSign(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream = Stream::Null())
    {
        CV_DbgAssert(src1.depth() == CV_32F);
        CV_DbgAssert(src1.type() == src2.type());
        CV_DbgAssert(src1.size() == src2.size());

        dst.create(src1.size(), src1.type());

        btv_device::diffSign(src1.reshape(1), src2.reshape(1), dst.reshape(1), StreamAccessor::getStream(stream));
    }

    void calcBtvDiffTerm(const GpuMat& y, const GpuSparseMat_CSR& DHF, const GpuMat& X, GpuMat& diffTerm, GpuMat& buf,
                         cusparseHandle_t handle, cusparseMatDescr_t descr)
    {
        mulSparseMat(handle, descr, DHF, X, buf, y.size());

        diffSign(buf, y, buf);

        mulSparseMat(handle, descr, DHF, buf, diffTerm, X.size(), true);
    }

    void calcBtvRegularization(const GpuMat& X, GpuMat& dst, int btvKernelSize, const vector<float>& btvWeights)
    {
        typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, int ksize, const float* weights, int count, cudaStream_t stream);
        static const func_t funcs[] =
        {
            0,
            btv_device::calcBtvRegularization<1>,
            0,
            btv_device::calcBtvRegularization<3>,
            btv_device::calcBtvRegularization<4>
        };

        CV_DbgAssert(X.depth() == CV_32F);
        CV_DbgAssert(X.channels() == 1 || X.channels() == 3 || X.channels() == 4);
        CV_DbgAssert(btvKernelSize > 0 && btvKernelSize <= 16);
        CV_DbgAssert(btvWeights.size() == btvKernelSize * btvKernelSize);

        dst.create(X.size(), X.type());
        dst.setTo(Scalar::all(0));

        const int ksize = (btvKernelSize - 1) / 2;

        funcs[X.channels()](X, dst, ksize, &btvWeights[0], static_cast<int>(btvWeights.size()), 0);
    }
}

void cv::superres::BTV_GPU_Base::process(const GpuMat& src, GpuMat& dst, const std::vector<GpuMat>& y, const std::vector<GpuSparseMat_CSR>& DHF, int count)
{
    CV_DbgAssert(src.depth() == CV_32F);
    CV_DbgAssert(count > 0);
    CV_DbgAssert(y.size() >= count);
    CV_DbgAssert(DHF.size() >= count);

    calcBtvWeights(btvKernelSize, alpha, btvWeights);

    Size lowResSize = src.size();
    const Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

#ifdef _DEBUG
    for (size_t i = 0; i < count; ++i)
    {
        CV_DbgAssert(y[i].isContinuous());
        CV_DbgAssert(y[i].size() == lowResSize);
        CV_DbgAssert(y[i].type() == src.type());
        CV_DbgAssert(DHF[i].rows() == lowResSize.area());
        CV_DbgAssert(DHF[i].cols() == highResSize.area());
        CV_DbgAssert(DHF[i].type() == CV_32FC1);
    }
#endif

    createContinuous(highResSize, src.type(), X);
    createContinuous(highResSize, src.type(), Xout);

    resize(src, X, highResSize, 0, 0, INTER_CUBIC);

    for (int i = 0; i < iterations; ++i)
    {
        for (int k = 0; k < count; ++k)
        {
            if (src.channels() == 1)
            {
                calcBtvDiffTerm(y[k], DHF[k], X, diffTerm, buf, handle, descr);
                addWeighted(k == 0 ? X : Xout, 1.0, diffTerm, -beta, 0.0, Xout);
            }
            else
            {
                y_cn.resize(src.channels());
                X_cn.resize(src.channels());
                Xout_cn.resize(src.channels());

                for (int c = 0; c < src.channels(); ++c)
                {
                    createContinuous(lowResSize, CV_32FC1, y_cn[c]);
                    createContinuous(highResSize, CV_32FC1, X_cn[c]);
                }

                split(y[k], y_cn);
                split(X, X_cn);

                for (int c = 0; c < src.channels(); ++c)
                {
                    calcBtvDiffTerm(y_cn[c], DHF[k], X_cn[c], diffTerm, buf, handle, descr);
                    addWeighted(k == 0 ? X_cn[c] : Xout_cn[c], 1.0, diffTerm, -beta, 0.0, Xout_cn[c]);
                }

                merge(Xout_cn, Xout);
            }
        }

        if (lambda > 0)
        {
            calcBtvRegularization(X, regTerm, btvKernelSize, btvWeights);
            addWeighted(Xout, 1.0, regTerm, -beta * lambda, 0.0, Xout);
        }

        Xout.swap(X);
    }

    X(Rect(btvKernelSize, btvKernelSize, X.cols - 2 * btvKernelSize, X.rows - 2 * btvKernelSize)).convertTo(dst, CV_8U);
}

void cv::superres::BTV_GPU_Base::calcBlurWeights(BlurModel blurModel, int blurKernelSize, std::vector<float>& blurWeights)
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

    template <class Motion>
    void calcDhfImpl(Size lowResSize, int scale, int blurKernelSize, const vector<float>& blurWeights, const Motion& motion,
                     vector<float>& vals, vector<int>& rowPtr, vector<int>& colInd, GpuSparseMat_CSR& DHF)
    {
        CV_DbgAssert(scale > 1);
        CV_DbgAssert(blurKernelSize > 0);
        CV_DbgAssert(blurWeights.size() == blurKernelSize * blurKernelSize);

        Size highResSize(lowResSize.width * scale, lowResSize.height * scale);

        vals.clear();
        rowPtr.resize(lowResSize.area() + 1, 0);
        colInd.clear();

        for (int y = 0, lowResInd = 0; y < lowResSize.height; ++y)
        {
            for (int x = 0; x < lowResSize.width; ++x, ++lowResInd)
            {
                Point2f lowOrigCoord = motion.calcCoord(Point(x, y));

                int count = 0;

                for (int i = 0, blurKerInd = 0; i < blurKernelSize; ++i)
                {
                    for (int j = 0; j < blurKernelSize; ++j, ++blurKerInd)
                    {
                        const float w = blurWeights[blurKerInd];

                        const int X = cvFloor(lowOrigCoord.x * scale + j - blurKernelSize / 2);
                        const int Y = cvFloor(lowOrigCoord.y * scale + i - blurKernelSize / 2);

                        if (X >= 0 && X < highResSize.width && Y >= 0 && Y < highResSize.height)
                        {
                            vals.push_back(w);
                            colInd.push_back(Y * highResSize.width + X);
                            ++count;
                        }
                    }
                }

                rowPtr[lowResInd + 1] = rowPtr[lowResInd] + count;
            }
        }

        DHF.create(lowResSize.area(), highResSize.area(), vals.size(), CV_32FC1);
        DHF.setData(vals, rowPtr, colInd);
    }
}

void cv::superres::BTV_GPU_Base::calcDhf(Size lowResSize, int scale, int blurKernelSize, const vector<float>& blurWeights,
                                         MotionModel motionModel, const Mat& m1, const Mat& m2,
                                         vector<float>& vals, vector<int>& rowPtr, vector<int>& colInd,
                                         GpuSparseMat_CSR& DHF)
{
    if (motionModel == MM_UNKNOWN)
    {
        CV_DbgAssert(m1.type() == CV_32FC1);
        CV_DbgAssert(m1.size() == lowResSize);
        CV_DbgAssert(m1.size() == m2.size());
        CV_DbgAssert(m1.type() == m2.type());

        GeneralMotion motion(m1, m2);
        calcDhfImpl(lowResSize, scale, blurKernelSize, blurWeights, motion, vals, rowPtr, colInd, DHF);
    }
    else if (motionModel < MM_HOMOGRAPHY)
    {
        CV_DbgAssert(m1.type() == CV_32FC1);
        CV_DbgAssert(m1.rows == 2 || m1.rows == 3);
        CV_DbgAssert(m1.cols == 3);

        AffineMotion motion(m1);
        calcDhfImpl(lowResSize, scale, blurKernelSize, blurWeights, motion, vals, rowPtr, colInd, DHF);
    }
    else
    {
        CV_DbgAssert(m1.type() == CV_32FC1);
        CV_DbgAssert(m1.rows == 3);
        CV_DbgAssert(m1.cols == 3);

        PerspectiveMotion motion(m1);
        calcDhfImpl(lowResSize, scale, blurKernelSize, blurWeights, motion, vals, rowPtr, colInd, DHF);
    }
}

///////////////////////////////////////////////////////////////
// BTV_GPU

namespace cv
{
    namespace superres
    {
        typedef void (Algorithm::*IntSetter)(int);

        CV_INIT_ALGORITHM(BTV_GPU, "SuperResolution.BTV_GPU",
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
                          obj.info()->addParam(obj, "motionModel", obj.motionModel, false, 0, (IntSetter) &BTV_GPU::setMotionModel,
                                              "Motion model between frames.");
                          obj.info()->addParam(obj, "blurModel", obj.blurModel, false, 0, 0,
                                               "Blur model.");
                          obj.info()->addParam(obj, "blurKernelSize", obj.blurKernelSize, false, 0, 0,
                                               "Blur kernel size (if -1, than it will be equal scale factor)."));
    }
}

bool cv::superres::BTV_GPU::init()
{
    return !BTV_GPU_info_auto.name().empty();
}

Ptr<SuperResolution> cv::superres::BTV_GPU::create()
{
    Ptr<SuperResolution> alg(new BTV_GPU);
    return alg;
}

cv::superres::BTV_GPU::BTV_GPU()
{
    setMotionModel(MM_AFFINE);
    blurModel = BLUR_GAUSS;
    blurKernelSize = 5;
    temporalAreaRadius = 4;

    curBlurModel = -1;
}

void cv::superres::BTV_GPU::setMotionModel(int motionModel)
{
    CV_DbgAssert(motionModel >= MM_TRANSLATION && motionModel <= MM_UNKNOWN);

    motionEstimator = MotionEstimator::create(static_cast<MotionModel>(motionModel), true);
    this->motionModel = motionModel;
}

void cv::superres::BTV_GPU::initImpl(Ptr<IFrameSource>& frameSource)
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

Mat cv::superres::BTV_GPU::processImpl(Ptr<IFrameSource>& frameSource)
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

void cv::superres::BTV_GPU::addNewFrame(const Mat& frame)
{
    if (frame.empty())
        return;

    CV_DbgAssert(storePos < 0 || frame.size() == at(storePos, frames).size());

    ++storePos;
    at(storePos, frames).upload(frame);
}

void cv::superres::BTV_GPU::processFrame(int idx)
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

    GpuMat src = at(idx, frames);
    src.convertTo(src_f, CV_32F);

    for (size_t k = 0; k < frames.size(); ++k)
    {
        GpuMat curImage = frames[k];

        bool ok = motionEstimator->estimate(curImage, src, m1, m2);

        if (ok)
        {
            createContinuous(curImage.size(), CV_32FC(src.channels()), y[count]);
            curImage.convertTo(y[count], CV_32F);
            calcDhf(src.size(), scale, blurKernelSize, blurWeights, static_cast<MotionModel>(motionModel), m1, m2,
                    valsBuf, rowPtrBuf, colIndBuf, DHF[count]);
            ++count;
        }
    }

    process(src_f, at(idx, results), y, DHF, count);
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
