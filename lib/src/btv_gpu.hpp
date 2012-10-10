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

#pragma once

#ifndef __BTV_GPU_HPP__
#define __BTV_GPU_HPP__

#include <vector>
#include <cusparse_v2.h>
#include <opencv2/gpu/gpu.hpp>
#include "super_resolution.hpp"
#include "motion_estimation.hpp"
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        using cv::gpu::GpuMat;

        class SUPER_RESOLUTION_NO_EXPORT GpuSparseMat_CSR
        {
        public:
            GpuSparseMat_CSR() : rows_(0), cols_(0), nonZeroCount_(0) {}
            GpuSparseMat_CSR(int rows, int cols, int nonZeroCount, int type) : rows_(0), cols_(0), nonZeroCount_(0) { create(rows, cols, nonZeroCount, type); }
            GpuSparseMat_CSR(Size size, int nonZeroCount, int type) : rows_(0), cols_(0), nonZeroCount_(0) { create(size, nonZeroCount, type); }

            void create(int rows, int cols, int nonZeroCount, int type);
            void create(Size size, int nonZeroCount, int type) { create(size.height, size.width, nonZeroCount, type); }

            void release();

            template <typename T>
            void setData(const std::vector<T>& vals, const std::vector<int>& rowPtr, const std::vector<int>& colInd);

            template <typename T> T* vals() { return data_.ptr<T>(0); }
            template <typename T> const T* vals() const { return data_.ptr<T>(0); }

            int* rowPtr() { return data_.ptr<int>(1); }
            const int* rowPtr() const { return data_.ptr<int>(1); }

            int* colInd() { return data_.ptr<int>(2); }
            const int* colInd() const { return data_.ptr<int>(2); }

            int rows() const { return rows_; }
            int cols() const { return cols_; }
            Size size() const { return Size(cols(), rows()); }
            bool empty() const { return data_.empty(); }
            int nonZeroCount() const { return nonZeroCount_; }

            int type() const { return type_; }
            int depth() const { return CV_MAT_DEPTH(type_); }
            int channels() const { return CV_MAT_CN(type_); }

        private:
            void setDataImpl(const void* vals, const int* rowPtr, const int* colInd);

            int rows_;
            int cols_;
            int nonZeroCount_;
            int type_;
            GpuMat data_;
        };

        template <typename T>
        void GpuSparseMat_CSR::setData(const std::vector<T>& vals, const std::vector<int>& rowPtr, const std::vector<int>& colInd)
        {
            CV_DbgAssert(DataType<T>::type == type_);
            CV_DbgAssert(vals.size() == nonZeroCount_);
            CV_DbgAssert(rowPtr.size() == rows_ + 1);
            CV_DbgAssert(colInd.size() == nonZeroCount_);
            setDataImpl(&vals[0], &rowPtr[0], &colInd[0]);
        }

        // S. Farsiu , D. Robinson, M. Elad, P. Milanfar. Fast and robust multiframe super resolution.
        class SUPER_RESOLUTION_NO_EXPORT BTV_GPU_Base
        {
        protected:
            BTV_GPU_Base();
            ~BTV_GPU_Base();

            void process(const GpuMat& src, GpuMat& dst, const std::vector<GpuMat>& y, const std::vector<GpuSparseMat_CSR>& DHF, int count);

            static void calcBlurWeights(BlurModel blurModel, int blurKernelSize, std::vector<float>& blurWeights);
            static void calcDhf(Size lowResSize, int scale, int blurKernelSize, const std::vector<float>& blurWeights,
                                MotionModel motionModel, const Mat& m1, const Mat& m2,
                                std::vector<float>& vals, std::vector<int>& rowPtr, std::vector<int>& colInd,
                                GpuSparseMat_CSR& DHF);

            int scale;
            int iterations;
            double beta;
            double lambda;
            double alpha;
            int btvKernelSize;

        private:
            GpuMat X;
            GpuMat Xout;
            GpuMat diffTerm;
            GpuMat regTerm;
            GpuMat buf;

            std::vector<float> btvWeights;

            cusparseHandle_t handle;
            cusparseMatDescr_t descr;
        };

        class SUPER_RESOLUTION_NO_EXPORT BTV_GPU : public SuperResolution, private BTV_GPU_Base
        {
        public:
            AlgorithmInfo* info() const;

            static bool init();
            static Ptr<SuperResolution> create();

            BTV_GPU();

        protected:
            void initImpl(Ptr<IFrameSource>& frameSource);
            Mat processImpl(Ptr<IFrameSource>& frameSource);

        private:
            void setMotionModel(int motionModel);
            void addNewFrame(const Mat& frame);
            void processFrame(int idx);

            int motionModel;
            int blurModel;
            int blurKernelSize;
            int temporalAreaRadius;

            Ptr<MotionEstimator> motionEstimator;
            Mat m1, m2;

            std::vector<GpuMat> frames;
            std::vector<GpuMat> results;
            Mat h_dst;

            GpuMat src_f;
            std::vector<GpuMat> y;
            std::vector<GpuSparseMat_CSR> DHF;

            int storePos;
            int procPos;
            int outPos;

            std::vector<float> blurWeights;
            int curBlurModel;

            std::vector<float> valsBuf;
            std::vector<int> rowPtrBuf;
            std::vector<int> colIndBuf;
        };
    }
}

#endif // __BTV_GPU_HPP__
