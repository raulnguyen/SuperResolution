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
#include "image_super_resolution.hpp"
#include "video_super_resolution.hpp"
#include "motion_estimation.hpp"
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
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
            gpu::GpuMat data_;
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
        class SUPER_RESOLUTION_NO_EXPORT BilateralTotalVariation_GPU
        {
        protected:
            BilateralTotalVariation_GPU();
            ~BilateralTotalVariation_GPU();

            void setMotionModel(int motionModel);

            void process(const std::vector<gpu::GpuMat>& y, const std::vector<GpuSparseMat_CSR>& DHF, int count, OutputArray dst);

            int scale;
            int iterations;
            double beta;
            double lambda;
            double alpha;
            int btvKernelSize;
            int motionModel;
            int blurModel;
            int blurKernelSize;

            Ptr<MotionEstimator> motionEstimator;

        private:
            gpu::GpuMat X;
            gpu::GpuMat Xout;
            gpu::GpuMat diffTerm;
            gpu::GpuMat regTerm;
            gpu::GpuMat buf;
            gpu::GpuMat d_dst;
            gpu::GpuMat yBuf;

            cusparseHandle_t handle;
            cusparseMatDescr_t descr;
        };

        class SUPER_RESOLUTION_NO_EXPORT BTV_Image_GPU : public ImageSuperResolution, private BilateralTotalVariation_GPU
        {
        public:
            static bool init();
            static Ptr<ImageSuperResolution> create();

            BTV_Image_GPU();

            AlgorithmInfo* info() const;

            void train(InputArrayOfArrays images);

            bool empty() const;
            void clear();

            void process(InputArray src, OutputArray dst);

        private:
            void trainImpl(const std::vector<Mat>& images);

            std::vector<Mat> images;

            std::vector<gpu::GpuMat> y;
            std::vector<GpuSparseMat_CSR> DHF;

            Mat m1, m2;

            gpu::GpuMat srcBuf;
            gpu::GpuMat curImageBuf;

            std::vector<float> valsBuf;
            std::vector<int> rowPtrBuf;
            std::vector<int> colIndBuf;

            Mat_<float> blurWeights;
            int curBlurModel;
        };

        class SUPER_RESOLUTION_NO_EXPORT BTV_Video_GPU : public VideoSuperResolution, private BilateralTotalVariation_GPU
        {
        public:
            static bool init();
            static Ptr<VideoSuperResolution> create();

            AlgorithmInfo* info() const;

            BTV_Video_GPU();

        protected:
            void initImpl(Ptr<IFrameSource>& frameSource);
            Mat processImpl(const Mat& frame);

        private:
            void addNewFrame(const Mat& frame);
            void processFrame(int idx);

            int temporalAreaRadius;

            std::vector<Mat> frames;
            std::vector<Mat> results;

            std::vector<gpu::GpuMat> y;
            std::vector<GpuSparseMat_CSR> DHF;

            int storePos;
            int procPos;
            int outPos;

            Mat m1, m2;

            gpu::GpuMat srcBuf;
            gpu::GpuMat curImageBuf;

            std::vector<float> valsBuf;
            std::vector<int> rowPtrBuf;
            std::vector<int> colIndBuf;

            Mat_<float> blurWeights;
            int curBlurModel;
        };
    }
}

#endif // __BTV_GPU_HPP__
