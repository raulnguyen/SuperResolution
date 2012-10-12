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

#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/transform.hpp>
#include <opencv2/gpu/device/vec_traits.hpp>
#include <opencv2/gpu/device/vec_math.hpp>

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace
{
    __global__ void buildMotionMaps(const PtrStepSzf motionx, const PtrStepf motiony,
                                    PtrStepf forwardx, PtrStepf forwardy,
                                    PtrStepf backwardx, PtrStepf backwardy)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= motionx.cols || y >= motionx.rows)
            return;

        const float mx = motionx(y, x);
        const float my = motiony(y, x);

        forwardx(y, x) = x - mx;
        forwardy(y, x) = y - my;

        backwardx(y, x) = x + mx;
        backwardy(y, x) = y + my;
    }
}

namespace tv_l1_device
{
    void buildMotionMaps(PtrStepSzf motionx, PtrStepSzf motiony, PtrStepSzf forwardx, PtrStepSzf forwardy, PtrStepSzf backwardx, PtrStepSzf backwardy)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(motionx.cols, block.x), divUp(motionx.rows, block.y));

        ::buildMotionMaps<<<grid, block>>>(motionx, motiony, forwardx, forwardy, backwardx, backwardy);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
}

namespace
{
    template <typename T>
    __global__ void upscale(const PtrStepSz<T> src, PtrStep<T> dst, const int scale)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= src.cols || y >= src.rows)
            return;

        dst(y * scale, x * scale) = src(y, x);
    }
}

namespace tv_l1_device
{
    template <int cn>
    void upscale(const PtrStepSzb src, PtrStepSzb dst, int scale)
    {
        typedef typename TypeVec<float, cn>::vec_type src_t;

        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        ::upscale<src_t><<<grid, block>>>((PtrStepSz<src_t>) src, (PtrStepSz<src_t>) dst, scale);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void upscale<1>(const PtrStepSzb src, PtrStepSzb dst, int scale);
    template void upscale<3>(const PtrStepSzb src, PtrStepSzb dst, int scale);
    template void upscale<4>(const PtrStepSzb src, PtrStepSzb dst, int scale);
}

namespace
{
    __device__ __forceinline__ float diffSign(float a, float b)
    {
        return a > b ? 1.0f : a < b ? -1.0f : 0.0f;
    }
    __device__ __forceinline__ float3 diffSign(const float3& a, const float3& b)
    {
        return make_float3(
            a.x > b.x ? 1.0f : a.x < b.x ? -1.0f : 0.0f,
            a.y > b.y ? 1.0f : a.y < b.y ? -1.0f : 0.0f,
            a.z > b.z ? 1.0f : a.z < b.z ? -1.0f : 0.0f
        );
    }
    __device__ __forceinline__ float4 diffSign(const float4& a, const float4& b)
    {
        return make_float4(
            a.x > b.x ? 1.0f : a.x < b.x ? -1.0f : 0.0f,
            a.y > b.y ? 1.0f : a.y < b.y ? -1.0f : 0.0f,
            a.z > b.z ? 1.0f : a.z < b.z ? -1.0f : 0.0f,
            0.0f
        );
    }

    struct DiffSign : binary_function<float, float, float>
    {
        __device__ __forceinline__ float operator ()(float a, float b) const
        {
            return diffSign(a, b);
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits<DiffSign> : DefaultTransformFunctorTraits<DiffSign>
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
}}}

namespace tv_l1_device
{
    void diffSign(PtrStepSzf src1, PtrStepSzf src2, PtrStepSzf dst)
    {
        transform(src1, src2, dst, DiffSign(), WithOutMask(), 0);
    }
}

namespace
{
    __constant__ float c_btvRegWeights[16*16];

    template <typename T>
    __global__ void calcBtvRegularization(const PtrStepSz<T> src, PtrStep<T> dst, const int ksize)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x + ksize;
        const int y = blockIdx.y * blockDim.y + threadIdx.y + ksize;

        if (y >= src.rows - ksize || x >= src.cols - ksize)
            return;

        const T srcVal = src(y, x);

        T dstVal = VecTraits<T>::all(0);

        for (int m = 0, count = 0; m <= ksize; ++m)
        {
            for (int l = ksize; l + m >= 0; --l, ++count)
                dstVal = dstVal + c_btvRegWeights[count] * (diffSign(srcVal, src(y + m, x + l)) - diffSign(src(y - m, x - l), srcVal));
        }

        dst(y, x) = dstVal;
    }
}

namespace tv_l1_device
{
    void loadBtvWeights(const float* weights, size_t count)
    {
        cudaSafeCall( cudaMemcpyToSymbol(c_btvRegWeights, weights, count * sizeof(float)) );
    }

    template <int cn>
    void calcBtvRegularization(PtrStepSzb src, PtrStepSzb dst, int ksize)
    {
        typedef typename TypeVec<float, cn>::vec_type src_t;

        const dim3 block(32, 8);
        const dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

        ::calcBtvRegularization<src_t><<<grid, block>>>((PtrStepSz<src_t>) src, (PtrStepSz<src_t>) dst, ksize);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }

    template void calcBtvRegularization<1>(PtrStepSzb src, PtrStepSzb dst, int ksize);
    template void calcBtvRegularization<3>(PtrStepSzb src, PtrStepSzb dst, int ksize);
    template void calcBtvRegularization<4>(PtrStepSzb src, PtrStepSzb dst, int ksize);
}
