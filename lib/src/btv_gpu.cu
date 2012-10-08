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
#include <opencv2/gpu/device/functional.hpp>

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace
{
    template <typename T> __device__ __forceinline__ T diffSign(T a, T b)
    {
        return a > b ? static_cast<T>(1) : a < b ? static_cast<T>(-1) : static_cast<T>(0);
    }

    template <typename T> struct DiffSign : binary_function<T, T, T>
    {
        __device__ __forceinline__ T operator ()(T a, T b) const
        {
            return diffSign(a, b);
        }
    };
}

namespace cv { namespace gpu { namespace device
{
    template <> struct TransformFunctorTraits< DiffSign<float> > : DefaultTransformFunctorTraits< DiffSign<float> >
    {
        enum { smart_block_dim_y = 8 };
        enum { smart_shift = 4 };
    };
}}}

namespace btv_device
{
    template <typename T> void diffSign(const PtrStepSzb& src1, const PtrStepSzb& src2, const PtrStepSzb& dst, cudaStream_t stream)
    {
        transform((PtrStepSz<T>)src1, (PtrStepSz<T>)src2, (PtrStepSz<T>)dst, DiffSign<T>(), WithOutMask(), stream);
    }

    template void diffSign<float>(const PtrStepSzb& src1, const PtrStepSzb& src2, const PtrStepSzb& dst, cudaStream_t stream);
    template void diffSign<double>(const PtrStepSzb& src1, const PtrStepSzb& src2, const PtrStepSzb& dst, cudaStream_t stream);
}
