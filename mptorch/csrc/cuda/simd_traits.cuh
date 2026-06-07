#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

template <typename scalar_t>
struct SIMDTraits;

template <>
struct SIMDTraits<float>
{
    static constexpr int vec_elems = 4;
    template <typename F>
    static __device__ __forceinline__ float4 process(float4 v, F f)
    {
        return make_float4(f(v.x, 0), f(v.y, 1), f(v.z, 2), f(v.w, 3));
    }
};

template <>
struct SIMDTraits<double>
{
    static constexpr int vec_elems = 2;
    template <typename F>
    static __device__ __forceinline__ float4 process(float4 v, F f)
    {
        const double2 *in = reinterpret_cast<const double2 *>(&v);
        double2 out;
        out.x = f(static_cast<float>(in->x), 0);
        out.y = f(static_cast<float>(in->y), 1);
        return *reinterpret_cast<float4 *>(&out);
    }
};

template <>
struct SIMDTraits<c10::Half>
{
    static constexpr int vec_elems = 8;
    template <typename F>
    static __device__ __forceinline__ float4 process(float4 v, F f)
    {
        const __half2 *in = reinterpret_cast<const __half2 *>(&v);
        __half2 out[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 f2 = __half22float2(in[i]);
            f2.x = f(f2.x, i * 2);
            f2.y = f(f2.y, i * 2 + 1);
            out[i] = __float22half2_rn(f2);
        }
        return *reinterpret_cast<float4 *>(out);
    }
};

template <>
struct SIMDTraits<c10::BFloat16>
{
    static constexpr int vec_elems = 8;
    template <typename F>
    static __device__ __forceinline__ float4 process(float4 v, F f)
    {
        const __nv_bfloat162 *in = reinterpret_cast<const __nv_bfloat162 *>(&v);
        __nv_bfloat162 out[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            float2 f2 = __bfloat1622float2(in[i]);
#else
            const __nv_bfloat16 *b1 = reinterpret_cast<const __nv_bfloat16 *>(&in[i]);
            float2 f2 = make_float2(__bfloat162float(b1[0]), __bfloat162float(b1[1]));
#endif
            f2.x = f(f2.x, i * 2);
            f2.y = f(f2.y, i * 2 + 1);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            out[i] = __float22bfloat162_rn(f2);
#else
            __nv_bfloat16 *b1_out = reinterpret_cast<__nv_bfloat16 *>(&out[i]);
            b1_out[0] = __float2bfloat16(f2.x);
            b1_out[1] = __float2bfloat16(f2.y);
#endif
        }
        return *reinterpret_cast<float4 *>(out);
    }
};
