#include "../common/cast_binaryK.h"
#include "../quant_ops.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <climits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace at;

namespace
{
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

    template <typename scalar_t, RoundMode RM>
    struct BinaryKQuantizer
    {
        int man_bits, exp_bits, bias;
        bool is_signed;
        SaturationMode sat_mode;
        SubnormalsMode sub_mode;
        int prng_bits;

        __device__ __forceinline__ float eval(float x_f) const
        {
            if (RM == RoundMode::RNE)
                return cast_binaryK_nearest_even(x_f, man_bits, exp_bits, bias, is_signed, sat_mode, sub_mode);
            if (RM == RoundMode::RNA)
                return cast_binaryK_nearest_away(x_f, man_bits, exp_bits, bias, is_signed, sat_mode, sub_mode);
            if (RM == RoundMode::RU)
                return cast_binaryK_up(x_f, man_bits, exp_bits, bias, is_signed, sat_mode, sub_mode);
            if (RM == RoundMode::RD)
                return cast_binaryK_down(x_f, man_bits, exp_bits, bias, is_signed, sat_mode, sub_mode);
            if (RM == RoundMode::RZ)
                return cast_binaryK_zero(x_f, man_bits, exp_bits, bias, is_signed, sat_mode, sub_mode);
            return x_f;
        }

        __device__ __forceinline__ float eval_sr(float x_f, uint32_t rv) const
        {
            return cast_binaryK_stochastic(x_f, rv, prng_bits, man_bits, exp_bits, bias, is_signed, sat_mode, sub_mode);
        }

        __device__ __forceinline__ scalar_t scalar(scalar_t x) const
        {
            return static_cast<scalar_t>(eval(static_cast<float>(x)));
        }

        __device__ __forceinline__ scalar_t scalar_sr(scalar_t x, uint32_t rv) const
        {
            return static_cast<scalar_t>(eval_sr(static_cast<float>(x), rv));
        }

        __device__ __forceinline__ float4 vec(float4 x) const
        {
            return SIMDTraits<scalar_t>::process(x, [&](float val, int /*idx*/)
                                                 { return eval(val); });
        }

        __device__ __forceinline__ float4 vec_sr(float4 x, const int *r) const
        {
            return SIMDTraits<scalar_t>::process(x, [&](float val, int idx)
                                                 { return eval_sr(val, static_cast<uint32_t>(r[idx])); });
        }
    };

    template <typename scalar_t>
    void binaryK_kernel_impl(scalar_t *__restrict__ a, scalar_t *o, int size,
                             int K, int P, int bias, bool is_signed,
                             RoundMode round_mode,
                             SaturationMode saturation_mode,
                             SubnormalsMode subnormals_mode)
    {
        int blockSize = 256;
        int elements_per_vec = SIMDTraits<scalar_t>::vec_elems;
        int vec_size = size / elements_per_vec;
        int rem_size = size % elements_per_vec;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        auto launch_kernels = [&](auto quantizer)
        {
            if (vec_size > 0)
            {
                int blockNums = (vec_size + blockSize - 1) / blockSize;
                quant_kernel_vec<<<blockNums, blockSize, 0, stream>>>(a, o, vec_size, quantizer);
            }
            if (rem_size > 0)
            {
                quant_kernel_rem<<<1, rem_size, 0, stream>>>(a + vec_size * elements_per_vec, o + vec_size * elements_per_vec, rem_size, quantizer);
            }
        };

        int man_bits = P - 1;
        int exp_bits = is_signed ? K - P : K - P + 1;

        switch (round_mode)
        {
        case RoundMode::RNE:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RNE>{man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode, 0});
            break;
        case RoundMode::RNA:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RNA>{man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode, 0});
            break;
        case RoundMode::RU:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RU>{man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode, 0});
            break;
        case RoundMode::RD:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RD>{man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode, 0});
            break;
        default:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RZ>{man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode, 0});
            break;
        }
    }

    template <typename scalar_t>
    void binaryK_kernel_sr_impl(scalar_t *__restrict__ a, int *__restrict__ r, scalar_t *o, int size,
                                int K, int P, int bias, int prng_bits, bool is_signed,
                                SaturationMode saturation_mode,
                                SubnormalsMode subnormals_mode)
    {
        int blockSize = 256;
        int elements_per_vec = SIMDTraits<scalar_t>::vec_elems;
        int vec_size = size / elements_per_vec;
        int rem_size = size % elements_per_vec;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        int man_bits = P - 1;
        int exp_bits = is_signed ? K - P : K - P + 1;
        auto quantizer = BinaryKQuantizer<scalar_t, RoundMode::SR>{man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode, prng_bits};

        if (vec_size > 0)
        {
            int blockNums = (vec_size + blockSize - 1) / blockSize;
            quant_kernel_vec_sr<<<blockNums, blockSize, 0, stream>>>(a, r, o, vec_size, quantizer);
        }
        if (rem_size > 0)
        {
            quant_kernel_rem_sr<<<1, rem_size, 0, stream>>>(a + vec_size * elements_per_vec, r + vec_size * elements_per_vec, o + vec_size * elements_per_vec, rem_size, quantizer);
        }
    }

} // namespace

Tensor binaryK_quantize_cuda(
    Tensor a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
    int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
    auto o = empty_like(a);
    int size = a.numel();
    if (size == 0)
        return o;
    RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
    SubnormalsMode subnormals_mode_ = static_cast<SubnormalsMode>(subnormals_mode);
    SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);

    const int K_ = static_cast<int>(K);
    const int P_ = static_cast<int>(P);
    const int bias_ = static_cast<int>(bias);
    const int prng_bits_ = static_cast<int>(prng_bits);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "binaryK_quantize_cuda", [&]
                                    {
        scalar_t *p_a = a.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();

        if (round_mode_ != RoundMode::SR)
        {
            binaryK_kernel_impl<scalar_t>(
                p_a, p_o, size, K_, P_, bias_, is_signed,
                round_mode_, saturation_mode_, subnormals_mode_);
        }
        else
        {
            auto rand_ints = randint_like(a, INT_MAX, device(a.device()).dtype(kInt));
            int *p_r = rand_ints.data_ptr<int>();
            binaryK_kernel_sr_impl<scalar_t>(
                p_a, p_r, p_o, size, K_, P_, bias_, prng_bits_, is_signed,
                saturation_mode_, subnormals_mode_);
        } });

    return o;
}
