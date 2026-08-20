#include "../common/cast_superfp.h"
#include "../quant_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <climits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils.cuh"

using namespace at;

namespace
{

    template <typename scalar_t, RoundMode RM>
    struct SuperfpQuantizer
    {
        int man_bits, exp_bits, normal_binades, bias;
        bool is_signed;
        SaturationMode sat_mode;
        int prng_bits;

        __device__ __forceinline__ float eval(float x_f) const
        {
            if (RM == RoundMode::RNE)
                return cast_superfp_nearest_even(x_f, man_bits, exp_bits, normal_binades, bias, is_signed, sat_mode);
            if (RM == RoundMode::RNA)
                return cast_superfp_nearest_away(x_f, man_bits, exp_bits, normal_binades, bias, is_signed, sat_mode);
            if (RM == RoundMode::RU)
                return cast_superfp_up(x_f, man_bits, exp_bits, normal_binades, bias, is_signed, sat_mode);
            if (RM == RoundMode::RD)
                return cast_superfp_down(x_f, man_bits, exp_bits, normal_binades, bias, is_signed, sat_mode);
            if (RM == RoundMode::RZ)
                return cast_superfp_zero(x_f, man_bits, exp_bits, normal_binades, bias, is_signed, sat_mode);
            if (RM == RoundMode::RO)
                return cast_superfp_odd(x_f, man_bits, exp_bits, normal_binades, bias, is_signed, sat_mode);
            return x_f;
        }

        __device__ __forceinline__ float eval_sr(float x_f, uint32_t rv) const
        {
            return cast_superfp_stochastic(x_f, rv, prng_bits, man_bits, exp_bits, normal_binades, bias,
                                           is_signed, sat_mode);
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
    void superfp_kernel_impl(scalar_t *__restrict__ a, scalar_t *o, int size,
                             int man_bits, int exp_bits, int normal_binades, int bias,
                             bool is_signed, RoundMode round_mode,
                             SaturationMode saturation_mode)
    {
        constexpr int BLOCK_SIZE = 256;
        int elements_per_vec = SIMDTraits<scalar_t>::vec_elems;
        int vec_size = size / elements_per_vec;
        int rem_size = size % elements_per_vec;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        auto launch_kernels = [&](auto quantizer)
        {
            int max_threads = std::max(vec_size, rem_size);
            int grid = std::max(1, (max_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
            quant_kernel_all<<<grid, BLOCK_SIZE, 0, stream>>>(a, nullptr, o, size, quantizer);
        };

        switch (round_mode)
        {
        case RoundMode::RNE:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RNE>{man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode, 0});
            break;
        case RoundMode::RNA:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RNA>{man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode, 0});
            break;
        case RoundMode::RU:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RU>{man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode, 0});
            break;
        case RoundMode::RD:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RD>{man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode, 0});
            break;
        case RoundMode::RZ:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RZ>{man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode, 0});
            break;
        default: // RO
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RO>{man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode, 0});
            break;
        }
    }

    template <typename scalar_t>
    void superfp_kernel_sr_impl(scalar_t *__restrict__ a, int *__restrict__ r, scalar_t *o, int size,
                                int man_bits, int exp_bits, int normal_binades, int bias, int prng_bits,
                                bool is_signed, SaturationMode saturation_mode)
    {
        constexpr int BLOCK_SIZE = 256;
        int elements_per_vec = SIMDTraits<scalar_t>::vec_elems;
        int vec_size = size / elements_per_vec;
        int rem_size = size % elements_per_vec;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        auto quantizer = SuperfpQuantizer<scalar_t, RoundMode::SR>{man_bits, exp_bits, normal_binades, bias, is_signed, saturation_mode, prng_bits};

        int max_threads = std::max(vec_size, rem_size);
        int grid = std::max(1, (max_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        quant_kernel_all<<<grid, BLOCK_SIZE, 0, stream>>>(a, r, o, size, quantizer);
    }

} // namespace

Tensor superfp_quantize_cuda(
    Tensor a, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias,
    int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode)
{
    auto o = empty_like(a);
    int size = a.numel();
    if (size == 0)
        return o;
    RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
    SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);

    const int man_bits_ = static_cast<int>(man_bits);
    const int exp_bits_ = static_cast<int>(exp_bits);
    const int normal_binades_ = static_cast<int>(normal_binades);
    const int bias_ = static_cast<int>(bias);
    const int prng_bits_ = static_cast<int>(prng_bits);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "superfp_quantize_cuda", [&]
                                    {
        scalar_t *p_a = a.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();

        if (round_mode_ != RoundMode::SR)
        {
            superfp_kernel_impl<scalar_t>(
                p_a, p_o, size, man_bits_, exp_bits_, normal_binades_, bias_, is_signed,
                round_mode_, saturation_mode_);
        }
        else
        {
            auto rand_ints = randint_like(a, INT_MAX, device(a.device()).dtype(kInt));
            int *p_r = rand_ints.data_ptr<int>();
            superfp_kernel_sr_impl<scalar_t>(
                p_a, p_r, p_o, size, man_bits_, exp_bits_, normal_binades_, bias_, prng_bits_,
                is_signed, saturation_mode_);
        } });

    return o;
}
