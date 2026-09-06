#include "../common/cast_superfp.h"
#include "../common/dispatch.h"
#include "../quant_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty_like.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils.cuh"

using namespace at;

namespace
{

    // The superfp twin of binaryK_kernel.cu's BinaryKQuantizer: `p` is a
    // SuperfpParams built once on the host, so the casts taken here are the
    // precomputed-parameter overloads rather than the (man_bits, exp_bits,
    // normal_binades, bias, ...) ones this used to call, which re-derived the
    // region cutoffs, the rounding masks and the clipping range on every
    // element. See that struct, and finding E3 in dev/gemm_perf_audit.md.
    template <typename scalar_t, RoundMode RM>
    struct SuperfpQuantizer
    {
        SuperfpParams p;
        bool is_signed;
        int prng_bits;

        __device__ __forceinline__ float eval(float x_f) const
        {
            if (RM == RoundMode::RNE)
                return cast_superfp_nearest_even(x_f, is_signed, p);
            if (RM == RoundMode::RNA)
                return cast_superfp_nearest_away(x_f, is_signed, p);
            if (RM == RoundMode::RU)
                return cast_superfp_up(x_f, is_signed, p);
            if (RM == RoundMode::RD)
                return cast_superfp_down(x_f, is_signed, p);
            if (RM == RoundMode::RZ)
                return cast_superfp_zero(x_f, is_signed, p);
            if (RM == RoundMode::RO)
                return cast_superfp_odd(x_f, is_signed, p);
            return x_f;
        }

        __device__ __forceinline__ float eval_sr(float x_f, uint32_t rv) const
        {
            return cast_superfp_stochastic(x_f, rv, prng_bits, is_signed, p);
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
    };

    template <typename scalar_t>
    void superfp_kernel_impl(const scalar_t *__restrict__ a, scalar_t *o, int64_t size,
                             int man_bits, int exp_bits, int normal_binades, int bias,
                             bool is_signed, RoundMode round_mode,
                             SaturationMode saturation_mode)
    {
        constexpr int BLOCK_SIZE = 256;
        int64_t elements_per_vec = SIMDTraits<scalar_t>::vec_elems;
        int64_t vec_size = size / elements_per_vec;
        int64_t rem_size = size % elements_per_vec;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        auto launch_kernels = [&](auto quantizer)
        {
            int64_t max_threads = std::max(vec_size, rem_size);
            int grid = grid_for(max_threads, BLOCK_SIZE);
            quant_kernel_all<<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, quantizer);
        };

        const SuperfpParams p = make_superfp_params(man_bits, exp_bits, normal_binades,
                                                    bias, saturation_mode);

        switch (round_mode)
        {
        case RoundMode::RNE:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RNE>{p, is_signed, 0});
            break;
        case RoundMode::RNA:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RNA>{p, is_signed, 0});
            break;
        case RoundMode::RU:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RU>{p, is_signed, 0});
            break;
        case RoundMode::RD:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RD>{p, is_signed, 0});
            break;
        case RoundMode::RZ:
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RZ>{p, is_signed, 0});
            break;
        default: // RO
            launch_kernels(SuperfpQuantizer<scalar_t, RoundMode::RO>{p, is_signed, 0});
            break;
        }
    }

    template <typename scalar_t>
    void superfp_kernel_sr_impl(const scalar_t *__restrict__ a,
                                scalar_t *o, int64_t size,
                                int man_bits, int exp_bits, int normal_binades, int bias, int prng_bits,
                                bool is_signed, SaturationMode saturation_mode)
    {
        constexpr int BLOCK_SIZE = 256;
        int64_t elements_per_vec = SIMDTraits<scalar_t>::vec_elems;
        int64_t vec_size = size / elements_per_vec;
        int64_t rem_size = size % elements_per_vec;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        const SuperfpParams p = make_superfp_params(man_bits, exp_bits, normal_binades,
                                                    bias, saturation_mode);
        auto quantizer = SuperfpQuantizer<scalar_t, RoundMode::SR>{p, is_signed, prng_bits};

        int64_t max_threads = std::max(vec_size, rem_size);
        int grid = grid_for(max_threads, BLOCK_SIZE);
        at::PhiloxCudaState rng_args = quant_rng_engine_inputs();
        quant_kernel_all_sr<<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, quantizer, rng_args);
    }

} // namespace

Tensor superfp_quantize_cuda(
    Tensor a, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias,
    int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode)
{
    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a);

    // see binaryK_quantize_cuda for why the input is made contiguous here
    auto a_c = a.contiguous();
    auto o = empty_like(a_c);
    const int64_t size = a_c.numel(); // int would truncate past 2^31 elements
    if (size == 0)
        return mptorch::widen_float64(o, widen_f64);
    RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
    SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);

    const int man_bits_ = static_cast<int>(man_bits);
    const int exp_bits_ = static_cast<int>(exp_bits);
    const int normal_binades_ = static_cast<int>(normal_binades);
    const int bias_ = static_cast<int>(bias);
    const int prng_bits_ = static_cast<int>(prng_bits);

    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "superfp_quantize_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();

        if (round_mode_ != RoundMode::SR)
        {
            superfp_kernel_impl<scalar_t>(
                p_a, p_o, size, man_bits_, exp_bits_, normal_binades_, bias_, is_signed,
                round_mode_, saturation_mode_);
        }
        else
        {
            superfp_kernel_sr_impl<scalar_t>(
                p_a, p_o, size, man_bits_, exp_bits_, normal_binades_, bias_, prng_bits_,
                is_signed, saturation_mode_);
        } });

    return mptorch::widen_float64(o, widen_f64);
}
