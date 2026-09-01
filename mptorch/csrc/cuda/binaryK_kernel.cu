#include "../common/cast_binaryK.h"
#include "../quant_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils.cuh"

using namespace at;

namespace
{

    // `p` is a BinaryKParams built once on the host, so the casts taken here
    // are the precomputed-parameter overloads rather than the (man_bits,
    // exp_bits, bias, ...) ones this used to call. The latter re-derive the
    // rounding masks and the clipping range from those integers on *every
    // element*; make_binaryK_params does it once per tensor. It is also what
    // puts RNE on cast_binaryK_rne_fast, since that float-arithmetic path is
    // gated inside the BinaryKParams overload -- the two are one edit.
    // See finding E3 in dev/gemm_perf_audit.md, and C4 for the CPU twin.
    //
    // The struct is passed by value as a kernel argument, so it lands in
    // constant memory and every thread's read of it is broadcast and cached.
    template <typename scalar_t, RoundMode RM>
    struct BinaryKQuantizer
    {
        BinaryKParams p;
        bool is_signed;
        SubnormalsMode sub_mode;
        int prng_bits;

        __device__ __forceinline__ float eval(float x_f) const
        {
            if (RM == RoundMode::RNE)
                return cast_binaryK_nearest_even(x_f, is_signed, sub_mode, p);
            if (RM == RoundMode::RNA)
                return cast_binaryK_nearest_away(x_f, is_signed, sub_mode, p);
            if (RM == RoundMode::RU)
                return cast_binaryK_up(x_f, is_signed, sub_mode, p);
            if (RM == RoundMode::RD)
                return cast_binaryK_down(x_f, is_signed, sub_mode, p);
            if (RM == RoundMode::RZ)
                return cast_binaryK_zero(x_f, is_signed, sub_mode, p);
            if (RM == RoundMode::RO)
                return cast_binaryK_odd(x_f, is_signed, sub_mode, p);
            return x_f;
        }

        __device__ __forceinline__ float eval_sr(float x_f, uint32_t rv) const
        {
            return cast_binaryK_stochastic(x_f, rv, prng_bits, is_signed, sub_mode, p);
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

    BinaryKParams binaryK_params(int K, int P, int bias, bool is_signed,
                                 SaturationMode saturation_mode,
                                 SubnormalsMode subnormals_mode)
    {
        const int man_bits = P - 1;
        const int exp_bits = is_signed ? K - P : K - P + 1;
        return make_binaryK_params(man_bits, exp_bits, bias, saturation_mode,
                                   subnormals_mode == SubnormalsMode::EXTENDED_NORMALS);
    }

    template <typename scalar_t>
    void binaryK_kernel_impl(const scalar_t *__restrict__ a, scalar_t *o, int64_t size,
                             int K, int P, int bias, bool is_signed,
                             RoundMode round_mode,
                             SaturationMode saturation_mode,
                             SubnormalsMode subnormals_mode)
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

        const BinaryKParams p =
            binaryK_params(K, P, bias, is_signed, saturation_mode, subnormals_mode);

        switch (round_mode)
        {
        case RoundMode::RNE:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RNE>{p, is_signed, subnormals_mode, 0});
            break;
        case RoundMode::RNA:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RNA>{p, is_signed, subnormals_mode, 0});
            break;
        case RoundMode::RU:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RU>{p, is_signed, subnormals_mode, 0});
            break;
        case RoundMode::RD:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RD>{p, is_signed, subnormals_mode, 0});
            break;
        case RoundMode::RZ:
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RZ>{p, is_signed, subnormals_mode, 0});
            break;
        default: // RO
            launch_kernels(BinaryKQuantizer<scalar_t, RoundMode::RO>{p, is_signed, subnormals_mode, 0});
            break;
        }
    }

    template <typename scalar_t>
    void binaryK_kernel_sr_impl(const scalar_t *__restrict__ a,
                                scalar_t *o, int64_t size,
                                int K, int P, int bias, int prng_bits, bool is_signed,
                                SaturationMode saturation_mode,
                                SubnormalsMode subnormals_mode)
    {
        constexpr int BLOCK_SIZE = 256;
        int64_t elements_per_vec = SIMDTraits<scalar_t>::vec_elems;
        int64_t vec_size = size / elements_per_vec;
        int64_t rem_size = size % elements_per_vec;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        const BinaryKParams p =
            binaryK_params(K, P, bias, is_signed, saturation_mode, subnormals_mode);
        auto quantizer =
            BinaryKQuantizer<scalar_t, RoundMode::SR>{p, is_signed, subnormals_mode, prng_bits};

        int64_t max_threads = std::max(vec_size, rem_size);
        int grid = grid_for(max_threads, BLOCK_SIZE);
        at::PhiloxCudaState rng_args = quant_rng_engine_inputs();
        quant_kernel_all_sr<<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, quantizer, rng_args);
    }

} // namespace

Tensor binaryK_quantize_cuda(
    Tensor a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
    int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
    // data_ptr() walks storage linearly, so a non-contiguous input would be
    // read in the wrong order. mptorch/quant/ops.py already calls .contiguous(),
    // but a direct torch.ops.mptorch.binaryK_quant call need not.
    auto a_c = a.contiguous();
    auto o = empty_like(a_c);
    const int64_t size = a_c.numel(); // int would truncate past 2^31 elements
    if (size == 0)
        return o;
    RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
    SubnormalsMode subnormals_mode_ = static_cast<SubnormalsMode>(subnormals_mode);
    SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);

    const int K_ = static_cast<int>(K);
    const int P_ = static_cast<int>(P);
    const int bias_ = static_cast<int>(bias);
    const int prng_bits_ = static_cast<int>(prng_bits);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a_c.scalar_type(), "binaryK_quantize_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        scalar_t *p_o = o.data_ptr<scalar_t>();

        if (round_mode_ != RoundMode::SR)
        {
            binaryK_kernel_impl<scalar_t>(
                p_a, p_o, size, K_, P_, bias_, is_signed,
                round_mode_, saturation_mode_, subnormals_mode_);
        }
        else
        {
            binaryK_kernel_sr_impl<scalar_t>(
                p_a, p_o, size, K_, P_, bias_, prng_bits_, is_signed,
                saturation_mode_, subnormals_mode_);
        } });

    return o;
}
