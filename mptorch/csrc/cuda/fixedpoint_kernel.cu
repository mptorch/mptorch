#include "../common/cast_fixedpoint.h"
#include "../common/dispatch.h"
#include "../quant_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty_like.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils.cuh"
#include "vector_load.h"

using namespace at;

namespace
{

    // The fixed-point twin of superfp_kernel.cu's SuperfpQuantizer: the
    // per-thread quantizer of one launch, taken by the kernel by value so it
    // rides in the constant bank. `p` is a FixedPointParams built once on the
    // host by make_fixedpoint_params (the step, the two ends of the range and
    // SR's shift). T is the carrier the cast rounds in, double for a float64
    // tensor and float for every other dtype, and the rounding mode is a
    // template parameter so each instantiation carries exactly one cast body.
    template <typename scalar_t, RoundMode RM>
    struct FixedPointQuantizer
    {
        using T = carrier_t<scalar_t>;
        using word_t = typename FloatTraits<T>::word_t;

        FixedPointParamsT<T> p;
        bool is_signed;
        int prng_bits;

        __device__ __forceinline__ T eval(T x_f) const
        {
            if (RM == RoundMode::RNE)
                return cast_fixedpoint_nearest_even(x_f, is_signed, p);
            if (RM == RoundMode::RNA)
                return cast_fixedpoint_nearest_away(x_f, is_signed, p);
            if (RM == RoundMode::RU)
                return cast_fixedpoint_up(x_f, is_signed, p);
            if (RM == RoundMode::RD)
                return cast_fixedpoint_down(x_f, is_signed, p);
            if (RM == RoundMode::RZ)
                return cast_fixedpoint_zero(x_f, is_signed, p);
            if (RM == RoundMode::RO)
                return cast_fixedpoint_odd(x_f, is_signed, p);
            return x_f;
        }

        __device__ __forceinline__ T eval_sr(T x_f, word_t rv) const
        {
            return cast_fixedpoint_stochastic(x_f, rv, prng_bits, is_signed, p);
        }

        __device__ __forceinline__ scalar_t scalar(scalar_t x) const
        {
            return static_cast<scalar_t>(eval(static_cast<T>(x)));
        }

        __device__ __forceinline__ scalar_t scalar_sr(scalar_t x, word_t rv) const
        {
            return static_cast<scalar_t>(eval_sr(static_cast<T>(x), rv));
        }

        __device__ __forceinline__ float4 vec(float4 x) const
        {
            return SIMDTraits<scalar_t>::process(x, [&](T val, int /*idx*/)
                                                 { return eval(val); });
        }
    };

    // The six deterministic modes: build the params once, then launch the
    // quant_kernel_all instantiation of the requested mode. The grid covers
    // max(vector count, remainder count) threads because the kernel serves
    // the vector body and the scalar tail from the same thread index.
    template <typename scalar_t>
    void fixedpoint_kernel_impl(const scalar_t *a, scalar_t *o, int64_t size, int wl, int fl,
                                bool is_signed, bool symmetric, RoundMode round_mode)
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

        const FixedPointParamsT<carrier_t<scalar_t>> p = make_fixedpoint_params<carrier_t<scalar_t>>(
            wl, fl, is_signed, symmetric);

        switch (round_mode)
        {
        case RoundMode::RNE:
            launch_kernels(FixedPointQuantizer<scalar_t, RoundMode::RNE>{p, is_signed, 0});
            break;
        case RoundMode::RNA:
            launch_kernels(FixedPointQuantizer<scalar_t, RoundMode::RNA>{p, is_signed, 0});
            break;
        case RoundMode::RU:
            launch_kernels(FixedPointQuantizer<scalar_t, RoundMode::RU>{p, is_signed, 0});
            break;
        case RoundMode::RD:
            launch_kernels(FixedPointQuantizer<scalar_t, RoundMode::RD>{p, is_signed, 0});
            break;
        case RoundMode::RZ:
            launch_kernels(FixedPointQuantizer<scalar_t, RoundMode::RZ>{p, is_signed, 0});
            break;
        default: // RO
            launch_kernels(FixedPointQuantizer<scalar_t, RoundMode::RO>{p, is_signed, 0});
            break;
        }
    }

    // RoundMode::SR: one launch of quant_kernel_all_sr with a (seed, offset)
    // pair drawn from ATen's generator; the kernel derives each element's
    // random word from that pair and the element's own index.
    template <typename scalar_t>
    void fixedpoint_kernel_sr_impl(const scalar_t *a, scalar_t *o, int64_t size, int wl, int fl,
                                   int prng_bits, bool is_signed, bool symmetric)
    {
        constexpr int BLOCK_SIZE = 256;
        int64_t elements_per_vec = SIMDTraits<scalar_t>::vec_elems;
        int64_t vec_size = size / elements_per_vec;
        int64_t rem_size = size % elements_per_vec;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        const FixedPointParamsT<carrier_t<scalar_t>> p = make_fixedpoint_params<carrier_t<scalar_t>>(
            wl, fl, is_signed, symmetric);
        auto quantizer = FixedPointQuantizer<scalar_t, RoundMode::SR>{p, is_signed, prng_bits};

        int64_t max_threads = std::max(vec_size, rem_size);
        int grid = grid_for(max_threads, BLOCK_SIZE);
        at::PhiloxCudaState rng_args = quant_rng_engine_inputs();
        quant_kernel_all_sr<<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, quantizer, rng_args);
    }

    // The body of both entry points, as binaryK_quantize_into is (binaryK_kernel.cu):
    // `o` is `a` itself for the in-place op.
    void fixedpoint_quantize_into(const Tensor &a, Tensor &o, int64_t wl, int64_t fl,
                                  int64_t prng_bits, bool is_signed, bool symmetric,
                                  int64_t round_mode)
    {
        const int64_t size = a.numel(); // int would truncate past 2^31 elements
        if (size == 0)
            return;
        RoundMode round_mode_ = static_cast<RoundMode>(round_mode);

        const int wl_ = static_cast<int>(wl);
        const int fl_ = static_cast<int>(fl);
        const int prng_bits_ = static_cast<int>(prng_bits);

        MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "fixedpoint_quantize_cuda", [&]
                                        {
            const scalar_t *p_a = a.data_ptr<scalar_t>();
            scalar_t *p_o = o.data_ptr<scalar_t>();

            if (round_mode_ != RoundMode::SR)
            {
                fixedpoint_kernel_impl<scalar_t>(
                    p_a, p_o, size, wl_, fl_, is_signed, symmetric, round_mode_);
            }
            else
            {
                fixedpoint_kernel_sr_impl<scalar_t>(
                    p_a, p_o, size, wl_, fl_, prng_bits_, is_signed, symmetric);
            } });
    }

} // namespace

Tensor fixedpoint_quantize_cuda(
    Tensor a, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode)
{
    // The input is made contiguous, and 16-byte aligned for the kernel's
    // vector loads, for the reasons given in binaryK_quantize_cuda.
    auto a_c = mptorch::vector_loadable(a);
    auto o = empty_like(a_c);
    fixedpoint_quantize_into(a_c, o, wl, fl, prng_bits, is_signed, symmetric, round_mode);
    return o;
}

// mptorch::fixedpoint_quant_: the same rounding written over `a`, which is
// checked rather than copied, as in binaryK_quantize_cuda_.
Tensor &fixedpoint_quantize_cuda_(
    Tensor &a, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode)
{
    mptorch::check_vector_loadable_in_place(a, "fixedpoint_quant_", "fixedpoint_quant");
    fixedpoint_quantize_into(a, a, wl, fl, prng_bits, is_signed, symmetric, round_mode);
    return a;
}
