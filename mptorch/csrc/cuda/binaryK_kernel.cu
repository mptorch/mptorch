#include "../common/cast_binaryK.h"
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

    // The per-thread quantizer of one launch: everything a thread needs to
    // round one element, in a struct the kernel takes by value. A kernel
    // argument lands in the constant bank, so every thread's read of it is a
    // broadcast served from cache and it occupies no registers until a field
    // is used.
    //
    // `p` is a BinaryKParams built once on the host by make_binaryK_params
    // (rounding masks, clipping range, subnormal floor), so the casts called
    // here are the precomputed-parameter overloads; the (man_bits, exp_bits,
    // bias, ...) spellings would re-derive all of that on every element. That
    // overload is also the one that admits the float-arithmetic RNE fast
    // path on the device. The rounding mode is a template parameter so that
    // each instantiation carries exactly one cast body instead of a switch
    // over seven.
    //
    // T is the carrier the cast rounds in: double for a float64 tensor, and
    // float for float32, float16 and bfloat16 (carrier_t in bit_helper.h).
    // scalar() and scalar_sr() take one storage value and hand one back;
    // vec() rounds the lanes of one 16-byte vector through SIMDTraits, which
    // converts them to the carrier and back.
    template <typename scalar_t, RoundMode RM>
    struct BinaryKQuantizer
    {
        using T = carrier_t<scalar_t>;
        using word_t = typename FloatTraits<T>::word_t;

        BinaryKParamsT<T> p;
        bool is_signed;
        SubnormalsMode sub_mode;
        int prng_bits;

        __device__ __forceinline__ T eval(T x_f) const
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

        __device__ __forceinline__ T eval_sr(T x_f, word_t rv) const
        {
            return cast_binaryK_stochastic(x_f, rv, prng_bits, is_signed, sub_mode, p);
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

    // binaryK's (K, P) spelling to the cast's (man_bits, exp_bits): P counts
    // the significand bits including the implicit one, and an unsigned
    // format spends the sign's bit on the exponent field.
    template <class T>
    BinaryKParamsT<T> binaryK_params(int K, int P, int bias, bool is_signed,
                                     SaturationMode saturation_mode,
                                     SubnormalsMode subnormals_mode)
    {
        const int man_bits = P - 1;
        const int exp_bits = is_signed ? K - P : K - P + 1;
        return make_binaryK_params<T>(man_bits, exp_bits, bias, is_signed, saturation_mode,
                                      subnormals_mode == SubnormalsMode::EXTENDED_NORMALS);
    }

    // The six deterministic modes: build the params once, then launch the
    // quant_kernel_all instantiation of the requested mode. The grid covers
    // max(vector count, remainder count) threads because the kernel serves
    // the vector body and the scalar tail from the same thread index.
    template <typename scalar_t>
    void binaryK_kernel_impl(const scalar_t *a, scalar_t *o, int64_t size,
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

        const BinaryKParamsT<carrier_t<scalar_t>> p =
            binaryK_params<carrier_t<scalar_t>>(K, P, bias, is_signed, saturation_mode, subnormals_mode);

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

    // RoundMode::SR: one launch of quant_kernel_all_sr with a (seed, offset)
    // pair drawn from ATen's generator; the kernel derives each element's
    // random word from that pair and the element's own index.
    template <typename scalar_t>
    void binaryK_kernel_sr_impl(const scalar_t *a,
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

        const BinaryKParamsT<carrier_t<scalar_t>> p =
            binaryK_params<carrier_t<scalar_t>>(K, P, bias, is_signed, saturation_mode, subnormals_mode);
        auto quantizer =
            BinaryKQuantizer<scalar_t, RoundMode::SR>{p, is_signed, subnormals_mode, prng_bits};

        int64_t max_threads = std::max(vec_size, rem_size);
        int grid = grid_for(max_threads, BLOCK_SIZE);
        at::PhiloxCudaState rng_args = quant_rng_engine_inputs();
        quant_kernel_all_sr<<<grid, BLOCK_SIZE, 0, stream>>>(a, o, size, quantizer, rng_args);
    }

    // The body of both entry points: `size` elements of `a` rounded into
    // `o`, which is `a` itself for the in-place op. Both are contiguous and
    // start on a 16-byte boundary, which each entry point sees to in its own
    // way. The kernels read element i (or vector i) before they write it and
    // touch no other, so a == o is value-safe; that is also why their input
    // pointer carries no __restrict__ (utils.cuh).
    void binaryK_quantize_into(const Tensor &a, Tensor &o, int64_t K, int64_t P, int64_t bias,
                               int64_t prng_bits, bool is_signed, int64_t round_mode,
                               int64_t saturation_mode, int64_t subnormals_mode)
    {
        const int64_t size = a.numel(); // int would truncate past 2^31 elements
        if (size == 0)
            return;
        RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
        SubnormalsMode subnormals_mode_ = static_cast<SubnormalsMode>(subnormals_mode);
        SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);

        const int K_ = static_cast<int>(K);
        const int P_ = static_cast<int>(P);
        const int bias_ = static_cast<int>(bias);
        const int prng_bits_ = static_cast<int>(prng_bits);

        MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_quantize_cuda", [&]
                                        {
            const scalar_t *p_a = a.data_ptr<scalar_t>();
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
    }

} // namespace

Tensor binaryK_quantize_cuda(
    Tensor a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
    int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
    // data_ptr() walks storage linearly, so a non-contiguous input would be
    // read in the wrong order; mptorch/quant/ops.py calls .contiguous(), but
    // a direct torch.ops.mptorch.binaryK_quant call need not. The kernel's
    // 16-byte vector loads also fault on a contiguous view that does not
    // start on a 16-byte boundary, so vector_loadable (vector_load.h) copies
    // such a view as well.
    auto a_c = mptorch::vector_loadable(a);
    auto o = empty_like(a_c);
    binaryK_quantize_into(a_c, o, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                          subnormals_mode);
    return o;
}

// mptorch::binaryK_quant_: the same rounding written over `a`, with no output
// allocation. A copy would defeat it (the kernel would write the copy), so
// what the out-of-place op launders, a strided tensor or a view off a 16-byte
// boundary, this one refuses.
Tensor &binaryK_quantize_cuda_(
    Tensor &a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
    int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
    mptorch::check_vector_loadable_in_place(a, "binaryK_quant_", "binaryK_quant");
    binaryK_quantize_into(a, a, K, P, bias, prng_bits, is_signed, round_mode, saturation_mode,
                          subnormals_mode);
    return a;
}
