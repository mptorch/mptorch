// The CUDA kernel of binaryK_quant_scaled: a binaryK format, each element
// multiplied by its scale, rounded and divided by the scale again, in one
// pass (cuda/scaled_quant.cuh; common/scaled_quant_host.h states the
// operation). A file of its own, so that binaryK_kernel.cu's objects are what
// they were.

#include "../common/cast_binaryK.h"
#include "../common/dispatch.h"
#include "../common/scaled_quant_host.h"
#include "../common/stochastic_host.h"
#include "../quant_ops.h"
#include "scaled_quant.cuh"
#include "vector_load.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty_like.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace at;
using namespace mptorch_cuda_scaled;

namespace
{

    // binaryK_kernel.cu's BinaryKQuantizer, with the multiply before the
    // cast and the divide after.
    template <typename scalar_t, RoundMode RM>
    struct BinaryKScaled
    {
        using T = carrier_t<scalar_t>;
        using word_t = typename FloatTraits<T>::word_t;

        BinaryKParamsT<T> p;
        bool is_signed;
        SubnormalsMode sub_mode;
        int prng_bits;

        __device__ __forceinline__ T eval(T x) const
        {
            if (RM == RoundMode::RNE)
                return cast_binaryK_nearest_even(x, is_signed, sub_mode, p);
            if (RM == RoundMode::RNA)
                return cast_binaryK_nearest_away(x, is_signed, sub_mode, p);
            if (RM == RoundMode::RU)
                return cast_binaryK_up(x, is_signed, sub_mode, p);
            if (RM == RoundMode::RD)
                return cast_binaryK_down(x, is_signed, sub_mode, p);
            if (RM == RoundMode::RZ)
                return cast_binaryK_zero(x, is_signed, sub_mode, p);
            return cast_binaryK_odd(x, is_signed, sub_mode, p);
        }

        __device__ __forceinline__ T scaled(T x, T s) const { return div_rn(eval(mul_rn(x, s)), s); }

        __device__ __forceinline__ T scaled_sr(T x, T s, word_t rv) const
        {
            return div_rn(cast_binaryK_stochastic(mul_rn(x, s), rv, prng_bits, is_signed, sub_mode, p), s);
        }
    };

    template <typename scalar_t>
    void binaryK_scaled_impl(const scalar_t *a, scalar_t *o, int64_t size, const carrier_t<scalar_t> *s,
                             const mptorch::ScaleLayout &l, int K, int P, int bias, int prng_bits,
                             bool is_signed, RoundMode rm, SaturationMode sat, SubnormalsMode sub)
    {
        using T = carrier_t<scalar_t>;
        const char *op = "binaryK_quant_scaled";
        const BinaryKParamsT<T> p = make_binaryK_params<T>(
            P - 1, is_signed ? K - P : K - P + 1, bias, is_signed, sat,
            sub == SubnormalsMode::EXTENDED_NORMALS);
        switch (rm)
        {
        case RoundMode::RNE:
            launch_scaled(a, o, size, s, l, BinaryKScaled<scalar_t, RoundMode::RNE>{p, is_signed, sub, 0}, op);
            break;
        case RoundMode::RNA:
            launch_scaled(a, o, size, s, l, BinaryKScaled<scalar_t, RoundMode::RNA>{p, is_signed, sub, 0}, op);
            break;
        case RoundMode::RU:
            launch_scaled(a, o, size, s, l, BinaryKScaled<scalar_t, RoundMode::RU>{p, is_signed, sub, 0}, op);
            break;
        case RoundMode::RD:
            launch_scaled(a, o, size, s, l, BinaryKScaled<scalar_t, RoundMode::RD>{p, is_signed, sub, 0}, op);
            break;
        case RoundMode::RZ:
            launch_scaled(a, o, size, s, l, BinaryKScaled<scalar_t, RoundMode::RZ>{p, is_signed, sub, 0}, op);
            break;
        case RoundMode::RO:
            launch_scaled(a, o, size, s, l, BinaryKScaled<scalar_t, RoundMode::RO>{p, is_signed, sub, 0}, op);
            break;
        default: // SR
            launch_scaled_sr(a, o, size, s, l,
                             BinaryKScaled<scalar_t, RoundMode::SR>{p, is_signed, sub, prng_bits}, op);
            break;
        }
    }

} // namespace

// binaryK_quant's format checks first, then the scale operands'
// (common/scaled_quant_host.h). vector_loadable copies a strided input, or a
// view off a 16-byte boundary, as binaryK_quant's entry point does.
Tensor binaryK_quantize_scaled_cuda(Tensor a, Tensor scale, int64_t block_size, int64_t K, int64_t P,
                                    int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode,
                                    int64_t saturation_mode, int64_t subnormals_mode)
{
    const char *op = "binaryK_quant_scaled";
    mptorch::check_quantizer_format(
        mptorch::binaryK_format(K, P, bias, is_signed, saturation_mode, subnormals_mode),
        mptorch::carrier_bits(a.scalar_type() == at::kDouble), prng_bits, round_mode, op);
    mptorch::check_scaled_operands(a, scale, block_size, op);
    auto a_c = mptorch::vector_loadable(a);
    auto s_c = scale.contiguous();
    auto o = empty_like(a_c);
    const int64_t size = a_c.numel();
    if (size == 0)
        return o;
    // SR_B and SR_C run as SR, with the variant recorded in prng_bits
    // (common/stochastic_host.h).
    int variant;
    round_mode = mptorch::sr_canonical(round_mode, variant);
    if (variant >= 0)
        prng_bits = mptorch::sr_encode(prng_bits, variant, op);
    const mptorch::ScaleLayout l = mptorch::scale_layout(a_c, block_size);
    MPTORCH_DISPATCH_QUANT_TYPES(a_c.scalar_type(), "binaryK_quant_scaled", [&]
                                 { binaryK_scaled_impl<scalar_t>(
                                       a_c.data_ptr<scalar_t>(), o.data_ptr<scalar_t>(), size,
                                       s_c.data_ptr<carrier_t<scalar_t>>(), l, static_cast<int>(K),
                                       static_cast<int>(P), static_cast<int>(bias),
                                       static_cast<int>(prng_bits), is_signed,
                                       static_cast<RoundMode>(round_mode),
                                       static_cast<SaturationMode>(saturation_mode),
                                       static_cast<SubnormalsMode>(subnormals_mode)); });
    return o;
}
