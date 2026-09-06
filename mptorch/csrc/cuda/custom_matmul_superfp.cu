#include "custom_matmul_kernel.cuh"
#include "../quant_ops.h"
#include <ATen/ops/empty.h>
#include <type_traits>

using namespace at;
using namespace mptorch::gemm_cuda;

Tensor superfp_matmul_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
    int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_superfp", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp", M, K, N);

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(2 * static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    const mptorch::GemmDtype dt = mptorch::gemm_dtype_of(a_c, b_c, "custom_matmul_superfp");
    const void *p_a = a_c.data_ptr();
    const void *p_b = b_c.data_ptr();
    void *p_c = c.data_ptr();

    mptorch::dispatch_round_mode(rm, [&](auto rm_c)
    {
        constexpr RoundMode RM = decltype(rm_c)::value;

        SuperfpMultiplier<RM> mul{static_cast<int>(mul_man_bits), static_cast<int>(mul_exp_bits),
                                  static_cast<int>(mul_normal_binades), static_cast<int>(mul_bias), mul_is_signed, sat,
                                  static_cast<int>(mul_prng_bits)};

        if (accumulate_quant)
        {
            SuperfpAdder<RM> add{static_cast<int>(acc_man_bits), static_cast<int>(acc_exp_bits),
                                 static_cast<int>(acc_normal_binades), static_cast<int>(acc_bias), acc_is_signed, sat,
                                 static_cast<int>(acc_prng_bits)};
            using Mac = SplitMac<SuperfpMultiplier<RM>, SuperfpAdder<RM>>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
            launch_custom_matmul(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        }
        else
        {
            using Mac = SplitMac<SuperfpMultiplier<RM>, IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
            launch_custom_matmul(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        }
    });

    return mptorch::widen_float64(c, widen_f64);
}

// superfp analogue of binaryK_matmul_mixed_cuda -- see its comment.
at::Tensor superfp_matmul_mixed_cuda(
    at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b,
    c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades,
    c10::IntArrayRef mul_bias, bool mul_is_signed,
    bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits,
    c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_superfp_mixed", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_mixed", M, K, N);

    int64_t n_fmt = static_cast<int64_t>(mul_man_bits.size());
    check_palette_lengths(n_fmt, "custom_matmul_superfp_mixed",
                          {static_cast<int64_t>(mul_exp_bits.size()),
                           static_cast<int64_t>(mul_normal_binades.size()),
                           static_cast<int64_t>(mul_bias.size()),
                           static_cast<int64_t>(acc_man_bits.size()),
                           static_cast<int64_t>(acc_exp_bits.size()),
                           static_cast<int64_t>(acc_normal_binades.size()),
                           static_cast<int64_t>(acc_bias.size())});

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    at::Tensor pidx;
    int64_t sr = 0, sc = 0;
    resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_superfp_mixed", n_fmt, pidx, sr, sc);
    const int32_t *p_idx = pidx.data_ptr<int32_t>();

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(2 * static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    const mptorch::GemmDtype dt = mptorch::gemm_dtype_of(a_c, b_c, "custom_matmul_superfp_mixed");
    const void *p_a = a_c.data_ptr();
    const void *p_b = b_c.data_ptr();
    void *p_c = c.data_ptr();

    mptorch::dispatch_round_mode(rm, [&](auto rm_c)
    {
        constexpr RoundMode RM = decltype(rm_c)::value;

        if (accumulate_quant)
        {
            // Two full superfp policies per palette slot is the one Mac in this
            // file that runs out of registers: at 100 it gets 2 blocks resident
            // per SM where its single-format twin gets 5. The Lean spelling
            // rebuilds the seven fast-path floats where they are used instead
            // of carrying them, which buys back a block for the same values.
            // Only this branch: the IdentityAdder one below already fits, and
            // paying the arithmetic there would be a straight loss. See
            // dev/gemm_perf_audit.md (finding G10).
            using Mac = SplitMac<SuperfpMultiplierLean<RM>, SuperfpAdderLean<RM>>;
            static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
            FormatPalette<Mac> pal;
            pal.n = static_cast<int>(n_fmt);
            for (int64_t i = 0; i < n_fmt; ++i)
            {
                SuperfpMultiplierLean<RM> mul{static_cast<int>(mul_man_bits[i]), static_cast<int>(mul_exp_bits[i]),
                                              static_cast<int>(mul_normal_binades[i]), static_cast<int>(mul_bias[i]),
                                              mul_is_signed, sat, static_cast<int>(mul_prng_bits)};
                SuperfpAdderLean<RM> add{static_cast<int>(acc_man_bits[i]), static_cast<int>(acc_exp_bits[i]),
                                         static_cast<int>(acc_normal_binades[i]), static_cast<int>(acc_bias[i]),
                                         acc_is_signed, sat, static_cast<int>(acc_prng_bits)};
                pal.slots[i] = Mac{mul, add};
            }
            NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
            launch_custom_matmul<true>(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto,
                                       use_rng, rng_args, pal, p_idx, sr, sc);
        }
        else
        {
            using Mac = SplitMac<SuperfpMultiplier<RM>, IdentityAdder>;
            static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
            FormatPalette<Mac> pal;
            pal.n = static_cast<int>(n_fmt);
            for (int64_t i = 0; i < n_fmt; ++i)
            {
                SuperfpMultiplier<RM> mul{static_cast<int>(mul_man_bits[i]), static_cast<int>(mul_exp_bits[i]),
                                          static_cast<int>(mul_normal_binades[i]), static_cast<int>(mul_bias[i]),
                                          mul_is_signed, sat, static_cast<int>(mul_prng_bits)};
                pal.slots[i] = Mac{mul, IdentityAdder{}};
            }
            NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
            launch_custom_matmul<true>(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto,
                                       use_rng, rng_args, pal, p_idx, sr, sc);
        }
    });

    return mptorch::widen_float64(c, widen_f64);
}
