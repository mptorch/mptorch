#include "custom_matmul_kernel.cuh"
#include "../quant_ops.h"
#include <ATen/ops/empty.h>
#include <type_traits>

using namespace at;
using namespace mptorch::gemm_cuda;

Tensor binaryK_matmul_fma_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode,
    int64_t fma_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_binaryK_fma", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_fma", M, K, N);

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
    SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    int fma_man_bits = static_cast<int>(fma_P - 1);
    int fma_exp_bits = static_cast<int>(fma_is_signed ? fma_K - fma_P : fma_K - fma_P + 1);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    const mptorch::GemmDtype dt = mptorch::gemm_dtype_of(a_c, b_c, "custom_matmul_binaryK_fma");
    const void *p_a = a_c.data_ptr();
    const void *p_b = b_c.data_ptr();
    void *p_c = c.data_ptr();

    mptorch::dispatch_round_mode(rm, [&](auto rm_c)
    {
        constexpr RoundMode RM = decltype(rm_c)::value;

        if (fma_quant)
        {
            BinaryKAdderT<RM> add{fma_man_bits, fma_exp_bits, static_cast<int>(fma_bias), fma_is_signed, sat, sub,
                                  static_cast<int>(fma_prng_bits)};
            using Mac = FusedMac<BinaryKAdderT<RM>>;
            NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
            launch_custom_matmul(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        }
        else
        {
            using Mac = FusedMac<IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
            launch_custom_matmul(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        }
    });

    return mptorch::widen_float64(c, widen_f64);
}

// Spatially-varying (per-output-element) mixed-format binaryK FMA GEMM: same
// FusedMac arithmetic as binaryK_matmul_fma_cuda (one rounding per K-step),
// but the FMA format for each output element C[i, j] is picked from a
// palette of up to MAX_GEMM_FORMATS entries by prec_idx[i, j] (see
// gemm_policy.h's FormatPalette). fma_K/fma_P/fma_bias are per-palette-entry
// lists (all the same length); round_mode/saturation/sign/prng_bits are
// shared across the palette. fma_quant=false is rejected: FusedMac<
// IdentityAdder> carries no format, so a palette of it would make prec_idx a
// no-op -- use custom_matmul_binaryK_fma for the unquantized fused step.
at::Tensor binaryK_matmul_fma_mixed_cuda(
    at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b,
    bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias,
    bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t subnormals_mode, int64_t fma_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_binaryK_fma_mixed", round_mode, accumulate_algorithm);
    TORCH_CHECK(fma_quant, "custom_matmul_binaryK_fma_mixed: fma_quant=false has no per-element "
                           "format to vary; use custom_matmul_binaryK_fma");

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_fma_mixed", M, K, N);

    int64_t n_fmt = static_cast<int64_t>(fma_K.size());
    check_palette_lengths(n_fmt, "custom_matmul_binaryK_fma_mixed",
                          {static_cast<int64_t>(fma_P.size()), static_cast<int64_t>(fma_bias.size())});

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
    resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_binaryK_fma_mixed", n_fmt, pidx, sr, sc);
    const int32_t *p_idx = pidx.data_ptr<int32_t>();

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    const mptorch::GemmDtype dt = mptorch::gemm_dtype_of(a_c, b_c, "custom_matmul_binaryK_fma_mixed");
    const void *p_a = a_c.data_ptr();
    const void *p_b = b_c.data_ptr();
    void *p_c = c.data_ptr();

    mptorch::dispatch_round_mode(rm, [&](auto rm_c)
    {
        constexpr RoundMode RM = decltype(rm_c)::value;

        using Mac = FusedMac<BinaryKAdderT<RM>>;
        static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
        FormatPalette<Mac> pal;
        pal.n = static_cast<int>(n_fmt);
        for (int64_t i = 0; i < n_fmt; ++i)
        {
            int mb = static_cast<int>(fma_P[i] - 1);
            int eb = static_cast<int>(fma_is_signed ? fma_K[i] - fma_P[i] : fma_K[i] - fma_P[i] + 1);
            BinaryKAdderT<RM> add{mb, eb, static_cast<int>(fma_bias[i]), fma_is_signed, sat, sub,
                                  static_cast<int>(fma_prng_bits)};
            pal.slots[i] = Mac{add};
        }
        NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
        launch_custom_matmul<true>(p_a, p_b, p_c, dt, M, K, N, trans_a, trans_b, acc_proto,
                                   use_rng, rng_args, pal, p_idx, sr, sc);
    });

    return mptorch::widen_float64(c, widen_f64);
}
