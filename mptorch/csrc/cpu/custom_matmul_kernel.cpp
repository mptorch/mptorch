#include "../common/gemm_policy.h"
#include "../common/modes.h"
#include "../quant_ops.h"
#include <ATen/ATen.h>
#include <algorithm>
#include <vector>

using namespace at;

namespace
{

  // Cache-tiled MxKxN GEMM: C = op(A) @ op(B), where op(X) = X.T if the
  // corresponding trans flag is set. A/B are always read through their own
  // (untransposed) storage layout -- trans_a/trans_b only change the index
  // arithmetic used to read them, no physical transpose is materialized.
  // Accumulator owns both the multiply and the accumulate step (via its Mac
  // policy -- see gemm_policy.h), so this kernel only ever calls
  // accumulate(a, b)/finalize() and never multiplies operands itself.
  template <typename scalar_t, class Accumulator>
  void matmul_cpu_kernel_impl(const scalar_t *A, const scalar_t *B, scalar_t *C,
                              int64_t M, int64_t K, int64_t N,
                              bool trans_a, bool trans_b,
                              Accumulator acc_proto)
  {
    constexpr int64_t TI = 32, TJ = 32, TK = 32;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i0 = 0; i0 < M; i0 += TI)
    {
      int64_t ti = std::min<int64_t>(TI, M - i0);
      for (int64_t j0 = 0; j0 < N; j0 += TJ)
      {
        int64_t tj = std::min<int64_t>(TJ, N - j0);
        std::vector<Accumulator> acc(static_cast<size_t>(ti * tj), acc_proto);

        for (int64_t k0 = 0; k0 < K; k0 += TK)
        {
          int64_t tk = std::min<int64_t>(TK, K - k0);
          for (int64_t i = 0; i < ti; ++i)
          {
            for (int64_t k = 0; k < tk; ++k)
            {
              float aVal = trans_a
                              ? static_cast<float>(A[(k0 + k) * M + (i0 + i)])
                              : static_cast<float>(A[(i0 + i) * K + (k0 + k)]);
              for (int64_t j = 0; j < tj; ++j)
              {
                float bVal = trans_b
                                ? static_cast<float>(B[(j0 + j) * K + (k0 + k)])
                                : static_cast<float>(B[(k0 + k) * N + (j0 + j)]);
                acc[static_cast<size_t>(i * tj + j)].accumulate(aVal, bVal);
              }
            }
          }
        }

        for (int64_t i = 0; i < ti; ++i)
          for (int64_t j = 0; j < tj; ++j)
            C[(i0 + i) * N + (j0 + j)] =
                static_cast<scalar_t>(acc[static_cast<size_t>(i * tj + j)].finalize());
      }
    }
  }

  void check_matmul_inputs(const Tensor &a, const Tensor &b, const char *op_name,
                           int64_t round_mode, int64_t accumulate_algorithm)
  {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, op_name, " expects 2D tensors, got ",
               a.dim(), "D and ", b.dim(), "D");
    TORCH_CHECK(static_cast<RoundMode>(round_mode) == RoundMode::RNE, op_name,
               ": only RoundMode.RNE is supported in this build");
    TORCH_CHECK(static_cast<AccumulateAlgorithm>(accumulate_algorithm) == AccumulateAlgorithm::NAIVE,
               op_name, ": only AccumulateAlgorithm.NAIVE is supported in this build");
  }

  void matmul_output_shape(const Tensor &a, const Tensor &b, bool trans_a, bool trans_b,
                           const char *op_name, int64_t &M, int64_t &K, int64_t &N)
  {
    M = trans_a ? a.size(1) : a.size(0);
    K = trans_a ? a.size(0) : a.size(1);
    int64_t K_b = trans_b ? b.size(1) : b.size(0);
    N = trans_b ? b.size(0) : b.size(1);
    TORCH_CHECK(K == K_b, op_name, ": inner dimensions must match (got ", K, " vs ", K_b, ")");
  }

} // namespace

Tensor binaryK_matmul_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
  check_matmul_inputs(a, b, "custom_matmul_binaryK", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK", M, K, N);

  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return c;

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);

  int mul_man_bits = static_cast<int>(mul_P - 1);
  int mul_exp_bits = static_cast<int>(mul_is_signed ? mul_K - mul_P : mul_K - mul_P + 1);
  int acc_man_bits = static_cast<int>(acc_P - 1);
  int acc_exp_bits = static_cast<int>(acc_is_signed ? acc_K - acc_P : acc_K - acc_P + 1);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "binaryK_matmul_cpu", [&]
                                  {
    BinaryKMultiplier mul{mul_man_bits, mul_exp_bits, static_cast<int>(mul_bias), mul_is_signed, sat, sub};
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    if (accumulate_quant)
    {
      BinaryKAdder add{acc_man_bits, acc_exp_bits, static_cast<int>(acc_bias), acc_is_signed, sat, sub};
      using Mac = SplitMac<BinaryKMultiplier, BinaryKAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
    }
    else
    {
      using Mac = SplitMac<BinaryKMultiplier, IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
    } });

  return c;
}

Tensor superfp_matmul_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
    int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode)
{
  check_matmul_inputs(a, b, "custom_matmul_superfp", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp", M, K, N);

  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return c;

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "superfp_matmul_cpu", [&]
                                  {
    SuperfpMultiplier mul{static_cast<int>(mul_man_bits), static_cast<int>(mul_exp_bits),
                          static_cast<int>(mul_normal_binades), static_cast<int>(mul_bias), mul_is_signed, sat};
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    if (accumulate_quant)
    {
      SuperfpAdder add{static_cast<int>(acc_man_bits), static_cast<int>(acc_exp_bits),
                       static_cast<int>(acc_normal_binades), static_cast<int>(acc_bias), acc_is_signed, sat};
      using Mac = SplitMac<SuperfpMultiplier, SuperfpAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
    }
    else
    {
      using Mac = SplitMac<SuperfpMultiplier, IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
    } });

  return c;
}

Tensor binaryK_matmul_fma_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
  check_matmul_inputs(a, b, "custom_matmul_binaryK_fma", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_fma", M, K, N);

  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return c;

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
  SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);

  int fma_man_bits = static_cast<int>(fma_P - 1);
  int fma_exp_bits = static_cast<int>(fma_is_signed ? fma_K - fma_P : fma_K - fma_P + 1);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "binaryK_matmul_fma_cpu", [&]
                                  {
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    if (fma_quant)
    {
      BinaryKAdder add{fma_man_bits, fma_exp_bits, static_cast<int>(fma_bias), fma_is_signed, sat, sub};
      using Mac = FusedMac<BinaryKAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
    }
    else
    {
      using Mac = FusedMac<IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
    } });

  return c;
}

Tensor superfp_matmul_fma_cpu(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades,
    int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode)
{
  check_matmul_inputs(a, b, "custom_matmul_superfp_fma", round_mode, accumulate_algorithm);

  int64_t M, K, N;
  matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_fma", M, K, N);

  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0)
    return c;

  SaturationMode sat = static_cast<SaturationMode>(saturation_mode);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "superfp_matmul_fma_cpu", [&]
                                  {
    const scalar_t *p_a = a_c.data_ptr<scalar_t>();
    const scalar_t *p_b = b_c.data_ptr<scalar_t>();
    scalar_t *p_c = c.data_ptr<scalar_t>();

    if (fma_quant)
    {
      SuperfpAdder add{static_cast<int>(fma_man_bits), static_cast<int>(fma_exp_bits),
                       static_cast<int>(fma_normal_binades), static_cast<int>(fma_bias), fma_is_signed, sat};
      using Mac = FusedMac<SuperfpAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
    }
    else
    {
      using Mac = FusedMac<IdentityAdder>;
      NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
      matmul_cpu_kernel_impl<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
    } });

  return c;
}
