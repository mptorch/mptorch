// The Meta dispatch-key registrations of every op, and the Meta kernels
// of the twelve that are not GEMM-shaped: the quantizers,
// narrow_float64, the block elementwise ops and _carrier_odd. The
// GEMM-shaped ones are meta_gemm.cpp's. A Meta kernel is what the meta
// device, FakeTensorMode, torch.export and torch.compile run in an op's
// place: the CPU op's checks, in its order and with its messages, and a
// result of the shape, dtype and layout the CPU op allocates, with
// nothing computed and no RNG state drawn (common/meta_host.h has the
// rules and the exceptions).

#include "common/meta_host.h"
#include "common/stochastic_cast_host.h"
#include "meta_ops.h"
#include <torch/library.h>

using namespace mptorch::meta;

namespace
{
  // The CPU quantizers refuse another dtype through AT_DISPATCH, whose
  // c10::NotImplementedError would make FakeTensorMode take the op for one
  // with no Meta kernel; this is a RuntimeError naming the dtype instead.
  void check_quant_dtype(const Tensor &a, const char *op)
  {
    const at::ScalarType st = a.scalar_type();
    TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16, op,
                ": expected a float32, float64, float16 or bfloat16 tensor, got ", st);
  }

  // An out-of-place quantizer's result: empty_like of the contiguous input,
  // as on every backend.
  Tensor quantized_like(const Tensor &a, const char *op)
  {
    check_quant_dtype(a, op);
    return at::empty_like(a.contiguous());
  }

  // An in-place quantizer's contiguity check, which the CPU makes before
  // anything else.
  void check_in_place(const Tensor &a, const char *op, const char *out_of_place)
  {
    TORCH_CHECK(a.is_contiguous(), op, " writes its argument in place and needs a contiguous "
                "tensor, got strides ", a.sym_strides(), " for sizes ", a.sym_sizes(), ": use ",
                out_of_place, ", which copies a strided input");
  }
  // The quantizers' prng_bits refusal under a stochastic mode
  // (common/stochastic_host.h), named as the CPU names it.
  void check_quant_prng(int64_t prng_bits, int64_t round_mode, const char *op)
  {
    int variant;
    mptorch::sr_canonical(round_mode, variant);
    if (variant >= 0)
      mptorch::sr_encode(prng_bits, variant, op);
  }
} // namespace

Tensor binaryK_quantize_meta(Tensor a, int64_t, int64_t, int64_t, int64_t prng_bits, bool,
                             int64_t round_mode, int64_t, int64_t)
{
  check_quant_prng(prng_bits, round_mode, "binaryK_quant");
  return quantized_like(a, "binaryK_quant");
}

Tensor &binaryK_quantize_meta_(Tensor &a, int64_t, int64_t, int64_t, int64_t prng_bits, bool,
                               int64_t round_mode, int64_t, int64_t)
{
  check_in_place(a, "binaryK_quant_", "binaryK_quant");
  check_quant_prng(prng_bits, round_mode, "binaryK_quant");
  check_quant_dtype(a, "binaryK_quant_");
  return a;
}

Tensor superfp_quantize_meta(Tensor a, int64_t, int64_t, int64_t, int64_t, int64_t prng_bits, bool,
                             int64_t round_mode, int64_t)
{
  check_quant_prng(prng_bits, round_mode, "superfp_quant");
  return quantized_like(a, "superfp_quant");
}

Tensor &superfp_quantize_meta_(Tensor &a, int64_t, int64_t, int64_t, int64_t, int64_t prng_bits,
                               bool, int64_t round_mode, int64_t)
{
  check_in_place(a, "superfp_quant_", "superfp_quant");
  check_quant_prng(prng_bits, round_mode, "superfp_quant");
  check_quant_dtype(a, "superfp_quant_");
  return a;
}

Tensor fixedpoint_quantize_meta(Tensor a, int64_t, int64_t, int64_t prng_bits, bool, bool,
                                int64_t round_mode)
{
  check_quant_prng(prng_bits, round_mode, "fixedpoint_quant");
  return quantized_like(a, "fixedpoint_quant");
}

Tensor &fixedpoint_quantize_meta_(Tensor &a, int64_t, int64_t, int64_t prng_bits, bool, bool,
                                  int64_t round_mode)
{
  check_in_place(a, "fixedpoint_quant_", "fixedpoint_quant");
  check_quant_prng(prng_bits, round_mode, "fixedpoint_quant");
  check_quant_dtype(a, "fixedpoint_quant_");
  return a;
}

// narrow_float64_tensors (common/narrow_host.h).
Tensor narrow_float64_meta(Tensor a, c10::ScalarType dtype)
{
  TORCH_CHECK(a.scalar_type() == c10::ScalarType::Double,
              "narrow_float64: the input must be float64, got ", a.scalar_type());
  TORCH_CHECK(dtype == c10::ScalarType::Half || dtype == c10::ScalarType::BFloat16 ||
                  dtype == c10::ScalarType::Float,
              "narrow_float64: narrows to float32, float16 or bfloat16, got ", dtype);
  const Tensor a_c = a.contiguous();
  return at::empty_symint(a_c.sym_sizes(), a_c.options().dtype(dtype));
}

std::tuple<Tensor, Tensor> block_pack_meta(Tensor x, c10::IntArrayRef fmt, double elem_max,
                                           double scale_max, double tensor_scale,
                                           int64_t round_mode)
{
  return meta_block_pack(x, fmt, elem_max, scale_max, tensor_scale, round_mode);
}

Tensor block_unpack_meta(Tensor data, Tensor scales, int64_t cols, c10::IntArrayRef fmt,
                         double elem_max, double scale_max, double tensor_scale,
                         at::ScalarType dtype)
{
  return meta_block_unpack(data, scales, cols, fmt, elem_max, scale_max, tensor_scale, dtype);
}

// block_quantize_cpu (cpu/block_kernel.cpp): the result is allocated before
// any check, laid out as x is.
Tensor block_quantize_meta(Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max,
                           double tensor_scale, int64_t round_mode)
{
  Tensor o = at::empty_like(x);
  meta_block_quantize_checks(x, fmt, elem_max, scale_max, tensor_scale, round_mode, "block_quant");
  return o;
}

Tensor &block_quantize_meta_(Tensor &x, c10::IntArrayRef fmt, double elem_max, double scale_max,
                             double tensor_scale, int64_t round_mode)
{
  TORCH_CHECK(x.is_non_overlapping_and_dense(), "block_quant_ writes its argument in place and "
              "needs a tensor whose elements do not overlap, got strides ", x.sym_strides(),
              " for sizes ", x.sym_sizes(), ": use block_quant, which writes a new tensor");
  meta_block_quantize_checks(x, fmt, elem_max, scale_max, tensor_scale, round_mode,
                             "block_quant_");
  return x;
}

Tensor carrier_odd_meta(Tensor a, Tensor b, Tensor c, int64_t op)
{
  sym_check_carrier_odd(a, b, c, op);
  return at::empty_like(a.contiguous());
}

// _stochastic_cast (common/stochastic_cast_host.h): check_tensors' checks on
// symbolic sizes, the job's, and a result like x.
Tensor stochastic_cast_meta(Tensor x, Tensor draws, int64_t family, c10::IntArrayRef format,
                            int64_t stochastic_mode)
{
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "_stochastic_cast: x is float32 or float64, got ", x.scalar_type());
  SymBool same(draws.scalar_type() == at::kLong && draws.device() == x.device() && draws.dim() == x.dim());
  for (int64_t i = 0; i < x.dim() && draws.dim() == x.dim(); ++i)
    same = same & draws.sym_size(i).sym_eq(x.sym_size(i));
  TORCH_SYM_CHECK(same, "_stochastic_cast: draws is an int64 tensor of x's shape and device");
  if (x.scalar_type() == at::kFloat)
    mptorch::stochastic_cast::make_job<float>(family, format, stochastic_mode);
  else
    mptorch::stochastic_cast::make_job<double>(family, format, stochastic_mode);
  return at::empty_like(x.contiguous());
}

TORCH_LIBRARY_IMPL(mptorch, Meta, m)
{
  m.impl("binaryK_quant", TORCH_FN(binaryK_quantize_meta));
  m.impl("superfp_quant", TORCH_FN(superfp_quantize_meta));
  m.impl("binaryK_quant_", TORCH_FN(binaryK_quantize_meta_));
  m.impl("superfp_quant_", TORCH_FN(superfp_quantize_meta_));
  m.impl("fixedpoint_quant", TORCH_FN(fixedpoint_quantize_meta));
  m.impl("fixedpoint_quant_", TORCH_FN(fixedpoint_quantize_meta_));
  m.impl("narrow_float64", TORCH_FN(narrow_float64_meta));
  m.impl("custom_matmul_binaryK", TORCH_FN(binaryK_matmul_meta));
  m.impl("custom_matmul_superfp", TORCH_FN(superfp_matmul_meta));
  m.impl("custom_matmul_binaryK_fma", TORCH_FN(binaryK_matmul_fma_meta));
  m.impl("custom_matmul_superfp_fma", TORCH_FN(superfp_matmul_fma_meta));
  m.impl("custom_matmul_binaryK_mixed", TORCH_FN(binaryK_matmul_mixed_meta));
  m.impl("custom_matmul_superfp_mixed", TORCH_FN(superfp_matmul_mixed_meta));
  m.impl("custom_matmul_binaryK_fma_mixed", TORCH_FN(binaryK_matmul_fma_mixed_meta));
  m.impl("custom_matmul_superfp_fma_mixed", TORCH_FN(superfp_matmul_fma_mixed_meta));
  m.impl("custom_matmul_binaryK_accumulated", TORCH_FN(binaryK_matmul_accumulated_meta));
  m.impl("custom_matmul_superfp_accumulated", TORCH_FN(superfp_matmul_accumulated_meta));
  m.impl("custom_matmul_binaryK_fma_accumulated", TORCH_FN(binaryK_matmul_fma_accumulated_meta));
  m.impl("custom_matmul_superfp_fma_accumulated", TORCH_FN(superfp_matmul_fma_accumulated_meta));
  m.impl("custom_conv_binaryK", TORCH_FN(binaryK_conv_meta));
  m.impl("custom_conv_binaryK_mixed", TORCH_FN(binaryK_conv_mixed_meta));
  m.impl("custom_conv_binaryK_fma", TORCH_FN(binaryK_conv_fma_meta));
  m.impl("custom_conv_binaryK_fma_mixed", TORCH_FN(binaryK_conv_fma_mixed_meta));
  m.impl("custom_conv_superfp", TORCH_FN(superfp_conv_meta));
  m.impl("custom_conv_superfp_mixed", TORCH_FN(superfp_conv_mixed_meta));
  m.impl("custom_conv_superfp_fma", TORCH_FN(superfp_conv_fma_meta));
  m.impl("custom_conv_superfp_fma_mixed", TORCH_FN(superfp_conv_fma_mixed_meta));
  m.impl("block_pack", TORCH_FN(block_pack_meta));
  m.impl("block_unpack", TORCH_FN(block_unpack_meta));
  m.impl("block_quant", TORCH_FN(block_quantize_meta));
  m.impl("block_quant_", TORCH_FN(block_quantize_meta_));
  m.impl("custom_matmul_block", TORCH_FN(block_matmul_meta));
  m.impl("custom_matmul_binaryK_superfp", TORCH_FN(binaryK_superfp_matmul_meta));
  m.impl("custom_matmul_superfp_binaryK", TORCH_FN(superfp_binaryK_matmul_meta));
  m.impl("custom_matmul_binaryK_superfp_accumulated",
         TORCH_FN(binaryK_superfp_matmul_accumulated_meta));
  m.impl("custom_matmul_superfp_binaryK_accumulated",
         TORCH_FN(superfp_binaryK_matmul_accumulated_meta));
  m.impl("custom_conv_binaryK_superfp", TORCH_FN(binaryK_superfp_conv_meta));
  m.impl("custom_conv_superfp_binaryK", TORCH_FN(superfp_binaryK_conv_meta));
  m.impl("custom_matmul_binaryK_odd", TORCH_FN(binaryK_matmul_odd_meta));
  m.impl("custom_matmul_superfp_odd", TORCH_FN(superfp_matmul_odd_meta));
  m.impl("custom_matmul_binaryK_fma_odd", TORCH_FN(binaryK_matmul_fma_odd_meta));
  m.impl("custom_matmul_superfp_fma_odd", TORCH_FN(superfp_matmul_fma_odd_meta));
  m.impl("_carrier_odd", TORCH_FN(carrier_odd_meta));
  m.impl("_stochastic_cast", TORCH_FN(stochastic_cast_meta));
}
