#include "quant.h"
#include "bit_helper.h"
#include "binary8.h"
#include <cassert>
#include <random>
#include <torch/torch.h>
#include <tuple>
#include <cmath>
#include <cstdint>

using namespace at;

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT(x)                                                         \
  CHECK_CPU(x);                                                                \
  CHECK_CONTIGUOUS(x);

#define RFLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define RBITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))
#define FLOAT_TO_BITS(f, i)                                                    \
  assert(sizeof f == sizeof i);                                                \
  std::memcpy(&i, &f, sizeof i)
#define BITS_TO_FLOAT(i, f)                                                    \
  assert(sizeof f == sizeof i);                                                \
  std::memcpy(&f, &i, sizeof f)

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0);

static uint32_t rn_prob[24] = {
    4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768,
    16384,   8192,    4096,    2048,   1024,   512,    256,   128,
    64,      32,      16,      8,      4,      2,      1,     0};

template <typename T> T clamp_helper(T a, T min, T max) {
  if (a > max)
    return max;
  else if (a < min)
    return min;
  else
    return a;
}

void printBits(size_t const size, void const *const ptr) {
  unsigned char *b = (unsigned char *)ptr;
  unsigned char byte;
  int i, j;

  for (i = size - 1; i >= 0; i--) {
    for (j = 7; j >= 0; j--) {

      byte = (b[i] >> j) & 1;
      printf("%u", byte);
      if ((i == size - 1 && j == 7) || (i == size - 2 && j == 7))
        printf(" ");
    }
  }
}

template <typename T> T clamp_mask_helper(T a, T min, T max, uint8_t *mask) {
  if (a > max) {
    *mask = 1;
    return max;
  } else if (a < min) {
    *mask = 1;
    return min;
  } else
    return a;
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl, bool symmetric) {
  CHECK_INPUT(a);
  auto r = rand_like(a);
  auto a_array = a.data_ptr<float>();
  auto r_array = r.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  auto m = zeros_like(a, torch::CPU(kByte));
  auto m_array = m.data_ptr<uint8_t>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++) {
    o_array[i] = round(a_array[i], r_array[i], sigma);
    o_array[i] =
        clamp_mask_helper<float>(o_array[i], t_min, t_max, m_array + i);
  }
  return std::make_tuple(o, m);
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl, bool symmetric) {
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  auto m = zeros_like(a, torch::CPU(kByte));
  auto m_array = m.data_ptr<uint8_t>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++) {
    o_array[i] = round(a_array[i], 0.5, sigma);
    o_array[i] =
        clamp_mask_helper<float>(o_array[i], t_min, t_max, m_array + i);
  }
  return std::make_tuple(o, m);
}

Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl, bool clamp,
                                       bool symmetric) {
  CHECK_INPUT(a);
  auto r = rand_like(a);
  auto a_array = a.data_ptr<float>();
  auto r_array = r.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++) {
    o_array[i] = round(a_array[i], r_array[i], sigma);
    if (clamp) {
      o_array[i] = clamp_helper(o_array[i], t_min, t_max);
    }
  }
  return o;
}

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl, bool clamp,
                                    bool symmetric) {
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++) {
    o_array[i] = round(a_array[i], 0.5, sigma);
    if (clamp) {
      o_array[i] = clamp_helper(o_array[i], t_min, t_max);
    }
  }
  return o;
}

uint32_t round_bitwise(uint32_t target, int man_bits, Mode rounding) {
  uint32_t mask = (1 << (23 - man_bits)) - 1;

  uint32_t rand_prob;
  if (rounding == rStochastic) {
    rand_prob = (dis(gen)) & mask;
  } else {
    rand_prob = rn_prob[man_bits];
  }
  uint32_t add_r = target + rand_prob;

  uint32_t quantized = add_r & ~mask;

  return quantized;
}

void block_quantize_helper(float *input, float *output, float *max_elem, int wl,
                           int size, Mode rounding) {
  for (int64_t i = 0; i < size; i++) {

    uint32_t max_num;
    FLOAT_TO_BITS(max_elem[i], max_num);
    uint32_t max_exp = max_num << 1 >> 24 << 23;
    float base_float;
    BITS_TO_FLOAT(max_exp, base_float);
    base_float *= 6;

    float target_rebase = input[i] + base_float;
    uint32_t target_bits;
    FLOAT_TO_BITS(target_rebase, target_bits);
    uint32_t quantized_bits = round_bitwise(
        target_bits, wl, rounding); // -1 sign, -1 virtual, +2 base
    float quantized_rebase;
    BITS_TO_FLOAT(quantized_bits, quantized_rebase);
    float quantized = quantized_rebase - base_float;

    uint32_t quantize_bits;
    FLOAT_TO_BITS(quantized, quantize_bits);
    uint32_t clip_quantize =
        clip_max_exponent(wl - 2, max_exp, quantize_bits);
    BITS_TO_FLOAT(clip_quantize, quantized);

    output[i] = quantized;
  }
}

Tensor get_max_entry(Tensor a, int dim) {
  Tensor max_entry;
  if (dim == -1) {
    max_entry = at::max(at::abs(a)).expand_as(a).contiguous();
  } else if (dim == 0) {
    Tensor input_view = a.view({a.size(0), -1});
    max_entry = std::get<0>(input_view.abs().max(1, true))
                    .expand_as(input_view)
                    .view_as(a)
                    .contiguous();
  } else {
    Tensor input_transpose = a.transpose(0, dim);
    Tensor input_view =
        input_transpose.contiguous().view({input_transpose.size(0), -1});
    Tensor max_transpose = std::get<0>(input_view.abs().max(1, true))
                               .expand_as(input_view)
                               .view_as(input_transpose);
    max_entry = max_transpose.transpose(dim, 0).contiguous();
  }
  return max_entry;
}

Tensor block_quantize_nearest(Tensor a, int wl, int dim) {
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();

  // get maximum number and base
  Tensor max_entry = get_max_entry(a, dim);
  auto max_elem = max_entry.data_ptr<float>();
  block_quantize_helper(a_array, o_array, max_elem, wl, size, rNearest);
  return o;
}

Tensor block_quantize_stochastic(Tensor a, int wl, int dim) {
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();

  // get maximum number and base
  Tensor max_entry = get_max_entry(a, dim);
  auto max_elem = max_entry.data_ptr<float>();
  // std::srand(time(0));
  block_quantize_helper(a_array, o_array, max_elem, wl, size, rStochastic);
  return o;
}

float cast_fp_stochastic(float origin_float, uint32_t rand_prob,
                                    int man_bits, int exp_bits,
                                    bool subnormal_support = true,
                                    bool saturate = false) {
  uint32_t target, quantize_bits;
  target = RFLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);

  if (subnormal && subnormal_support) {
    float shift_float, val;
    int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
    shift_float = RBITS_TO_FLOAT(&shift_bits);
    val = origin_float + shift_float;
    target = RFLOAT_TO_BITS(&val);
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantized = RBITS_TO_FLOAT(&quantize_bits) - shift_float;
  } else {
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantize_bits =
        clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
    quantized = RBITS_TO_FLOAT(&quantize_bits);
  }

  return quantized;
}

float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
                                       bool subnormal_support = true,
                                       bool saturate = false)
{
    uint32_t target, quantize_bits;
    target = RFLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && subnormal_support)
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest(target, exp_diff);
            quantize_bits =
                clip_exponent_with_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = RBITS_TO_FLOAT(&quantize_bits);
        }
        // handle NaN/inf inputs
        else if (target_exp == 128)
        {
            quantized = origin_float;
        }
        // normal value range or overflow
        else
        {
            quantize_bits = round_bitwise_nearest(target, man_bits);
            quantize_bits =
                clip_exponent_without_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = RBITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

float float_quantize(float origin_float, int man_bits, int exp_bits,
                     Mode rounding, bool subnormal_support,
                     bool saturate = false) {
  float quantized;
  switch(rounding) {
    case Mode::rStochastic:
      {
      uint32_t mask = (1 << (23 - man_bits)) - 1;
      uint32_t rand_prob = (dis(gen)) & mask;
      quantized = cast_fp_stochastic(
                          origin_float, rand_prob, man_bits, exp_bits, 
                          subnormal_support, saturate);
      }
      break;
    default:
      quantized = cast_fp_nearest(
                          origin_float, man_bits, exp_bits, 
                          subnormal_support, saturate);
  }
  return quantized;
}


Tensor float_quantize(Tensor a, int man_bits, int exp_bits, Mode rounding,
                      bool subnormal_support, bool saturate = false) {
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();

  for (int64_t i = 0; i < size; i++) {
    o_array[i] = float_quantize(a_array[i], man_bits, exp_bits, rounding, subnormal_support, saturate);
  }
  return o;
}

// TODO: support saturate logic
// Remark: bias = 2^{e-1}
float superfp_quantize(float origin, int man_bits, int exp_bits, int binades, bool saturate = false) {
    int32_t sat = saturate;
    uint32_t target;
    target = RFLOAT_TO_BITS(&origin);
    float ftarget{0u};

    int32_t target_exp = (target << 1 >> 24) - 127;
    int32_t min_exp = 1 - ((1 << (exp_bits - 1))) + (binades - 1);
    int32_t max_exp = ((1 << (exp_bits - 1)) - 2) - (binades - 1);
    bool subnormal = (target_exp < min_exp);
    bool supnormal = (target_exp > max_exp);
    if(subnormal) {
        if (target_exp < min_exp - binades * (1 << man_bits) + 1) // underflow
            return 0.0f;
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = RBITS_TO_FLOAT(&qtarget);
    } else if (supnormal) {
        if (target_exp == 128) { // NaN/inf
            if (saturate) {
                if ((target & 0x7FFFFFFF) == 0x7F800000) { // inf
                    uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                    return RBITS_TO_FLOAT(&qtarget);
                } else { // NaN
                    return origin;
                }
            } else {
                return origin;
            }
        } else if (target_exp >= max_exp + binades * (1 << man_bits) - 1 + sat) { // overflow
            if (saturate) {
                uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                return RBITS_TO_FLOAT(&qtarget);
            } else {
                if (((target << 9) == 0u) && (target_exp == max_exp + binades * (1 << man_bits) - 1))
                    return origin;
                else {
                    float infty = INFINITY;
                    uint32_t qtarget = (target >> 31 << 31) | RFLOAT_TO_BITS(&infty);
                    return RBITS_TO_FLOAT(&qtarget);
                }
            }
        }
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = RBITS_TO_FLOAT(&qtarget);
    } else {
        uint32_t qtarget = round_bitwise_nearest(target, man_bits);
        ftarget = RBITS_TO_FLOAT(&qtarget);
    }

    return ftarget;
}

Tensor superfp_quantize(Tensor a, int man_bits, int exp_bits, int binades, bool saturate = false) 
{
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();

  for (int64_t i = 0; i < size; i++) {
    o_array[i] = superfp_quantize(a_array[i], man_bits, exp_bits, binades, saturate);
  }
  return o;
}

void mm_fp(float *a, float *b, float *c, int M, int K, int N, int man_add,
           int exp_add, int man_mul, int exp_mul, bool subnormals,
           bool saturate) {
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      for (int64_t k = 0; k < K; ++k) {
        c[i * N + j] = float_quantize(
            c[i * N + j] + float_quantize(a[i * K + k] * b[k * N + j], man_mul,
                                          exp_mul, rNearest, subnormals,
                                          saturate),
            man_add, exp_add, rNearest, subnormals, saturate);
      }
}

void mm_fp_fma(float *a, float *b, float *c, int M, int K, int N, int man_fma,
               int exp_fma, bool subnormals, bool saturate) {
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      for (int64_t k = 0; k < K; ++k) {
        c[i * N + j] =
            float_quantize(fmaf(a[i * K + k], b[k * N + j], c[i * N + j]),
                           man_fma, exp_fma, rNearest, subnormals, saturate);
      }
}

void float_quantize_nearest_mm(Tensor a, Tensor b, Tensor c, int M, int N,
                               int K, int man_add, int exp_add, int man_mul,
                               int exp_mul, bool subnormals, bool saturate) {
  mm_fp(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, K, N,
        man_add, exp_add, man_mul, exp_mul, subnormals, saturate);
}

void float_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c, int M, int N,
                                   int K, int man_fma, int exp_fma,
                                   bool subnormals, bool saturate) {
  mm_fp_fma(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, K,
            N, man_fma, exp_fma, subnormals, saturate);
}

Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits,
                                 bool subnormals, bool saturate) {
  return float_quantize(a, man_bits, exp_bits, rStochastic, subnormals,
                        saturate);
}

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits,
                              bool subnormals, bool saturate) {
  return float_quantize(a, man_bits, exp_bits, rNearest, subnormals, saturate);
}

Tensor superfp_quantize_nearest(Tensor a, int man_bits, int exp_bits, int binades,
                              bool saturate) {
  return superfp_quantize(a, man_bits, exp_bits, binades, saturate);
}

// Uses the standard softmax formula e^xi/sum(e^xj)
// Returns ouput tensor of softmaxxed and quantized values
void float_quantize_nearest_softmax_forward(Tensor a, Tensor o, int dim,
                                            int man_expf, int exp_expf,
                                            int man_off, int exp_off,
                                            int man_acc, int exp_acc,
                                            int man_div, int exp_div,
                                            bool subnormals, bool saturate)
{
  auto a_array = a.data_ptr<float>();
  auto o_array = o.data_ptr<float>();

  DimStrides strides;
  dim_striding(a, dim, strides);

  auto quant = [subnormals, saturate](float x, int man_bits, int exp_bits) {
    return float_quantize(x, man_bits, exp_bits, rNearest, subnormals, saturate);
  };

  for (int i = 0; i < strides.outer_size * strides.inner_size; ++i) {
    int outer_idx = i / strides.inner_size;
    int inner_idx = i % strides.inner_size;

    // Get beginning pointers to the row being softmaxed
    int base_index = outer_idx * strides.outer_stride + inner_idx;
    float* input = a_array + base_index;
    float* output = o_array + base_index;
  
    // Get the maximum of the current row being softmaxed.
    // This ensures no overflow and more stable exponentials.
    float max = input[0];
    for (int k = 1; k < strides.dim_size; ++k) {
      int idx = k * strides.dim_stride;
      max = fmaxf(max, input[idx]);
    }

    // Calculate the sum and store quantized values of exp(input) into output
    float sum = 0.0f;
    for (int k = 0; k < strides.dim_size; ++k) {
      int idx = k * strides.dim_stride;
      float x = quant(input[idx] - max, man_off, exp_off);
      float e_x = quant(expf(x), man_expf, exp_expf);
      output[idx] = e_x;
      sum = quant(sum + e_x, man_acc, exp_acc);
    }
    // Divide all outputs by the current sum
    for (int k = 0; k < strides.dim_size; ++k) {
      int idx = k * strides.dim_stride;
      output[idx] = quant(output[idx] / sum, man_div, exp_div);
    }
  }
}

// Uses the LogSumExp formula to compute softmax without divisions
// Returns ouput tensor of softmaxxed and quantized values
void float_quantize_nearest_softmax_lse_forward(Tensor a, Tensor o, int dim,
                                                int man_expf, int exp_expf,
                                                int man_off, int exp_off,
                                                int man_lse, int exp_lse,
                                                bool subnormals, bool saturate)
{
  auto a_array = a.data_ptr<float>();
  auto o_array = o.data_ptr<float>();

  DimStrides strides;
  dim_striding(a, dim, strides);

  auto quant = [subnormals, saturate](float x, int man_bits, int exp_bits) {
    return float_quantize(x, man_bits, exp_bits, rNearest, subnormals, saturate);
  };

  for (int i = 0; i < strides.outer_size * strides.inner_size; ++i) {
    int outer_idx = i / strides.inner_size;
    int inner_idx = i % strides.inner_size;

    // Get beginning pointers to the row being softmaxed
    int base_index = outer_idx * strides.outer_stride + inner_idx;
    float* input = a_array + base_index;
    float* output = o_array + base_index;

    // Get the maximum of the current row being softmaxed.
    // This ensures no overflow and more stable exponentials.
    // Index of maximum is saved for later.;
    float max = input[0];
    for (int k = 1; k < strides.dim_size; ++k) {
      int idx = k * strides.dim_stride;
      max = fmaxf(max, input[idx]);
    }

    // Calculate LogSumExp
    float x0 = quant(input[0] - max, man_off, exp_off);
    output[0] = x0;
    float lgs = x0; // = log(exp(x[0] - max))
    for (int k = 1; k < strides.dim_size; ++k) {
      int idx = k * strides.dim_stride;
      float x = quant(input[idx] - max, man_off, exp_off);
      output[idx] = x;
      lgs = quant(logf(expf(lgs) + expf(x)), man_lse, exp_lse);
    }
    // Compute output tensor elements
    for (int k = 0; k < strides.dim_size; ++k) {
      int idx = k * strides.dim_stride;
      float x = quant(output[idx] - lgs, man_off, exp_off);
      output[idx] = quant(expf(x), man_expf, exp_expf);
    }
  }
}

void float_quantize_nearest_softmax_backward(Tensor a, Tensor g, Tensor o, int dim,
                                             int man_add, int exp_add,
                                             int man_mul, int exp_mul,
                                             int man_div, int exp_div,
                                             bool subnormals, bool saturate)
{
  auto g_array = g.data_ptr<float>();
  auto a_array = a.data_ptr<float>();
  auto o_array = o.data_ptr<float>();

  DimStrides strides;
  dim_striding(a, dim, strides);

  auto quant = [subnormals, saturate](float x, int man_bits, int exp_bits) {
    return float_quantize(x, man_bits, exp_bits, rNearest, subnormals, saturate);
  };

  for (int i = 0; i < strides.outer_size * strides.inner_size; ++i) {
    int outer_idx = i / strides.inner_size;
    int inner_idx = i % strides.inner_size;

    int base_index = outer_idx * strides.outer_stride + inner_idx;
    float* input = a_array + base_index;
    float* grad = g_array + base_index;
    float* output = o_array + base_index;

    float input_sum = 0.f;
    float weighted_grad_sum = 0.f;
    for (int k = 0; k < strides.dim_size; ++k) {
      int idx = k * strides.dim_stride;
      input_sum = quant(input_sum + input[idx], man_add, exp_add);
      float prod = quant(input[idx] * grad[idx], man_mul, exp_mul);
      weighted_grad_sum = quant(weighted_grad_sum + prod, man_add, exp_add);
    }

    for (int k = 0; k < strides.dim_size; ++k) {
      int idx = k * strides.dim_stride;
      float a = quant(grad[idx] - weighted_grad_sum, man_add, exp_add);
      float b = quant(a * input[idx], man_mul, exp_mul);
      output[idx] = quant(b / input_sum, man_div, exp_div);
    }
  }
}

Tensor binary8_quantize_nearest_cpu(Tensor a, int P, bool is_signed, OverflowPolicy overflow_policy, bool subnormals)
{
  auto o = zeros_like(a);
  int size = a.numel(); // gets number of elements in tensor a

  if (is_signed == true){ // signed
      binary8_signed_nearest(
      a.data_ptr<float>(), o.data_ptr<float>(), size, P, overflow_policy, subnormals);
  } else {  // unsigned
      binary8_unsigned_nearest(
      a.data_ptr<float>(), o.data_ptr<float>(), size, P, overflow_policy, subnormals);
  }

  return o;
}

Tensor binary8_quantize_stochastic_cpu(Tensor a, int P, int prng_bits, bool is_signed, OverflowPolicy overflow_policy, bool subnormals)
{
  auto o = zeros_like(a);
  // generate random number on the CPU for the SR operation
  auto rand_ints = randint_like(a, INT_MAX, device(kCPU).dtype(kInt));
  int size = a.numel(); // gets number of elements in tensor a

  if (is_signed == true){ // signed
      binary8_signed_stochastic(
      a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size, P, prng_bits, overflow_policy, subnormals);
  } else {  // unsigned
      binary8_unsigned_stochastic(
      a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size, P, prng_bits, overflow_policy, subnormals);
  }

  return o;
}

Tensor binary8_quantize_truncate_cpu(Tensor a, int P, bool is_signed, OverflowPolicy overflow_policy, bool subnormals)
{
  auto o = zeros_like(a);
  int size = a.numel(); // gets number of elements in tensor a

  if (is_signed == true){ // signed
      binary8_signed_truncate(
      a.data_ptr<float>(), o.data_ptr<float>(), size, P, overflow_policy, subnormals);
  } else {  // unsigned
      binary8_unsigned_truncate(
      a.data_ptr<float>(), o.data_ptr<float>(), size, P, overflow_policy, subnormals);
  }

  return o;
}