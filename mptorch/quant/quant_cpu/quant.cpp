#include "quant.h"
#include "bit_helper.h"
#include "layernorm.h"
#include "softmax.h"
#include <cassert>
#include <random>
#include <torch/torch.h>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <vector>

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

void float_quantize_nearest_softmax_forward(Tensor a, Tensor o, int dim,
                                            int man_exp, int exp_exp,
                                            int man_off, int exp_off,
                                            int man_acc, int exp_acc,
                                            bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_forward(
    a.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [subnormals, saturate, man_exp, exp_exp] (float x) {
      return float_quantize(x, man_exp, exp_exp, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_off, exp_off] (float x) {
      return float_quantize(x, man_off, exp_off, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_acc, exp_acc] (float x) {
      return float_quantize(x, man_acc, exp_acc, rNearest, subnormals, saturate);
    }
  );
}

void float_quantize_nearest_softmax_lse_forward(Tensor a, Tensor o, int dim,
                                                int man_off, int exp_off,
                                                int man_lse, int exp_lse,
                                                bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_lse_forward(
    a.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [subnormals, saturate, man_off, exp_off] (float x) {
      return float_quantize(x, man_off, exp_off, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_lse, exp_lse] (float x) {
      return float_quantize(x, man_lse, exp_lse, rNearest, subnormals, saturate);
    }
  );
}

void float_quantize_nearest_softmax_backward(Tensor a, Tensor g, Tensor o, int dim,
                                             int man_add, int exp_add,
                                             int man_mul, int exp_mul,
                                             bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_backward(
    a.data_ptr<float>(), g.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [subnormals, saturate, man_add, exp_add] (float x) {
      return float_quantize(x, man_add, exp_add, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_mul, exp_mul] (float x) {
      return float_quantize(x, man_mul, exp_mul, rNearest, subnormals, saturate);
    }
  );
}

void superfp_quantize_nearest_softmax_forward(Tensor a, Tensor o, int dim,
                                int man_exp, int exp_exp, int binades_exp,
                                int man_off, int exp_off, int binades_off,
                                int man_acc, int exp_acc, int binades_acc,
                                bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_forward(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [man_exp, exp_exp, binades_exp, saturate] (float x) { 
      return superfp_quantize(x, man_exp, exp_exp, binades_exp, saturate);
    },
    [man_off, exp_off, binades_off, saturate] (float x) { 
      return superfp_quantize(x, man_off, exp_off, binades_off, saturate);
    },
    [man_acc, exp_acc, binades_acc, saturate] (float x) { 
      return superfp_quantize(x, man_acc, exp_acc, binades_acc, saturate);
    }
  );
}

void superfp_quantize_nearest_softmax_lse_forward(Tensor a, Tensor o, int dim,
                                int man_off, int exp_off, int binades_off,
                                int man_lse, int exp_lse, int binades_lse,
                                bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_lse_forward(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [man_off, exp_off, binades_off, saturate] (float x) { 
      return superfp_quantize(x, man_off, exp_off, binades_off, saturate);
    },
    [man_lse, exp_lse, binades_lse, saturate] (float x) { 
      return superfp_quantize(x, man_lse, exp_lse, binades_lse, saturate);
    }
  );
}

void superfp_quantize_nearest_softmax_backward(Tensor a, Tensor g, Tensor o, int dim,
                                int man_add, int exp_add, int binades_add,
                                int man_mul, int exp_mul, int binades_mul,
                                bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_backward(a.data_ptr<float>(), g.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [man_add, exp_add, binades_add, saturate] (float x) { 
      return superfp_quantize(x, man_add, exp_add, binades_add, saturate);
    },
    [man_mul, exp_mul, binades_mul, saturate] (float x) { 
      return superfp_quantize(x, man_mul, exp_mul, binades_mul, saturate);
    }
  );
}

void float_quantize_layernorm_forward(Tensor input, Tensor weight, Tensor bias,
                                      Tensor output, Tensor mean, Tensor rstd,
                                      float eps, std::vector<int> &dims,
                                      int man_acc, int exp_acc,
                                      int man_mul, int exp_mul,
                                      int man_div, int exp_div,
                                      int man_sqrt, int exp_sqrt,
                                      bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_forward(
    input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
    output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), 
    eps, sizes,
    [subnormals, saturate, man_acc, exp_acc] (float x) {
      return float_quantize(x, man_acc, exp_acc, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_mul, exp_mul] (float x) {
      return float_quantize(x, man_mul, exp_mul, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_div, exp_div] (float x) {
      return float_quantize(x, man_div, exp_div, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_sqrt, exp_sqrt] (float x) {
      return float_quantize(x, man_sqrt, exp_sqrt, rNearest, subnormals, saturate);
    }
  );
}

void float_quantize_layernorm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor bias, Tensor mean, Tensor rstd,
                                      Tensor grad_input, Tensor grad_weight, Tensor grad_bias,
                                      std::vector<int> &dims,
                                      int man_acc, int exp_acc,
                                      int man_mul, int exp_mul,
                                      int man_div, int exp_div,
                                      bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_backward(
    input.data_ptr<float>(), grad_output.data_ptr<float>(), 
    weight.data_ptr<float>(), bias.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), 
    grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(), sizes,
    [subnormals, saturate, man_acc, exp_acc] (float x) {
      return float_quantize(x, man_acc, exp_acc, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_mul, exp_mul] (float x) {
      return float_quantize(x, man_mul, exp_mul, rNearest, subnormals, saturate);
    },
    [subnormals, saturate, man_div, exp_div] (float x) {
      return float_quantize(x, man_div, exp_div, rNearest, subnormals, saturate);
    }
  );
}

void superfp_quantize_layernorm_forward(Tensor input, Tensor weight, Tensor bias,
                                      Tensor output, Tensor mean, Tensor rstd,
                                      float eps, std::vector<int> &dims,
                                      int man_acc, int exp_acc, int binades_acc,
                                      int man_mul, int exp_mul, int binades_mul,
                                      int man_div, int exp_div, int binades_div,
                                      int man_sqrt, int exp_sqrt, int binades_sqrt,
                                      bool saturate)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_forward(
    input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
    output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), 
    eps, sizes,
    [saturate, man_acc, exp_acc, binades_acc] (float x) {
      return superfp_quantize(x, man_acc, exp_acc, binades_acc, saturate);
    },
    [saturate, man_mul, exp_mul, binades_mul] (float x) {
      return superfp_quantize(x, man_mul, exp_mul, binades_mul, saturate);
    },
    [saturate, man_div, exp_div, binades_div] (float x) {
      return superfp_quantize(x, man_div, exp_div, binades_div, saturate);
    },
    [saturate, man_sqrt, exp_sqrt, binades_sqrt] (float x) {
      return superfp_quantize(x, man_sqrt, exp_sqrt, binades_sqrt, saturate);
    }
  );
}

void superfp_quantize_layernorm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor bias, Tensor mean, Tensor rstd,
                                      Tensor grad_input, Tensor grad_weight, Tensor grad_bias,
                                      std::vector<int> &dims,
                                      int man_acc, int exp_acc, int binades_acc,
                                      int man_mul, int exp_mul, int binades_mul,
                                      int man_div, int exp_div, int binades_div,
                                      bool saturate)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_backward(
    input.data_ptr<float>(), grad_output.data_ptr<float>(), 
    weight.data_ptr<float>(), bias.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), 
    grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(), sizes,
    [saturate, man_acc, exp_acc, binades_acc] (float x) {
      return superfp_quantize(x, man_acc, exp_acc, binades_acc, saturate);
    },
    [saturate, man_mul, exp_mul, binades_mul] (float x) {
      return superfp_quantize(x, man_mul, exp_mul, binades_mul, saturate);
    },
    [saturate, man_div, exp_div, binades_div] (float x) {
      return superfp_quantize(x, man_div, exp_div, binades_div, saturate);
    }
  );
}
