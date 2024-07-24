#include "quant.h"
#include "bit_helper.h"
#include "binary8.h"
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

void float_quantize_layernorm_forward(Tensor a, Tensor weight, Tensor bias,
                                      Tensor o, Tensor mean, Tensor rstd,
                                      float eps, std::vector<int> &dims,
                                      int man_acc, int exp_acc,
                                      int man_mul, int exp_mul,
                                      int man_div, int exp_div,
                                      int man_sqrt, int exp_sqrt,
                                      bool subnormals, bool saturate){
  auto a_array = a.data_ptr<float>();
  auto w_array = weight.data_ptr<float>();
  auto b_array = bias.data_ptr<float>();

  auto o_array = o.data_ptr<float>();
  auto m_array = mean.data_ptr<float>();
  auto r_array = rstd.data_ptr<float>();

  auto quant = [subnormals, saturate](float x, int man_bits, int exp_bits){
    return float_quantize(x, man_bits, exp_bits, rNearest, subnormals, saturate);
  };

  std::vector<int> real_dims(dims.size());
  for (int i = 0; i < dims.size(); i++){
    real_dims[i] = (a.dim() + (dims[i] % a.dim())) % a.dim();
  }

  int dim_size = 1;
  for (int dim : real_dims){
    dim_size *= a.size(dim);
  }

  int min_dim = real_dims.back();
  int max_dim = real_dims.front();

  int B = 1;
  for (int i = 0; i < min_dim; i++){
    B *= a.size(i);
  }

  int T = 1;
  for (int i = max_dim + 1; i < a.dim(); i++){
    T *= a.size(i);
  }

  int outer_stride = dim_size * T;

  for (int i = 0; i < B * T; i++){
    int b = i / T;
    int t = i % T;

    int base_index = (b*outer_stride) + t;
    float* input = a_array + base_index;
    float* output = o_array + base_index;

    float m_input = 0.0f;
    float m_input_2 = 0.0f;
    for (int k = 0; k < dim_size; k++){
      int idx = k*T;
      float input_2 = quant(input[idx] * input[idx], man_mul, exp_mul);

      m_input = quant(m_input + input[idx], man_acc, exp_acc);
      m_input_2 = quant(m_input_2 + input_2, man_acc, exp_acc);
    }
    m_input = quant(m_input/dim_size, man_div, exp_div);
    m_input_2 = quant(m_input_2/dim_size, man_div, exp_div);

    float M = quant(m_input*m_input, man_mul, exp_mul);
    float variance = quant(m_input_2 - M, man_acc, exp_acc);

    float rad = quant(variance + eps, man_acc, exp_acc);
    float std = quant(sqrtf(rad), man_sqrt, exp_sqrt);
    for (int k = 0; k < dim_size; k++){
      int idx = k * T;
      float numer = quant(input[idx] - m_input, man_acc, exp_acc);
      float norm = quant(numer/std, man_div, exp_div);
      float out = quant(w_array[k] * norm, man_mul, exp_mul);
      output[idx] = out + b_array[k];
    }
    m_array[b * T + t] = m_input;
    r_array[b * T + t] = quant(1.0f/std, man_div, exp_div); 
  }
}

void float_quantize_layernorm_backward(Tensor a, Tensor g, Tensor weight, Tensor bias,
                                       Tensor o, Tensor mean, Tensor rstd,
                                       std::vector<int> &dims,
                                       int man_acc, int exp_acc,
                                       int man_mul, int exp_mul,
                                       int man_div, int exp_div,
                                       bool subnormals, bool saturate){
  auto a_array = a.data_ptr<float>();
  auto g_array = g.data_ptr<float>();
  auto w_array = weight.data_ptr<float>();
  auto b_array = bias.data_ptr<float>();

  auto o_array = o.data_ptr<float>();
  auto m_array = mean.data_ptr<float>();
  auto r_array = rstd.data_ptr<float>();

  Tensor gamma = ones_like(weight);
  Tensor beta = zeros_like(bias);
  auto grad_gamma = gamma.data_ptr<float>();
  auto grad_beta = beta.data_ptr<float>();

  auto quant = [subnormals, saturate](float x, int man_bits, int exp_bits){
    return float_quantize(x, man_bits, exp_bits, rNearest, subnormals, saturate);
  };

  std::vector<int> real_dims(dims.size());
  for (int i = 0; i < dims.size(); i++){
    real_dims[i] = (a.dim() + (dims[i] % a.dim())) % a.dim();
  }

  int dim_size = 1;
  for (int dim : real_dims){
    dim_size *= a.size(dim);
  }

  int min_dim = real_dims.back();
  int max_dim = real_dims.front();

  int B = 1;
  for (int i = 0; i < min_dim; i++){
    B *= a.size(i);
  }

  int T = 1;
  for (int i = max_dim + 1; i < a.dim(); i++){
    T *= a.size(i);
  }

  int outer_stride = dim_size * T;

  for (int i = 0; i < B * T; i++){
    int b = i / T;
    int t = i % T;

    int base_index = (b*outer_stride) + t;
    float* input = a_array + base_index;
    float* gradient = g_array + base_index;
    float* output = o_array + base_index;

    float m = m_array[b * T + t];
    float r = r_array[b * T + t];

    // two reduce operations
    float grad_sum = 0.0f;
    float grad_sum_xhat = 0.0f;
    for (int k = 0; k < dim_size; k++){
      int idx = k * T;
      float in_m = quant(input[idx] - m, man_acc, exp_acc);
      float xhat = quant(in_m * r, man_mul, exp_mul);
      float grad_xhat = quant(w_array[k] * gradient[idx], man_mul, exp_mul);
      float dot_xhat = quant(xhat * grad_xhat, man_mul, exp_mul);
      grad_sum = quant(grad_sum + grad_xhat, man_acc, exp_acc);
      grad_sum_xhat = quant(grad_sum_xhat + dot_xhat, man_acc, exp_acc);
    }
    grad_sum = quant(grad_sum/dim_size, man_div, exp_div);
    grad_sum_xhat = quant(grad_sum_xhat/dim_size, man_div, exp_div);

    // iterate and accumulate 
    for (int k = 0; k < dim_size; k++){
      int idx = k * T;
      float in_m = quant(input[idx] - m, man_acc, exp_acc);
      float xhat = quant(in_m * r, man_mul, exp_mul);
      float xhat_gradient = quant(xhat * gradient[idx], man_mul, exp_mul);
      float grad_xhat = quant(w_array[k] * gradient[idx], man_mul, exp_mul);

      grad_beta[k] = quant(grad_beta[k] + gradient[idx], man_acc, exp_acc);
      grad_gamma[k] = quant(grad_gamma[k] + xhat_gradient, man_acc, exp_acc);

      float weighted_grad_sum = quant(xhat * grad_sum_xhat, man_mul, exp_mul);
      float grad_input = grad_xhat;
      grad_input = quant(grad_input - grad_sum, man_acc, exp_acc);
      grad_input = quant(grad_input - weighted_grad_sum, man_acc, exp_acc);
      grad_input = quant(grad_input * r, man_mul, exp_mul);

      output[idx] = grad_input;
    }
  }
}