#include "quant_cpu.h"
#include <assert.h>
#include <random>
#include <torch/torch.h>
#include <tuple>

using namespace at;

enum Mode { rNearest, rStochastic };

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT(x)                                                         \
  CHECK_CPU(x);                                                                \
  CHECK_CONTIGUOUS(x);

#define RFLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int *>(x))
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

static unsigned int rn_prob[24] = {
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

unsigned int round_bitwise(unsigned int target, int man_bits, Mode rounding) {
  unsigned int mask = (1 << (23 - man_bits)) - 1;

  unsigned int rand_prob;
  if (rounding == rStochastic) {
    rand_prob = (dis(gen)) & mask;
  } else {
    rand_prob = rn_prob[man_bits];
  }
  unsigned int add_r = target + rand_prob;

  unsigned int quantized = add_r & ~mask;

  return quantized;
}

void block_quantize_helper(float *input, float *output, float *max_elem, int wl,
                           int size, Mode rounding) {
  for (int64_t i = 0; i < size; i++) {

    unsigned int max_num;
    FLOAT_TO_BITS(max_elem[i], max_num);
    unsigned int max_exp = max_num << 1 >> 24 << 23;
    float base_float;
    BITS_TO_FLOAT(max_exp, base_float);
    base_float *= 6;

    float target_rebase = input[i] + base_float;
    unsigned int target_bits;
    FLOAT_TO_BITS(target_rebase, target_bits);
    unsigned int quantized_bits = round_bitwise(
        target_bits, wl, rounding); // -1 sign, -1 virtual, +2 base
    float quantized_rebase;
    BITS_TO_FLOAT(quantized_bits, quantized_rebase);
    float quantized = quantized_rebase - base_float;

    unsigned int quantize_bits;
    FLOAT_TO_BITS(quantized, quantize_bits);
    unsigned int clip_quantize =
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

Tensor float_quantize(Tensor a, int man_bits, int exp_bits, Mode rounding,
                      bool subnormal_support, bool saturate = false) {
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int size = a.numel();

  for (int64_t i = 0; i < size; i++) {
    unsigned int target, quantize_bits;
    FLOAT_TO_BITS(a_array[i], target);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal && subnormal_support) {
      float shift_float, val;
      int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
      BITS_TO_FLOAT(shift_bits, shift_float);
      val = a_array[i] + shift_float;
      FLOAT_TO_BITS(val, target);
      quantize_bits = round_bitwise(target, man_bits, rounding);
      BITS_TO_FLOAT(quantize_bits, quantized);
      quantized = quantized - shift_float;
    } else {
      quantize_bits = round_bitwise(target, man_bits, rounding);
      quantize_bits =
          clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
      BITS_TO_FLOAT(quantize_bits, quantized);
    }
    o_array[i] = quantized;
  }
  return o;
}

float float_quantize(float origin_float, int man_bits, int exp_bits,
                     Mode rounding, bool subnormal_support,
                     bool saturate = false) {
  unsigned int target, quantize_bits;
  FLOAT_TO_BITS(origin_float, target);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);
  if (subnormal && subnormal_support) {
    float shift_float, val;
    int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
    BITS_TO_FLOAT(shift_bits, shift_float);
    val = origin_float + shift_float;
    FLOAT_TO_BITS(val, target);
    quantize_bits = round_bitwise(target, man_bits, rounding);
    BITS_TO_FLOAT(quantize_bits, quantized);
    quantized = quantized - shift_float;
  } else {
    quantize_bits = round_bitwise(target, man_bits, rounding);
    quantize_bits =
        clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
    BITS_TO_FLOAT(quantize_bits, quantized);
  }
  return quantized;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixed_point_quantize_stochastic_mask",
        &fixed_point_quantize_stochastic_mask,
        "Fixed Point Number Stochastic Quantization with Mask (CPU)");
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic,
        "Fixed Point Number Stochastic Quantization (CPU)");
  m.def("block_quantize_stochastic", &block_quantize_stochastic,
        "Block Floating Point Number Stochastic Quantization (CPU)");
  m.def("float_quantize_stochastic", &float_quantize_stochastic,
        "Low-Bitwidth Floating Point Number Stochastic Quantization (CPU)");
  m.def("fixed_point_quantize_nearest_mask", &fixed_point_quantize_nearest_mask,
        "Fixed Point Number Nearest Quantization with Mask (CPU)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest,
        "Fixed Point Number Nearest Neighbor Quantization (CPU)");
  m.def("block_quantize_nearest", &block_quantize_nearest,
        "Block Floating Point Number Nearest Neighbor Quantization (CPU)");
  m.def(
      "float_quantize_nearest", &float_quantize_nearest,
      "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CPU)");
  m.def("float_quantize_nearest_mm", &float_quantize_nearest_mm,
        "Low-Bitwidth GEMM (CPU)");
  m.def("float_quantize_nearest_mm_fma", &float_quantize_nearest_mm_fma,
        "Low-Bitwidth GEMM (CPU)");
}
