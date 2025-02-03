#include "bit_helper.h"
#include "quant.h"
#include "binary8.h"
#include "softmax.h"
#include "layernorm.h"
#include <ATen/ATen.h>
#include <cmath>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

float cast_binary8_signed_nearest(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && (exp_val < min_exp)) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = (P == 1) ? round_bitwise_nearest_p1(uval32, man_bits - subnormal_shift)
                              : round_bitwise_nearest(uval32, man_bits - subnormal_shift);

    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_signed_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;
    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals) {
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits - 1)) - 1;
        int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

    uint32_t uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_signed_truncate(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = uval32 >> (23 - man_bits + subnormal_shift) << (23 - man_bits + subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_unsigned_nearest(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    if (origin_float < 0) return NAN;

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 9 - P;
    const int man_bits = P - 1;
    int subnormal_shift = 0;

    if (subnormals) {
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = (P == 1) - max_exp;

        if ((min_exp - exp_val) <= man_bits && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = (P == 1) ? round_bitwise_nearest_p1(uval32, man_bits - subnormal_shift)
                               : round_bitwise_nearest(uval32, man_bits - subnormal_shift);

    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_unsigned_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
    if (origin_float < 0) return NAN;

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 8 - P + 1;
    const int man_bits = P - 1;
    int subnormal_shift = 0;

    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

    uint32_t uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_unsigned_truncate(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    if (origin_float < 0) return NAN;

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    int subnormal_shift = 0;

    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = uval32 >> (23 - man_bits + subnormal_shift) << (23 - man_bits + subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}

void binary8_signed_nearest(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_signed_nearest(a[idx], P, overflow_policy, subnormals);
    }
}

void binary8_unsigned_nearest(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_unsigned_nearest(a[idx], P, overflow_policy, subnormals);
    }
}

void binary8_signed_stochastic(float *a, int *r, float *o, int size, int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_signed_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
    }
}

void binary8_unsigned_stochastic(float *a, int *r, float *o, int size, int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_unsigned_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
    }
}

void binary8_signed_truncate(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_signed_truncate(a[idx], P, overflow_policy, subnormals);
    }
}

void binary8_unsigned_truncate(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_unsigned_truncate(a[idx], P, overflow_policy, subnormals);
    }
}

void binary8_quantize_nearest_softmax_forward(Tensor a, Tensor o, int dim,
                                          int P_exp, OverflowPolicy op_exp, bool signed_exp,
                                          int P_off, OverflowPolicy op_off, bool signed_off,
                                          int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                          bool subnormals)
{
  auto sizes = partition_tensor(a, dim);
  softmax_forward(
    a.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [subnormals, P_exp, op_exp, signed_exp] (float x) {
      if (signed_exp) {
        return cast_binary8_signed_nearest(x, P_exp, op_exp, subnormals);
      }
      return cast_binary8_unsigned_nearest(x, P_exp, op_exp, subnormals);
    },
    [subnormals, P_off, op_off, signed_off] (float x) {
      if (signed_off) {
        return cast_binary8_signed_nearest(x, P_off, op_off, subnormals);
      }
      return cast_binary8_unsigned_nearest(x, P_off, op_off, subnormals);
    },
    [subnormals, P_acc, op_acc, signed_acc] (float x) {
      if (signed_acc) {
        return cast_binary8_signed_nearest(x, P_acc, op_acc, subnormals);
      }
      return cast_binary8_unsigned_nearest(x, P_acc, op_acc, subnormals);
    }
  );
}

void binary8_quantize_nearest_softmax_lse_forward(Tensor a, Tensor o, int dim,
                                            int P_off, OverflowPolicy op_off, bool signed_off,
                                            int P_lse, OverflowPolicy op_lse, bool signed_lse,
                                            bool subnormals)
{
  auto sizes = partition_tensor(a, dim);
  softmax_lse_forward(
    a.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [subnormals, P_off, op_off, signed_off] (float x) {
      if (signed_off) {
        return cast_binary8_signed_nearest(x, P_off, op_off, subnormals);
      }
      return cast_binary8_unsigned_nearest(x, P_off, op_off, subnormals);
    },
    [subnormals, P_lse, op_lse, signed_lse] (float x) {
      if (signed_lse) {
        return cast_binary8_signed_nearest(x, P_lse, op_lse, subnormals);
      }
      return cast_binary8_unsigned_nearest(x, P_lse, op_lse, subnormals);
    }
  );
}

void binary8_quantize_nearest_softmax_backward(Tensor a, Tensor g, Tensor o, int dim,
                                            int P_add, OverflowPolicy op_add, bool signed_add,
                                            int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                            bool subnormals)
{
  auto sizes = partition_tensor(a, dim);
  softmax_backward(
    a.data_ptr<float>(), g.data_ptr<float>(), o.data_ptr<float>(), sizes,
    [subnormals, P_add, op_add, signed_add] (float x) {
      if (signed_add) {
        return cast_binary8_signed_nearest(x, P_add, op_add, subnormals);
      }
      return cast_binary8_unsigned_nearest(x, P_add, op_add, subnormals);
    },
    [subnormals, P_mul, op_mul, signed_mul] (float x) {
      if (signed_mul) {
        return cast_binary8_signed_nearest(x, P_mul, op_mul, subnormals);
      }
      return cast_binary8_unsigned_nearest(x, P_mul, op_mul, subnormals);
    }
  );
}

void binary8_quantize_layernorm_forward(Tensor input, Tensor weight, Tensor bias,
                                    Tensor output, Tensor mean, Tensor rstd,
                                    float eps, std::vector<int> &dims,
                                    int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                    int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                    int P_div, OverflowPolicy op_div, bool signed_div,
                                    int P_sqrt, OverflowPolicy op_sqrt, bool signed_sqrt,
                                    bool subnormals)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_forward(
    input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
    output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), 
    eps, sizes,
    [subnormals, P_acc, op_acc, signed_acc] (float x) {
        if (signed_acc){
            return cast_binary8_signed_nearest(x, P_acc, op_acc, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_acc, op_acc, subnormals);
    },
    [subnormals, P_mul, op_mul, signed_mul] (float x) {
        if (signed_mul){
            return cast_binary8_signed_nearest(x, P_mul, op_mul, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_mul, op_mul, subnormals);
    },
    [subnormals, P_div, op_div, signed_div] (float x) {
        if (signed_div){
            return cast_binary8_signed_nearest(x, P_div, op_div, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_div, op_div, subnormals);
    },
    [subnormals, P_sqrt, op_sqrt, signed_sqrt] (float x) {
        if (signed_sqrt){
            return cast_binary8_signed_nearest(x, P_sqrt, op_sqrt, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_sqrt, op_sqrt, subnormals);
    }
  );
}

void binary8_quantize_layernorm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor bias, Tensor mean, Tensor rstd,
                                    Tensor grad_input, Tensor grad_weight, Tensor grad_bias,
                                    std::vector<int> &dims,
                                    int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                    int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                    int P_div, OverflowPolicy op_div, bool signed_div,
                                    bool subnormals)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_backward(
    input.data_ptr<float>(), grad_output.data_ptr<float>(), 
    weight.data_ptr<float>(), bias.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), 
    grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(), sizes,
    [subnormals, P_acc, op_acc, signed_acc] (float x) {
        if (signed_acc){
            return cast_binary8_signed_nearest(x, P_acc, op_acc, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_acc, op_acc, subnormals);
    },
    [subnormals, P_mul, op_mul, signed_mul] (float x) {
        if (signed_mul){
            return cast_binary8_signed_nearest(x, P_mul, op_mul, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_mul, op_mul, subnormals);
    },
    [subnormals, P_div, op_div, signed_div] (float x) {
        if (signed_div){
            return cast_binary8_signed_nearest(x, P_div, op_div, subnormals);
        }
        return cast_binary8_unsigned_nearest(x, P_div, op_div, subnormals);
    }
  );
}