#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#define HALF_TO_BITS(x) (*reinterpret_cast<uint16_t *>(&(x)))
#define BITS_TO_HALF(x) (*reinterpret_cast<__half *>(&(x)))

#define BFLOAT16_TO_BITS(x) (*reinterpret_cast<uint16_t *>(&(x)))
#define BITS_TO_BFLOAT16(x) (*reinterpret_cast<__nv_bfloat16 *>(&(x)))

__host__ __device__ __forceinline__ uint16_t round_bitwise_nearest_even_bf16(uint16_t target, int man_bits)
{
    if (man_bits >= 7)
        return target;
    uint16_t mask = (1 << (7 - man_bits)) - 1;
    uint16_t tie = 1 << (6 - man_bits);
    uint16_t add_r = target + tie;
    uint16_t quantized = add_r & ~mask;
    uint16_t is_tie = (target & mask) == tie;
    uint16_t odd = (man_bits == 0) ? 0 : 1;
    return quantized & ~((is_tie & odd) << (7 - man_bits));
}

__host__ __device__ __forceinline__ uint16_t round_bitwise_nearest_even_fp16(uint16_t target, int man_bits)
{
    if (man_bits >= 10)
        return target;
    uint16_t mask = (1 << (10 - man_bits)) - 1;
    uint16_t tie = 1 << (9 - man_bits);
    uint16_t add_r = target + tie;
    uint16_t quantized = add_r & ~mask;
    uint16_t is_tie = (target & mask) == tie;
    uint16_t odd = (man_bits == 0) ? 0 : 1;
    return quantized & ~((is_tie & odd) << (10 - man_bits));
}
