#include "../common/cast_binaryK.h"
#include "../quant_ops.h"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <climits>

using namespace at;

namespace
{

    void binaryK_kernel(float *__restrict__ a, float *o, int size,
                        int K, int P, int bias, bool is_signed,
                        RoundMode round_mode,
                        SaturationMode saturation_mode,
                        SubnormalsMode subnormals_mode)
    {
        int blockSize = 1024;
        int blockNums = (size + blockSize - 1) / blockSize;
        int man_bits, exp_bits;
        if (is_signed)
        {
            man_bits = P - 1;
            exp_bits = K - P;
        }
        else
        {
            man_bits = P - 1;
            exp_bits = K - P + 1;
        }

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        switch (round_mode)
        {
        case RoundMode::RNE:
            quant_kernel<<<blockNums, blockSize, 0, stream>>>(
                a, o, size, [man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode] __device__(float x)
                { return cast_binaryK_nearest_even(x, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode); });
            break;

        case RoundMode::RNA:
            quant_kernel<<<blockNums, blockSize, 0, stream>>>(
                a, o, size, [man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode] __device__(float x)
                { return cast_binaryK_nearest_away(x, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode); });
            break;

        case RoundMode::RU:
            quant_kernel<<<blockNums, blockSize, 0, stream>>>(
                a, o, size, [man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode] __device__(float x)
                { return cast_binaryK_up(x, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode); });
            break;

        case RoundMode::RD:
            quant_kernel<<<blockNums, blockSize, 0, stream>>>(
                a, o, size, [man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode] __device__(float x)
                { return cast_binaryK_down(x, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode); });
            break;

        default: // RZ
            quant_kernel<<<blockNums, blockSize, 0, stream>>>(
                a, o, size, [man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode] __device__(float x)
                { return cast_binaryK_zero(x, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode); });
            break;
        }
    }

    void binaryK_kernel(float *__restrict__ a,
                        int *__restrict__ r, float *o, int size,
                        int K, int P, int bias, int prng_bits, bool is_signed,
                        RoundMode round_mode, SaturationMode saturation_mode,
                        SubnormalsMode subnormals_mode)
    {
        int blockSize = 1024;
        int blockNums = (size + blockSize - 1) / blockSize;
        int man_bits, exp_bits;
        if (is_signed)
        {
            man_bits = P - 1;
            exp_bits = K - P;
        }
        else
        {
            man_bits = P - 1;
            exp_bits = K - P + 1;
        }

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        quant_kernel<<<blockNums, blockSize, 0, stream>>>(
            a, r, o, size,
            [man_bits, exp_bits, prng_bits, bias, is_signed, saturation_mode, subnormals_mode] __device__(float x, uint32_t rv)
            { return cast_binaryK_stochastic(x, rv, prng_bits, man_bits, exp_bits, bias, is_signed, saturation_mode, subnormals_mode); });
    }

} // namespace

Tensor binaryK_quantize_cuda(
    Tensor a, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed,
    int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
    auto o = zeros_like(a);
    int size = a.numel();
    RoundMode round_mode_ = static_cast<RoundMode>(round_mode);
    SubnormalsMode subnormals_mode_ = static_cast<SubnormalsMode>(subnormals_mode);
    SaturationMode saturation_mode_ = static_cast<SaturationMode>(saturation_mode);
    float *p_a = a.data_ptr<float>();
    float *p_o = o.data_ptr<float>();

    const int K_ = static_cast<int>(K);
    const int P_ = static_cast<int>(P);
    const int bias_ = static_cast<int>(bias);
    const int prng_bits_ = static_cast<int>(prng_bits);

    if (round_mode_ != RoundMode::SR)
    {
        binaryK_kernel(
            p_a, p_o, size, K_, P_, bias_, is_signed,
            round_mode_, saturation_mode_, subnormals_mode_);
    }
    else
    {
        auto rand_ints = randint_like(a, INT_MAX, device(a.device()).dtype(kInt));
        int *p_r = rand_ints.data_ptr<int>();
        binaryK_kernel(
            p_a, p_r, p_o, size, K_, P_, bias_, prng_bits_, is_signed,
            round_mode_, saturation_mode_, subnormals_mode_);
    }

    return o;
}
