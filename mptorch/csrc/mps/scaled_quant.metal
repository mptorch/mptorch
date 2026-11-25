// The scaled quantizers binaryK_quant_scaled, superfp_quant_scaled and
// fixedpoint_quant_scaled on an Apple GPU (common/scaled_quant_host.h):
// every element is multiplied by its scale, rounded to the format by
// quantize.metal's quantize_one, divided by the same scale and stored, in one
// pass, as the CPU kernel computes it (cpu/scaled_quant_kernel.cpp):
//
//     y[i] = storage(div(cast(mul(carrier(x[i]), s)), s)),
//
// each operation rounded to nearest in binary32, MPS's only carrier. The
// multiply and the divide go through bit_helper.h's metal_mul_rn and
// metal_div_rn, which give the CPU's result where a subnormal is involved:
// the GPU's own arithmetic flushes it ("The Apple GPU's flush of binary32
// subnormals"). So the result is the CPU's bit for bit.
//
// The tensor is a 2D tensor of rows and `scale` holds `per_row` scales for
// each row: element (r, c) takes scale[r * per_row + c / block]. The tail is
// quantize.metal's (quantize.cpp writes both), and so is the layout under
// RoundMode::SR: four consecutive elements a thread, element i taking word
// i & 3 of Philox block i >> 2 of the stream keyed on the CPU generator's
// seed, which is the CPU kernel's.

#include "quantize.metal"

kernel void mpt_quantize_scaled(device const mpt_storage_t *x [[buffer(0)]],
                                device const float *scale [[buffer(1)]],
                                device mpt_storage_t *y [[buffer(2)]],
                                constant mptorch_mps::ScaledQuantLaunch &q [[buffer(3)]],
                                uint t [[thread_position_in_grid]])
{
    const uint64_t first = (uint64_t)t * 4;
    PhiloxBlock block;
    if constexpr (MPT_ROUND_MODE == RoundMode::SR)
        block = philox_block(q.seed, t, 0);
    for (int l = 0; l < 4; ++l)
    {
        const uint64_t i = first + l;
        if (i >= q.n)
            return;
        const uint64_t row = i / q.cols;
        const uint64_t col = i - row * q.cols;
        const float s = scale[row * q.per_row + col / q.block];
        const float v = metal_mul_rn(mptorch_mps::to_carrier(x[i]), s);
        const float r = mptorch_mps::quantize_one(v, block.word(l));
        mptorch_mps::store_carrier(y, i, metal_div_rn(r, s));
    }
}
