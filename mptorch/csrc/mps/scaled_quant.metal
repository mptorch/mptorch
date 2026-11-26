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
//
// Integer division is costly on an Apple GPU, which has no divider and
// emulates 64-bit integers. So a thread divides only for its first element's
// row and block, and steps along the row for the next ones, and the indices
// are mpt_index_t, which the tail defines: uint32_t when every index of the
// tensor fits (quantize.cpp), uint64_t otherwise. With two 64-bit divisions
// per element instead, the kernel took 3.1x (by rows) and 4.5x (by blocks
// of 32) its time for one scale, at 4096 x 4096 on an M3 Pro, and 4.7x its
// present time by blocks.

#include "quantize.metal"

kernel void mpt_quantize_scaled(device const mpt_storage_t *x [[buffer(0)]],
                                device const float *scale [[buffer(1)]],
                                device mpt_storage_t *y [[buffer(2)]],
                                constant mptorch_mps::ScaledQuantLaunch &q [[buffer(3)]],
                                uint t [[thread_position_in_grid]])
{
    const mpt_index_t n = mpt_index_t(q.n), cols = mpt_index_t(q.cols), width = mpt_index_t(q.block),
                      per_row = mpt_index_t(q.per_row);
    const mpt_index_t first = mpt_index_t(t) * 4;
    if (first >= n)
        return;
    PhiloxBlock block;
    if constexpr (MPT_ROUND_MODE == RoundMode::SR)
        block = philox_block(q.seed, t, 0);
    // The element's row and column, the column's block, and its place there.
    mpt_index_t row = first / cols;
    mpt_index_t col = first - row * cols;
    mpt_index_t blk = col / width;
    mpt_index_t in_blk = col - blk * width;
    for (int l = 0; l < 4; ++l)
    {
        const mpt_index_t i = first + l;
        if (i >= n)
            return;
        const float s = scale[row * per_row + blk];
        const float v = metal_mul_rn(mptorch_mps::to_carrier(x[i]), s);
        const float r = mptorch_mps::quantize_one(v, block.word(l));
        mptorch_mps::store_carrier(y, i, metal_div_rn(r, s));
        if (++in_blk == width)
        {
            in_blk = 0;
            ++blk;
        }
        if (++col == cols)
        {
            col = 0;
            ++row;
            blk = 0;
            in_blk = 0;
        }
    }
}
