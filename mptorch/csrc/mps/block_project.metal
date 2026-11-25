// block_project on an Apple GPU (common/block_project.h states the op; the
// entry point is block_entry.cpp's block_project_mps). It runs the step the
// CPU kernel runs (cpu/block_project.cpp), project_scale and project_code,
// so its codes are the CPU's word for word, stochastic rounding included.
//
// A kernel is compiled once per instantiation, after prelude.metal and the
// lines block_project_mps writes: mpt_storage_t (the dtype of x),
// MPT_ROUND_MODE, and mpt_block_cast(), the casts the projection runs, as
// block.metal's pack takes them. The layout, BlockFormatParams, is bound as
// a buffer, as for the pack.
//
// One thread takes a group of consecutive elements of a row whose codes fill
// whole bytes (project_group), as the CUDA kernel does, so that no two
// threads write one byte; the codes go into the bytes through a bit
// accumulator, low bits first, and past the row's last element they are
// zero, as block_pack leaves the end of its last block. Under SR, element
// i = (b * rows + r) * cols + c draws word i & 3 of Philox block i >> 2 of
// the CPU generator's seed, the CPU's layout. The division by a cast scale
// goes through block_decode.h's MPTORCH_BLOCK_DIV, metal_div_rn on the GPU,
// so the GPU's flush of binary32 subnormals reaches none of it.

#include "../common/block_project.h"

kernel void mpt_block_project(device const mpt_storage_t *x [[buffer(0)]],
                              device const uint8_t *scales [[buffer(1)]],
                              device uint8_t *data [[buffer(2)]],
                              device uint32_t *unstored [[buffer(3)]],
                              constant mptorch::block::BlockFormatParams &fmt [[buffer(4)]],
                              constant mptorch_mps::BlockProjectLaunch &L [[buffer(5)]],
                              uint t [[thread_position_in_grid]])
{
    using namespace mptorch::block;
    constexpr RoundMode RM = MPT_ROUND_MODE;
    if (uint64_t(t) >= L.groups)
        return;
    const BlockFormatParams p = fmt;
    const BlockCastT<float> c = mpt_block_cast();
    const int G = project_group(p.elem_bits);
    const int group_bytes = G * p.elem_bits / 8;
    const int64_t g = int64_t(t) / L.per_row;
    const int64_t gi = int64_t(t) - g * L.per_row;
    const int64_t b = g / L.rows;
    const int64_t r = g - b * L.rows;
    const int64_t c0 = gi * G;
    const int64_t blk = c0 >> p.log2_block_size;
    const uint32_t sc =
        p.scale_cols != 0 ? uint32_t(scales[(b * p.row_tiles + (r >> p.log2_block_rows)) * p.scale_cols + blk]) : 0u;
    const ProjectScale<float> s = project_scale<float>(sc, p);
    device const mpt_storage_t *xr = x + b * L.xs0 + r * L.xs1;
    const int64_t row_idx = g * L.cols;
    device uint8_t *dst = data + g * p.row_bytes + gi * group_bytes;
    uint32_t acc = 0; // the codes not yet written, low bits first
    int held = 0;     // how many bits of them
    PhiloxBlock draws;
    int64_t drawn = -1;
    bool bad = false;
    for (int e = 0; e < G; ++e)
    {
        const int64_t cc = c0 + e;
        uint32_t code = 0;
        if (cc < L.cols)
        {
            const float v = mptorch_mps::to_carrier(xr[cc * L.xs2]);
            uint32_t rv = 0;
            if constexpr (RM == RoundMode::SR)
            {
                const int64_t i = row_idx + cc;
                if ((i >> 2) != drawn)
                {
                    drawn = i >> 2;
                    draws = philox_block(L.seed, static_cast<uint64_t>(drawn), 0);
                }
                rv = draws.word(static_cast<int>(i & 3));
            }
            code = project_code<float, RM>(v, s, c.elem_signed, rv, p, c, bad);
        }
        acc |= code << held;
        held += p.elem_bits;
        while (held >= 8)
        {
            *dst++ = uint8_t(acc);
            acc >>= 8;
            held -= 8;
        }
    }
    if (bad)
        unstored[0] = 1u;
}
