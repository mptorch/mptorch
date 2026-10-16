// The elementwise block-format ops: block_pack, block_quant and block_quant_
// (mpt_block_pack) and block_unpack (mpt_block_unpack), over the one decode
// and encode every block kernel shares (common/block_decode.h), so that a
// result is the CPU kernels' (cpu/block_pack_tiles.h, block_pack_tiles_fos.h)
// word for word, stochastic rounding and the Four Over Six rules included.
//
// Compiled once per instantiation, after prelude.metal and a few lines the
// backend writes (block_entry.cpp) that define
//
//   mpt_storage_t     float, half or bfloat: the dtype of the tensor packed
//                     or quantized, or of the result unpacked
//   MPT_UNPACK        1 for mpt_block_unpack's library, 0 for mpt_block_pack's
//   MPT_ROUND_MODE    the RoundMode (pack only)
//   MPT_QUANT         1 for block_quant (decoded values out), 0 for
//                     block_pack (codes and scales out)
//   MPT_FOS           1 under a Four Over Six rule
//   mpt_block_cast()  the casts the pack runs (block_decode.h's BlockCastT),
//                     built from the descriptor's widths as literals, so that
//                     their constants fold as the GEMM's and the quantizers'
//                     do (gemm.metal says why that matters)
//
// The operand's layout, BlockFormatParams, is bound as a buffer: it holds
// the tensor's extents next to the format, which change with every shape.
//
// The pack runs one thread per tile, a tile being block_rows rows of one
// block, as the CPU runs one task per tile: an amax pass over the tile's
// words, its scale, under Four Over Six an error pass that quantizes the
// tile under both candidate scales and keeps the second if its error,
// counted in integers, is strictly the smaller, and then an encode pass, row
// by row, which reads element i before it writes it and touches no other, so
// the in-place quantizer is sound with x bound as o. A row's codes go into
// its block's bytes through a bit accumulator, low bits first, and a block's
// tail past the tensor's last column is zero codes, as the CPU's zeroed
// buffer leaves it. Element (b, r, c)'s SR draw is word i & 3 of Philox
// block i >> 2 of the CPU generator's seed, i = (b * rows + r) * cols + c:
// the CPU's layout (common/block_kernels.h).
//
// The decode's float operations go through block_decode.h's
// MPTORCH_BLOCK_MUL, _DIV and _SUB, which on Metal are bit_helper.h's
// word-exact spellings, so that the GPU's flush of binary32 subnormals
// reaches none of them: a tile whose amax is a subnormal, an E8M0 scale of
// 2^-127, a tensor scale that makes a scale's factor subnormal all pack and
// decode as on the CPU.

#if MPT_UNPACK

// One thread per element: x along the row, y the row of [batch * rows].
kernel void mpt_block_unpack(device const uint8_t *data [[buffer(0)]],
                             device const uint8_t *scales [[buffer(1)]],
                             device mpt_storage_t *o [[buffer(2)]],
                             constant mptorch::block::BlockFormatParams &fmt [[buffer(3)]],
                             constant mptorch_mps::BlockUnpackLaunch &L [[buffer(4)]],
                             uint2 t [[thread_position_in_grid]])
{
    using namespace mptorch::block;
    const int64_t cc = t.x, g = t.y;
    if (cc >= L.cols || g >= L.batch * L.rows)
        return;
    const BlockFormatParams p = fmt;
    const int64_t b = g / L.rows;
    const int64_t r = g - b * L.rows;
    const int64_t blk = cc >> p.log2_block_size;
    const uint32_t code =
        extract_code(data + g * p.row_bytes + blk * p.block_bytes, cc & (p.block_size - 1), p.elem_bits);
    const uint32_t sc =
        p.scale_cols != 0 ? uint32_t(scales[(b * p.row_tiles + (r >> p.log2_block_rows)) * p.scale_cols + blk]) : 0u;
    mptorch_mps::store_carrier(o, uint64_t(g * L.cols + cc), decode_value<float>(code, sc, p));
}

#else

namespace mptorch_mps
{

// Element i's SR word, from the Philox block that holds it; `drawn` is the
// block `draws` holds, which consecutive elements of a row share.
inline uint32_t block_draw(uint64_t seed, int64_t i, thread PhiloxBlock &draws, thread int64_t &drawn)
{
    if ((i >> 2) != drawn)
    {
        drawn = i >> 2;
        draws = philox_block(seed, static_cast<uint64_t>(drawn), 0);
    }
    return draws.word(static_cast<int>(i & 3));
}

} // namespace mptorch_mps

kernel void mpt_block_pack(device const mpt_storage_t *x [[buffer(0)]],
                           device mpt_storage_t *o [[buffer(1)]],
                           device uint8_t *data [[buffer(2)]],
                           device uint8_t *scales [[buffer(3)]],
                           constant mptorch::block::BlockFormatParams &fmt [[buffer(4)]],
                           constant mptorch_mps::BlockPackLaunch &L [[buffer(5)]],
                           uint tid [[thread_position_in_grid]])
{
    using namespace mptorch::block;
    using F = FloatTraits<float>;
    constexpr RoundMode RM = MPT_ROUND_MODE;
    const uint64_t t = tid;
    if (t >= L.tiles)
        return;
    const BlockFormatParams p = fmt;
    const BlockCastT<float> c = mpt_block_cast();
    const int64_t per_batch = p.row_tiles * p.n_blocks;
    const int64_t b = int64_t(t) / per_batch;
    const int64_t rem = int64_t(t) - b * per_batch;
    const int64_t rt = rem / p.n_blocks;
    const int64_t blk = rem - rt * p.n_blocks;
    const int64_t r0 = rt * p.block_rows, r1 = metal::min(r0 + int64_t(p.block_rows), L.rows);
    const int64_t c0 = blk * p.block_size, c1 = metal::min(c0 + int64_t(p.block_size), L.cols);
    device const mpt_storage_t *xb = x + b * L.xs0;

    // The largest magnitude, compared as words (a non-negative float's
    // order is its word's), with the NaNs left out and noted.
    uint32_t amax = 0;
    bool nan = false;
    for (int64_t r = r0; r < r1; ++r)
        for (int64_t cc = c0; cc < c1; ++cc)
        {
            const uint32_t w = block_word(mptorch_mps::to_carrier(xb[r * L.xs1 + cc * L.xs2])) & F::ABS_MASK;
            if (w > F::INF_BITS)
                nan = true;
            else if (w > amax)
                amax = w;
        }
    TileScale<float> s = tile_scale<float>(block_float<float>(amax), nan, p, c);

#if MPT_FOS
    // The tile under both candidate scales, and the second if its error is
    // strictly the smaller. The encode pass below casts again under the
    // scale chosen, with the same draws, which gives the codes kept here.
    const TileScale<float> low = four_over_six_low<float>(block_float<float>(amax), s, p, c);
    if (low.code != s.code)
    {
        const int amax_exp = block_floor_log2(block_float<float>(amax));
        uint64_t err_six = 0, err_low = 0;
        for (int64_t r = r0; r < r1; ++r)
        {
            const int64_t row_idx = (b * L.rows + r) * L.cols;
            PhiloxBlock draws;
            int64_t drawn = -1;
            for (int64_t cc = c0; cc < c1; ++cc)
            {
                const float v = mptorch_mps::to_carrier(xb[r * L.xs1 + cc * L.xs2]);
                uint32_t rv = 0;
                if constexpr (RM == RoundMode::SR)
                    rv = mptorch_mps::block_draw(L.seed, row_idx + cc, draws, drawn);
                uint32_t code;
                err_six = add_block_error(
                    err_six, candidate_error_units<float, RM>(v, s, amax_exp, c.elem_signed, rv, p, c, code),
                    c.scale_rounding);
                err_low = add_block_error(
                    err_low, candidate_error_units<float, RM>(v, low, amax_exp, c.elem_signed, rv, p, c, code),
                    c.scale_rounding);
            }
        }
        if (err_low < err_six)
            s = low;
    }
#endif

    if (!MPT_QUANT && p.scale_cols != 0)
        scales[(b * p.row_tiles + rt) * p.scale_cols + blk] = static_cast<uint8_t>(s.code);

    for (int64_t r = r0; r < r1; ++r)
    {
        const int64_t row_idx = (b * L.rows + r) * L.cols;
        device uint8_t *dst = data + (b * L.rows + r) * p.row_bytes + blk * p.block_bytes;
        PhiloxBlock draws;
        int64_t drawn = -1;
        uint32_t bits = 0; // the codes not yet written, low bits first
        int held = 0;      // how many bits of them
        int byte = 0;
        for (int64_t cc = c0; cc < c0 + p.block_size; ++cc)
        {
            uint32_t code = 0;
            if (cc < c1)
            {
                const float v = mptorch_mps::to_carrier(xb[r * L.xs1 + cc * L.xs2]);
                uint32_t rv = 0;
                if constexpr (RM == RoundMode::SR)
                    rv = mptorch_mps::block_draw(L.seed, row_idx + cc, draws, drawn);
                const float q = block_cast<float, RM>(scaled_elem(v, s, p), c.elem_signed, rv, c);
                code = encode_elem(v, q, p);
                if (MPT_QUANT)
                    mptorch_mps::store_carrier(o, uint64_t(b * L.os0 + r * L.os1 + cc * L.os2),
                                               decode_value<float>(code, s.code, p));
            }
            else if (MPT_QUANT)
                break;
            if (!MPT_QUANT)
            {
                bits |= code << held;
                held += p.elem_bits;
                while (held >= 8)
                {
                    dst[byte++] = static_cast<uint8_t>(bits);
                    bits >>= 8;
                    held -= 8;
                }
            }
        }
    }
}

#endif
