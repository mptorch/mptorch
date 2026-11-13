// The elementwise block-format ops: block_pack, block_quant and block_quant_
// (mpt_block_pack for 1D blocks, mpt_block_pack_tile for 2D tiles) and
// block_unpack (mpt_block_unpack), over the one decode
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
//   MPT_TILE          1 for mpt_block_pack_tile's library, 0 for
//                     mpt_block_pack's (pack only)
//   mpt_block_cast()  the casts the pack runs (block_decode.h's BlockCastT),
//                     built from the descriptor's widths as literals, so that
//                     their constants fold as the GEMM's and the quantizers'
//                     do (gemm.metal says why that matters)
//
// The operand's layout, BlockFormatParams, is bound as a buffer: it holds
// the tensor's extents next to the format, which change with every shape.
//
// The pack of a tile, a tile being block_rows rows of one block, is an amax
// pass over the tile's words, its scale, under Four Over Six an error pass
// that quantizes the tile under both candidate scales and keeps the second
// if its error, counted in integers, is strictly the smaller, and then an
// encode pass, which reads element i before it writes it and touches no
// other, so the in-place quantizer is sound with x bound as o. A row's codes
// go into its block's bytes through a bit accumulator, low bits first, and a
// block's tail past the tensor's last column is zero codes, as the CPU's
// zeroed buffer leaves it. Either kernel packs any tile: mpt_block_pack
// gives a tile one thread, as the CPU gives it one task, and
// mpt_block_pack_tile gives it a threadgroup, which shares out the passes
// and reduces the amax and the errors across its threads. One thread runs
// its tile in series, 16,384 elements for a 128 x 128 tile, which under Four
// Over Six made the GPU slower than the CPU; the host picks the kernel from
// the tile's size and the number of tiles (block_entry.cpp's run_pack).
// Element (b, r, c)'s SR draw is word i & 3 of Philox
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

// Where tile t of the call is: its batch, its row tile and block, and its
// rows [r0, r1) and columns [c0, c1) clipped to the tensor's.
struct PackTile
{
    int64_t b, rt, blk, r0, r1, c0, c1;
};

inline PackTile pack_tile(uint64_t t, thread const mptorch::block::BlockFormatParams &p,
                          constant BlockPackLaunch &L)
{
    PackTile k;
    const int64_t per_batch = p.row_tiles * p.n_blocks;
    k.b = int64_t(t) / per_batch;
    const int64_t rem = int64_t(t) - k.b * per_batch;
    k.rt = rem / p.n_blocks;
    k.blk = rem - k.rt * p.n_blocks;
    k.r0 = k.rt * p.block_rows;
    k.r1 = metal::min(k.r0 + int64_t(p.block_rows), L.rows);
    k.c0 = k.blk * p.block_size;
    k.c1 = metal::min(k.c0 + int64_t(p.block_size), L.cols);
    return k;
}

// Each pass below covers one run of a tile's row, its columns c0 + [j0, j1):
// the whole block on one thread, or a threadgroup's share of it.

// The largest magnitude, compared as words (a non-negative float's order is
// its word's), with the NaNs left out and noted.
inline void amax_run(device const mpt_storage_t *x, constant BlockPackLaunch &L, thread const PackTile &k,
                     int64_t r, int j0, int j1, thread uint32_t &amax, thread bool &nan)
{
    using F = FloatTraits<float>;
    device const mpt_storage_t *xr = x + k.b * L.xs0 + r * L.xs1;
    for (int64_t cc = k.c0 + j0; cc < metal::min(k.c0 + j1, k.c1); ++cc)
    {
        const uint32_t w = mptorch::block::block_word(to_carrier(xr[cc * L.xs2])) & F::ABS_MASK;
        if (w > F::INF_BITS)
            nan = true;
        else if (w > amax)
            amax = w;
    }
}

// Four Over Six's errors for the run under both candidate scales, with the
// draws encode_run will give its elements, added to err_six and err_low.
template <RoundMode RM>
inline void error_run(device const mpt_storage_t *x, constant BlockPackLaunch &L,
                      thread const mptorch::block::BlockFormatParams &p,
                      thread const mptorch::block::BlockCastT<float> &c, thread const PackTile &k, int64_t r,
                      int j0, int j1, thread const mptorch::block::TileScale<float> &six,
                      thread const mptorch::block::TileScale<float> &low, int amax_exp, thread uint64_t &err_six,
                      thread uint64_t &err_low)
{
    using namespace mptorch::block;
    device const mpt_storage_t *xr = x + k.b * L.xs0 + r * L.xs1;
    const int64_t row_idx = (k.b * L.rows + r) * L.cols;
    PhiloxBlock draws;
    int64_t drawn = -1;
    for (int64_t cc = k.c0 + j0; cc < metal::min(k.c0 + j1, k.c1); ++cc)
    {
        const float v = to_carrier(xr[cc * L.xs2]);
        uint32_t rv = 0;
        if constexpr (RM == RoundMode::SR)
            rv = block_draw(L.seed, row_idx + cc, draws, drawn);
        uint32_t code; // encode_run casts again under the scale chosen
        err_six = add_block_error(
            err_six, candidate_error_units<float, RM>(v, six, amax_exp, c.elem_signed, rv, p, c, code),
            c.scale_rounding);
        err_low = add_block_error(
            err_low, candidate_error_units<float, RM>(v, low, amax_exp, c.elem_signed, rv, p, c, code),
            c.scale_rounding);
    }
}

// The run's codes under scale s, or under MPT_QUANT their decoded values.
// Reads element i before it writes it and touches no other, so the in-place
// quantizer is sound with x bound as o. The codes go into the block's bytes
// through a bit accumulator, low bits first, from the byte where code j0
// starts (j0 * elem_bits is a multiple of 8), and columns past the tensor's
// last are zero codes, as the CPU's zeroed buffer leaves them.
template <RoundMode RM>
inline void encode_run(device const mpt_storage_t *x, device mpt_storage_t *o, device uint8_t *data,
                       constant BlockPackLaunch &L, thread const mptorch::block::BlockFormatParams &p,
                       thread const mptorch::block::BlockCastT<float> &c, thread const PackTile &k, int64_t r,
                       int j0, int j1, thread const mptorch::block::TileScale<float> &s)
{
    using namespace mptorch::block;
    device const mpt_storage_t *xr = x + k.b * L.xs0 + r * L.xs1;
    const int64_t row_idx = (k.b * L.rows + r) * L.cols;
    device uint8_t *dst =
        data + (k.b * L.rows + r) * p.row_bytes + k.blk * p.block_bytes + (j0 * p.elem_bits) / 8;
    PhiloxBlock draws;
    int64_t drawn = -1;
    uint32_t bits = 0; // the codes not yet written, low bits first
    int held = 0;      // how many bits of them
    int byte = 0;
    for (int64_t cc = k.c0 + j0; cc < k.c0 + j1; ++cc)
    {
        uint32_t code = 0;
        if (cc < k.c1)
        {
            const float v = to_carrier(xr[cc * L.xs2]);
            uint32_t rv = 0;
            if constexpr (RM == RoundMode::SR)
                rv = block_draw(L.seed, row_idx + cc, draws, drawn);
            const float q = block_cast<float, RM>(scaled_elem(v, s, p), c.elem_signed, rv, c);
            code = encode_elem(v, q, p);
            if (MPT_QUANT)
                store_carrier(o, uint64_t(k.b * L.os0 + r * L.os1 + cc * L.os2),
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

} // namespace mptorch_mps

#if MPT_TILE

namespace mptorch_mps
{

// A partial error of Four Over Six merged over the simdgroup, by the rule's
// measure. A simdgroup shuffle moves 32-bit words, so the 64-bit sum moves as
// two.
inline uint64_t simd_merge_error(uint64_t e, int32_t rule, uint width)
{
    for (uint off = width / 2; off > 0; off >>= 1)
    {
        const uint2 w = as_type<uint2>(e);
        const uint2 other =
            uint2(metal::simd_shuffle_xor(w.x, ushort(off)), metal::simd_shuffle_xor(w.y, ushort(off)));
        e = mptorch::block::merge_block_error(e, as_type<uint64_t>(other), rule);
    }
    return e;
}

} // namespace mptorch_mps

// A tile by a threadgroup: tile t is threadgroup t, of the 32 to
// BLOCK_TILE_THREADS threads the host sizes it to (64 elements a thread).
// The passes are the one-thread kernel's, over runs of `run` columns of the
// tile's rows, run k taken by thread k mod the threadgroup's size. The amax
// is reduced by simdgroup and then through threadgroup memory, so every
// thread holds the tile's and computes the same scale, which makes the
// branch below uniform. Under Four Over Six the errors are reduced the same
// way; they are integers, so the order cannot matter and the choice is the
// one-thread kernel's and the CPU's. A run is four columns, or the codes
// that fill whole bytes if that is more (eight 5-bit codes), so each thread
// writes whole bytes of the block, and a run's SR draws come from one or two
// Philox blocks. Every thread reaches every barrier: the threadgroup is
// whole and no thread returns early. The barriers also make the in-place
// quantizer sound: every read of the amax and error passes is done before
// any thread writes.
[[max_total_threads_per_threadgroup(mptorch_mps::BLOCK_TILE_THREADS)]]
kernel void mpt_block_pack_tile(device const mpt_storage_t *x [[buffer(0)]],
                                device mpt_storage_t *o [[buffer(1)]],
                                device uint8_t *data [[buffer(2)]],
                                device uint8_t *scales [[buffer(3)]],
                                constant mptorch::block::BlockFormatParams &fmt [[buffer(4)]],
                                constant mptorch_mps::BlockPackLaunch &L [[buffer(5)]],
                                uint t [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint threads [[threads_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint width [[threads_per_simdgroup]],
                                uint sgs [[simdgroups_per_threadgroup]])
{
    using namespace mptorch::block;
    constexpr RoundMode RM = MPT_ROUND_MODE;
    threadgroup uint32_t sg_amax[32];
    threadgroup uint32_t sg_nan[32];
    const BlockFormatParams p = fmt;
    const BlockCastT<float> c = mpt_block_cast();
    const mptorch_mps::PackTile k = mptorch_mps::pack_tile(t, p, L);

    // Columns per run: four, or the codes that fill whole bytes (8 / gcd of
    // the width and 8), within the block.
    const int unit = 8 / metal::min(p.elem_bits & -p.elem_bits, 8);
    const int run = metal::min(metal::max(unit, 4), p.block_size);
    const int log2_runs = p.log2_block_size - int(metal::ctz(uint(run)));
    const int runs = int(k.r1 - k.r0) << log2_runs;

    uint32_t amax = 0;
    bool nan = false;
    for (int i = int(tid); i < runs; i += int(threads))
    {
        const int j0 = (i & ((1 << log2_runs) - 1)) * run;
        mptorch_mps::amax_run(x, L, k, k.r0 + (i >> log2_runs), j0, j0 + run, amax, nan);
    }
    amax = metal::simd_max(amax);
    const uint32_t any_nan = metal::simd_max(nan ? 1u : 0u);
    if (lane == 0)
    {
        sg_amax[sg] = amax;
        sg_nan[sg] = any_nan;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    amax = 0;
    nan = false;
    for (uint w = 0; w < sgs; ++w)
    {
        amax = metal::max(amax, sg_amax[w]);
        nan = nan || sg_nan[w] != 0;
    }
    TileScale<float> s = tile_scale<float>(block_float<float>(amax), nan, p, c);

#if MPT_FOS
    threadgroup uint64_t sg_err_six[32];
    threadgroup uint64_t sg_err_low[32];
    const TileScale<float> low = four_over_six_low<float>(block_float<float>(amax), s, p, c);
    if (low.code != s.code)
    {
        const int amax_exp = block_floor_log2(block_float<float>(amax));
        uint64_t err_six = 0, err_low = 0;
        for (int i = int(tid); i < runs; i += int(threads))
        {
            const int j0 = (i & ((1 << log2_runs) - 1)) * run;
            mptorch_mps::error_run<RM>(x, L, p, c, k, k.r0 + (i >> log2_runs), j0, j0 + run, s, low, amax_exp,
                                       err_six, err_low);
        }
        err_six = mptorch_mps::simd_merge_error(err_six, c.scale_rounding, width);
        err_low = mptorch_mps::simd_merge_error(err_low, c.scale_rounding, width);
        if (lane == 0)
        {
            sg_err_six[sg] = err_six;
            sg_err_low[sg] = err_low;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        err_six = 0;
        err_low = 0;
        for (uint w = 0; w < sgs; ++w)
        {
            err_six = merge_block_error(err_six, sg_err_six[w], c.scale_rounding);
            err_low = merge_block_error(err_low, sg_err_low[w], c.scale_rounding);
        }
        if (err_low < err_six)
            s = low;
    }
#endif

    if (!MPT_QUANT && tid == 0 && p.scale_cols != 0)
        scales[(k.b * p.row_tiles + k.rt) * p.scale_cols + k.blk] = static_cast<uint8_t>(s.code);
    for (int i = int(tid); i < runs; i += int(threads))
    {
        const int j0 = (i & ((1 << log2_runs) - 1)) * run;
        if (MPT_QUANT && k.c0 + j0 >= k.c1)
            continue;
        mptorch_mps::encode_run<RM>(x, o, data, L, p, c, k, k.r0 + (i >> log2_runs), j0, j0 + run, s);
    }
}

#else

// A tile by one thread.
kernel void mpt_block_pack(device const mpt_storage_t *x [[buffer(0)]],
                           device mpt_storage_t *o [[buffer(1)]],
                           device uint8_t *data [[buffer(2)]],
                           device uint8_t *scales [[buffer(3)]],
                           constant mptorch::block::BlockFormatParams &fmt [[buffer(4)]],
                           constant mptorch_mps::BlockPackLaunch &L [[buffer(5)]],
                           uint tid [[thread_position_in_grid]])
{
    using namespace mptorch::block;
    constexpr RoundMode RM = MPT_ROUND_MODE;
    const uint64_t t = tid;
    if (t >= L.tiles)
        return;
    const BlockFormatParams p = fmt;
    const BlockCastT<float> c = mpt_block_cast();
    const mptorch_mps::PackTile k = mptorch_mps::pack_tile(t, p, L);

    uint32_t amax = 0;
    bool nan = false;
    for (int64_t r = k.r0; r < k.r1; ++r)
        mptorch_mps::amax_run(x, L, k, r, 0, p.block_size, amax, nan);
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
        for (int64_t r = k.r0; r < k.r1; ++r)
            mptorch_mps::error_run<RM>(x, L, p, c, k, r, 0, p.block_size, s, low, amax_exp, err_six, err_low);
        if (err_low < err_six)
            s = low;
    }
#endif

    if (!MPT_QUANT && p.scale_cols != 0)
        scales[(k.b * p.row_tiles + k.rt) * p.scale_cols + k.blk] = static_cast<uint8_t>(s.code);
    for (int64_t r = k.r0; r < k.r1; ++r)
        mptorch_mps::encode_run<RM>(x, o, data, L, p, c, k, r, 0, p.block_size, s);
}

#endif

#endif
