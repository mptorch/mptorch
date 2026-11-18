// The GEMM kernel, which runs every GEMM-shaped op on an Apple GPU: the
// eight NAIVE GEMMs, their *_accumulated twins (KAHAN, BLOCK and TREE), the
// cross-family GEMMs, the conv ops, the block GEMM and, compiled after
// odd.metal, the odd ops. Each thread computes one element of C: it runs
// that element's whole sum over K through the same Accumulator policy the
// CPU and CUDA kernels use (common/gemm_policy.h, common/gemm_accumulate.h),
// in the same order and with the same casts. Under RoundMode::SR it also
// draws from the same Philox stream, keyed on the element's linear index
// into [batch, M, N] and seeded from the CPU generator
// (MpsBackend::make_context), so the result is bit-identical to the CPU op's.
//
// The kernel is compiled once per instantiation. Its source is prelude.metal,
// then a few lines the backend writes (the "tail": gemm_backend.cpp's
// tail_head and args_source), then this file. The tail defines
//
//   mpt_storage_t     float, half or bfloat: the dtype of C, and of the
//                     operands except in the block GEMM, whose operands are
//                     uint8 codes
//   MPT_ROUND_MODE    the RoundMode, a compile-time constant here as it is a
//                     template argument on the other backends
//                     (dispatch_round_mode)
//   mpt_args()        a function returning the op's Args struct
//                     (common/gemm_args.h) with every field a literal; for a
//                     conv op or the block GEMM, the Args of the mac it runs
//
// and, for an op beyond the eight NAIVE GEMMs, its variant:
//
//   MPT_ACCUMULATE    KAHAN, BLOCK or TREE: mpt_args() returns an
//                     AccumulateArgs (common/gemm_accumulate.h), and the
//                     policy it builds takes the positional contract below
//   MPT_GATHERED 1    a conv op (common/gemm_gather.h): the operand tiles are
//                     gathered from the convolution's tensors through the
//                     ConvGeom bound at buffer 5, and MPT_CONV_PASS names the
//                     pass, so that the compiler removes the gathers'
//                     switches on it
//   MPT_BLOCKED 1     the block GEMM (common/gemm_block.h): A and B are
//                     packed codes, decoded through per-threadgroup tables
//                     built from the BlockFormatParams bound at buffers 5 and
//                     6; the scales are at buffers 7 and 8, and their batch
//                     strides at 9
//
// The formats are literals, while the geometry and the block layouts are
// buffers: a layer's formats are fixed, but its geometry changes with every
// pass and shape, and a literal geometry would cost a compile per shape.
// The policies are built here, on the device, from mpt_args() and by the
// same with_accumulator the host backends call. Because the Args are
// literals, every format constant is known at compile time: the casts'
// masks, range bounds and fast-path gates become immediates, and a branch
// on a gate is removed rather than predicted. Reading the same Args from a
// buffer measured 1.7-1.8x slower (dev/benchmarks/mps_backend.py).
// Computing the constants at compile time also keeps them exact: the
// compiler evaluates them in IEEE arithmetic, whereas the GPU flushes a
// subnormal result to zero. (make_normal_range_params computes its one
// subnormal constant with integer operations on the bits, so it would be
// exact even if it ran on the GPU.)
//
// A mixed (palette) op builds each element's policy with the Args' with_slot
// rather than with_palette: with_slot selects the element's slot widths from
// the literal lists and builds one Mac from them. The CPU and CUDA kernels
// instead take a palette of Macs already built and select one per element;
// here that cost 1.5-2.9x the single-format kernel on the same formats. A
// split binaryK mac's cost grew by the same step with each slot, whatever
// the map (one uniform format, rows, columns, random), which is what a
// policy reselected at every use, rather than held in registers, costs.
// Built from the selected widths, the policy is computed once, and a mixed
// op costs 1.2-1.8x its single-format twin at 1024^3, about what reading
// the constants as run-time values rather than immediates costs.
//
// The tile. A threadgroup of GEMM_TILE x GEMM_TILE threads (launch_params.h)
// computes that block of C and walks K GEMM_TILE steps at a time: each
// thread copies one element of op(A)'s tile and one of op(B)'s into
// threadgroup memory, and then each reads its row of the first and its
// column of the second from there. Each element of A and B is thus loaded
// from device memory once per threadgroup instead of once per thread, which
// took an unquantized sum at 1024^3 from 7.6 to 3.5 ms. A quantized sum is
// limited by its casts, so the gain there is smaller (a fused binaryK mac
// 8.6 -> 6.9 ms, a split superfp mac about none). 16 x 16 matches or beats
// every other shape tried (8x32, 16x32, 32x32, K steps of 16 and 32), and is
// the CUDA kernel's block too. A tile whose K steps all lie inside K runs a
// loop of fixed length, which the compiler unrolls, and the last tile runs a
// loop over the steps that are left; the fixed loop is worth 2x on the
// unquantized sum. Stochastic rounding is the exception: its step computes
// a Philox block and two casts, and a second copy of that body costs more in
// instruction fetch than unrolling saves, so under SR every tile runs the
// one loop of variable length (60 ms at 1024^3 for a split binaryK mac,
// where the two loops took 86).
//
// KAHAN, BLOCK and TREE take the positional contract of
// common/gemm_accumulate.h: each step is told its position in the 16-step
// slab (pos = k & 15, which is the step's depth in the tile), and
// end_slab(k_end, K) is called after each tile, with k_end = k0 + 16 or K.
// No step past K runs, as on the other backends: a zero product is not free
// for KAHAN, which would apply its pending compensation, and under SR it
// would spend a draw and put the GPU out of step with the CPU's order. So
// the last tile runs the loop of variable length here too, and SR runs it
// for every tile. Under SR a fixed loop plus a remainder loop measured the
// same as the one loop, within 3% for every algorithm and mac at 1024^3
// (KAHAN 224 / 227 ms, BLOCK 89 / 89, TREE 99 / 100 for a split binaryK
// mac), so SR keeps the one loop, as it does for NAIVE.
//
// A conv op's tiles are the GEMM's tiles of operands that are never built:
// a thread's element of op(A) is its own output row's at K index k0 + l.x,
// and its element of op(B) is its own output column's at k0 + l.y (the
// dense stager's assignment when nothing is transposed). Each is read from
// the convolution's tensors at the offset gemm_gather.h computes, or is a
// zero where it falls in the padding. The parts of those offsets that
// depend only on the thread's row and column are computed once, before the
// loop over K. The result is stored at the result tensor's own column, which
// in the input gradient's per-class GEMMs is not the GEMM's column, and a
// palette's precision map is read there too. The block GEMM's tiles are
// filled in the same places, by decoding through the tables: two reads and
// a multiply per element (block_load_table), with A's and B's rows and
// columns those of an [M, K] and an [N, K] operand.
//
// What the GPU's arithmetic cannot match. The Apple GPU flushes binary32
// subnormals to zero, operands and results, in every Metal math mode. The
// casts themselves are immune: they round on the bits, and bit_helper.h
// spells the few operations they apply to the value so that the flush
// cannot reach them. The arithmetic between the casts is not: a product
// a*b, a running sum, or a fused a*b + c whose exact value is below 2^-126
// in magnitude, or whose operand is, becomes zero before the format sees
// it. Under RZ such a value rounds to zero on the CPU too, and under RNE
// and RNA it does whenever the format's smallest value is at least 2^-125,
// which holds for every SUBNORMALS format the binary32 carrier holds (a
// NORMALS or EXTENDED_NORMALS floor can sit at 2^-126, with its rounding
// boundary below it). Under RU, RD, RO and SR the CPU can round such a value
// to the format's smallest value where the GPU gives zero, and an
// unquantized sum (accumulate_quant=False) can keep a subnormal that the GPU
// drops. tests/test_mps.py pins both cases. The same holds for the
// arithmetic KAHAN, BLOCK and TREE add (the compensation, the block sums)
// and for the block GEMM's products and sums; the block decode itself is
// exact, since block_decode.h computes its multiply through
// MPTORCH_BLOCK_MUL, which the flush cannot reach. The odd ops are the
// exception: their carrier operations are common/carrier_odd_metal.h's,
// which are exact on the GPU too.

#ifndef MPT_GATHERED
#define MPT_GATHERED 0
#endif
#ifndef MPT_BLOCKED
#define MPT_BLOCKED 0
#endif
// The operands' element type: the block GEMM's are bytes of codes.
#if MPT_BLOCKED
#define MPT_OPERAND_T uint8_t
#else
#define MPT_OPERAND_T mpt_storage_t
#endif

// The layouts of the structs the host binds byte for byte (gemm_backend.cpp
// asserts the same).
static_assert(sizeof(mptorch::gemm::ConvGeom) == 336 && __builtin_offsetof(mptorch::gemm::ConvGeom, IN) == 96 &&
                  __builtin_offsetof(mptorch::gemm::ConvGeom, res_N) == 328,
              "ConvGeom must have one layout on the host and the GPU");
static_assert(sizeof(mptorch::block::BlockFormatParams) == 184 &&
                  __builtin_offsetof(mptorch::block::BlockFormatParams, tensor_scale) == 104 &&
                  __builtin_offsetof(mptorch::block::BlockFormatParams, trans) == 176,
              "BlockFormatParams must have one layout on the host and the GPU");


namespace mptorch_mps
{

// One output element: its coordinates, whether it is inside C (a tile at
// C's edge has threads past the edge, which copy their share of the tile and
// wait at its barriers but compute nothing), and its linear index into
// [batch, M, N], which keys its SR stream.
struct Element
{
    uint32_t i, j, bt;
    bool inside;
    uint64_t out;
};

// Whether an Accumulator takes common/gemm_accumulate.h's positional
// contract, that is, whether it has a `positional` member. On the host
// backends this is is_positional_accumulator_v, a C++20 requires-expression;
// MSL 3.1 is C++14, so here it is the older detection idiom.
template <class...>
struct Voider
{
    using type = void;
};

template <class Acc, class = void>
struct IsPositional
{
    static constexpr constant bool value = false;
};

template <class Acc>
struct IsPositional<Acc, typename Voider<decltype(Acc::positional)>::type>
{
    static constexpr constant bool value = true;
};

// One thread's share of copying an operand's tiles into threadgroup memory:
// the same position in every tile along K. With GEMM_TILE^2 threads and
// GEMM_TILE^2 elements in a tile, that is one element per tile.
struct Staged
{
    device const mpt_storage_t *src; // this thread's element of the next tile
    uint64_t step;                   // from one tile's element to the next's
    uint dst;                        // its place in the threadgroup's copy
    uint32_t k;                      // its position along K in the first tile
    bool live;                       // its row of op(A), or column of op(B), is in the operand
};

// op(X) is logically [rows, cols], and is stored as its transpose when
// `trans` is set. The tiles start at (r0, c0) and walk along K, which runs
// along op(X)'s columns for A and along its rows for B. Adjacent threads
// take adjacent elements of X's storage (along a row of the tile when X is
// stored as is, down a column when it is stored transposed), so that a SIMD
// group's loads read one contiguous run either way.
inline Staged stager(device const mpt_storage_t *X, uint64_t off, bool trans, uint32_t rows,
                     uint32_t cols, uint32_t r0, uint32_t c0, bool k_along_cols, uint2 l)
{
    const uint r = trans ? l.x : l.y, c = trans ? l.y : l.x;
    const uint32_t gr = r0 + r, gc = c0 + c;
    Staged s;
    s.src = X + off + (trans ? (uint64_t)gc * rows + gr : (uint64_t)gr * cols + gc);
    // The next tile is GEMM_TILE further along K: that many elements further
    // on in X's storage when K runs along the dimension X stores
    // contiguously, and that many rows of the storage further on when it runs
    // across it.
    const bool k_contiguous = k_along_cols != trans;
    s.step = k_contiguous ? GEMM_TILE : (uint64_t)GEMM_TILE * (trans ? rows : cols);
    s.dst = r * GEMM_TILE + c;
    s.k = k_along_cols ? gc : gr;
    s.live = k_along_cols ? gr < rows : gc < cols;
    return s;
}

// Copies this thread's element of the tile at K offset k0 into threadgroup
// memory, converted to the carrier, or a zero where the tile extends past
// the operand (that zero is never read: the loop over K stops at K, and no
// thread outside C accumulates), and advances to the next tile's element.
inline void stage(thread Staged &s, threadgroup float *tile, uint32_t k0, uint32_t K)
{
    tile[s.dst] = s.live && s.k + k0 < K ? to_carrier(*s.src) : 0.0f;
    s.src += s.step;
}

// The operand tiles of a dense GEMM: op(A)'s and op(B)'s elements, copied
// by the two Staged above.
struct DenseLoad
{
    Staged a, b;

    void stage(threadgroup float *As, threadgroup float *Bs, uint32_t k0, uint32_t K)
    {
        mptorch_mps::stage(a, As, k0, K);
        mptorch_mps::stage(b, Bs, k0, K);
    }
};

#if MPT_GATHERED
// The operand tiles of a conv pass (common/gemm_gather.h): this thread's
// element of op(A) is (its row, k0 + lx) and its element of op(B) is
// (k0 + ly, its column), read at the offsets the geometry gives; op(B)'s is
// a zero in the padding. `a_row` and `col` are the parts of those offsets
// that depend only on the thread's row and column.
struct GatherLoad
{
    device const mpt_storage_t *A;
    device const mpt_storage_t *B;
    const thread mptorch::gemm::ConvGeom *g;
    int64_t a_row;
    mptorch::gemm::ConvCol col;
    uint dst, lx, ly;
    bool row_live, col_live;

    void stage(threadgroup float *As, threadgroup float *Bs, uint32_t k0, uint32_t K)
    {
        const uint32_t ka = k0 + lx, kb = k0 + ly;
        As[dst] = row_live && ka < K
                      ? to_carrier(A[a_row + mptorch::gemm::conv_a_k(*g, static_cast<int32_t>(ka))])
                      : 0.0f;
        int64_t off = -1;
        if (col_live && kb < K)
            off = mptorch::gemm::conv_b_offset(*g, col, mptorch::gemm::conv_b_k(*g, static_cast<int32_t>(kb)));
        Bs[dst] = off >= 0 ? to_carrier(B[off]) : 0.0f;
    }
};
#endif

#if MPT_BLOCKED
// The operand tiles of the block GEMM (common/gemm_block.h): this thread's
// element of A, logically [M, K], is (its row, k0 + lx), and its element of
// B, logically [N, K], is (its column, k0 + ly), each decoded from its
// packed operand through the threadgroup's tables. Each operand records its
// own orientation (p.trans).
struct BlockLoad
{
    device const uint8_t *A;
    device const uint8_t *B;
    mptorch::gemm::BlockOperand oa, ob;
    threadgroup const float *ta;
    threadgroup const float *tb;
    int64_t bt, a_off, b_off; // the batch element, and its data offsets in bytes
    uint32_t i, j;
    uint dst, lx, ly;
    bool row_live, col_live;

    void stage(threadgroup float *As, threadgroup float *Bs, uint32_t k0, uint32_t K)
    {
        const uint32_t ka = k0 + lx, kb = k0 + ly;
        As[dst] = row_live && ka < K ? mptorch::gemm::block_load_table<float>(A, oa, ta, bt, a_off, i, ka) : 0.0f;
        Bs[dst] = col_live && kb < K ? mptorch::gemm::block_load_table<float>(B, ob, tb, bt, b_off, j, kb) : 0.0f;
    }
};
#endif

template <class Acc, class Load>
inline void gemm_tile(Acc acc, Element e, constant GemmLaunch &g, thread Load &load,
                      device mpt_storage_t *C, uint64_t store_at, threadgroup float *As,
                      threadgroup float *Bs, uint2 l)
{
    if (e.inside && g.use_rng)
        acc.seed_rng(g.seed, e.out);
    threadgroup const float *a = As + l.y * GEMM_TILE; // row i of op(A)'s tile
    threadgroup const float *b = Bs + l.x;             // column j of op(B)'s tile
    for (uint32_t k0 = 0; k0 < g.K; k0 += GEMM_TILE)
    {
        load.stage(As, Bs, k0, g.K);
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (e.inside)
        {
            if constexpr (IsPositional<Acc>::value)
            {
                const uint32_t n = metal::min((uint32_t)GEMM_TILE, g.K - k0);
                if constexpr (MPT_ROUND_MODE == RoundMode::SR)
                {
                    for (uint32_t k = 0; k < n; ++k)
                        acc.accumulate(a[k], b[k * GEMM_TILE], static_cast<int>(k));
                }
                else if (n == GEMM_TILE)
                {
                    for (uint32_t k = 0; k < GEMM_TILE; ++k)
                        acc.accumulate(a[k], b[k * GEMM_TILE], static_cast<int>(k));
                }
                else
                {
                    for (uint32_t k = 0; k < n; ++k)
                        acc.accumulate(a[k], b[k * GEMM_TILE], static_cast<int>(k));
                }
                acc.end_slab(static_cast<int64_t>(k0 + n), static_cast<int64_t>(g.K));
            }
            else if constexpr (MPT_ROUND_MODE == RoundMode::SR)
            {
                const uint32_t n = metal::min((uint32_t)GEMM_TILE, g.K - k0);
                for (uint32_t k = 0; k < n; ++k)
                    acc.accumulate(a[k], b[k * GEMM_TILE]);
            }
            else if (k0 + GEMM_TILE <= g.K)
            {
                for (uint32_t k = 0; k < GEMM_TILE; ++k)
                    acc.accumulate(a[k], b[k * GEMM_TILE]);
            }
            else
            {
                for (uint32_t k = 0; k < g.K - k0; ++k)
                    acc.accumulate(a[k], b[k * GEMM_TILE]);
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    if (e.inside)
        store_carrier(C, store_at, acc.finalize());
}

// gemm_tile with every argument but the policy bound, which the Args'
// factories call with the policy they build. It is a struct rather than a
// lambda because the library is compiled as MSL 3.1 (metal_runtime.mm's
// compile), which has no lambdas; Metal has them only from 3.2.
template <class Load>
struct TileRunner
{
    Element e;
    constant GemmLaunch *g;
    thread Load *load;
    device mpt_storage_t *C;
    uint64_t store_at;
    threadgroup float *As;
    threadgroup float *Bs;
    uint2 l;

    template <class Acc>
    void operator()(Acc acc) const
    {
        gemm_tile(acc, e, *g, *load, C, store_at, As, Bs, l);
    }
};

// Builds the element's policy from the op's Args and runs its tile: on a
// mixed op with with_slot, from the element's entry of the precision map
// (at `map_at`), and otherwise with with_accumulator, given the algorithm
// under MPT_ACCUMULATE. It is a template only so that `if constexpr`
// discards the branch that calls a member the Args type lacks.
template <class Args, class Load>
inline void gemm(const thread Args &args, constant GemmLaunch &g, thread Load &load, Element e,
                 uint64_t store_at, uint64_t map_at, device mpt_storage_t *C,
                 device const int32_t *prec_idx, threadgroup float *As, threadgroup float *Bs, uint2 l)
{
    TileRunner<Load> run{e, &g, &load, C, store_at, As, Bs, l};
    if constexpr (Args::mixed)
    {
        const int32_t idx = e.inside ? prec_idx[map_at] : 0;
        args.template with_slot<float, MPT_ROUND_MODE>(idx, run);
    }
    else
    {
#if defined(MPT_ACCUMULATE)
        args.template with_accumulator<float, MPT_ROUND_MODE, MPT_ACCUMULATE>(run);
#else
        args.template with_accumulator<float, MPT_ROUND_MODE>(run);
#endif
    }
}

} // namespace mptorch_mps

// The grid is (ceil(N / GEMM_TILE), ceil(M / GEMM_TILE), batch) threadgroups
// of GEMM_TILE x GEMM_TILE threads, so a SIMD group spans two rows of a tile
// and 16 adjacent columns. prec_idx is bound on every op (to C where there is
// no map) and read only on the mixed ones.
[[max_total_threads_per_threadgroup(mptorch_mps::GEMM_TILE * mptorch_mps::GEMM_TILE)]]
kernel void mpt_gemm(device const MPT_OPERAND_T *A [[buffer(0)]],
                     device const MPT_OPERAND_T *B [[buffer(1)]],
                     device mpt_storage_t *C [[buffer(2)]],
                     constant mptorch_mps::GemmLaunch &g [[buffer(3)]],
                     device const int32_t *prec_idx [[buffer(4)]],
#if MPT_GATHERED
                     constant mptorch::gemm::ConvGeom &geom [[buffer(5)]],
#elif MPT_BLOCKED
                     constant mptorch::block::BlockFormatParams &fmt_a [[buffer(5)]],
                     constant mptorch::block::BlockFormatParams &fmt_b [[buffer(6)]],
                     device const uint8_t *scales_a [[buffer(7)]],
                     device const uint8_t *scales_b [[buffer(8)]],
                     constant mptorch_mps::BlockGemmLaunch &bl [[buffer(9)]],
#endif
                     uint3 group [[threadgroup_position_in_grid]],
                     uint3 local [[thread_position_in_threadgroup]])
{
    using mptorch_mps::GEMM_TILE;
    threadgroup float As[GEMM_TILE * GEMM_TILE];
    threadgroup float Bs[GEMM_TILE * GEMM_TILE];
    const uint2 l = local.xy;
    mptorch_mps::Element e;
    e.i = group.y * GEMM_TILE + l.y;
    e.j = group.x * GEMM_TILE + l.x;
    e.bt = group.z;
    e.inside = e.i < g.M && e.j < g.N;
    e.out = ((uint64_t)e.bt * g.M + e.i) * g.N + e.j;
#if MPT_GATHERED
    // A copy of the geometry in thread memory, where gemm_gather.h's
    // functions take it, with the pass replaced by the compile-time
    // constant MPT_CONV_PASS.
    mptorch::gemm::ConvGeom cg = geom;
    cg.pass = MPT_CONV_PASS;
    int64_t b;
    int32_t grp;
    mptorch::gemm::conv_batch(cg, e.bt, b, grp);
    mptorch_mps::GatherLoad load;
    load.A = A;
    load.B = B;
    load.g = &cg;
    load.a_row = mptorch::gemm::conv_a_row(cg, grp, e.i);
    load.col = mptorch::gemm::conv_b_col(cg, b, grp, static_cast<int32_t>(e.j));
    load.dst = l.y * GEMM_TILE + l.x;
    load.lx = l.x;
    load.ly = l.y;
    load.row_live = e.i < g.M;
    load.col_live = e.j < g.N;
    const int64_t res_col = mptorch::gemm::conv_result_col(cg, static_cast<int32_t>(e.j));
    const uint64_t store_at = ((uint64_t)e.bt * g.M + e.i) * (uint64_t)cg.res_N + (uint64_t)res_col;
    const uint64_t map_at = (uint64_t)b * g.idx_batch_stride +
                            (uint64_t)mptorch::gemm::conv_result_row(grp, g.M, e.i) * g.idx_row_stride +
                            (uint64_t)res_col * g.idx_col_stride;
#elif MPT_BLOCKED
    // Each operand's decode tables; each thread fills one entry of each.
    static_assert(GEMM_TILE * GEMM_TILE == mptorch::gemm::BLOCK_TABLE, "one table entry per thread");
    threadgroup float tables[4 * mptorch::gemm::BLOCK_TABLE];
    mptorch_mps::BlockLoad load;
    load.oa.p = fmt_a;
    load.oa.scales = scales_a;
    load.oa.scale_stride = bl.scale_stride_a;
    load.ob.p = fmt_b;
    load.ob.scales = scales_b;
    load.ob.scale_stride = bl.scale_stride_b;
    const int t = static_cast<int>(l.y * GEMM_TILE + l.x);
    mptorch::gemm::fill_block_table<float>(tables, t, load.oa.p);
    mptorch::gemm::fill_block_table<float>(tables + 2 * mptorch::gemm::BLOCK_TABLE, t, load.ob.p);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    load.A = A;
    load.B = B;
    load.ta = tables;
    load.tb = tables + 2 * mptorch::gemm::BLOCK_TABLE;
    load.bt = e.bt;
    load.a_off = (int64_t)e.bt * (int64_t)g.stride_a;
    load.b_off = (int64_t)e.bt * (int64_t)g.stride_b;
    load.i = e.i;
    load.j = e.j;
    load.dst = l.y * GEMM_TILE + l.x;
    load.lx = l.x;
    load.ly = l.y;
    load.row_live = e.i < g.M;
    load.col_live = e.j < g.N;
    const uint64_t store_at = e.out;
    const uint64_t map_at = 0;
#else
    mptorch_mps::DenseLoad load;
    load.a = mptorch_mps::stager(A, (uint64_t)e.bt * g.stride_a, g.trans_a, g.M, g.K, group.y * GEMM_TILE, 0,
                                 true, l);
    load.b = mptorch_mps::stager(B, (uint64_t)e.bt * g.stride_b, g.trans_b, g.K, g.N, 0, group.x * GEMM_TILE,
                                 false, l);
    const uint64_t store_at = e.out;
    const uint64_t map_at = e.bt * g.idx_batch_stride + e.i * g.idx_row_stride + e.j * g.idx_col_stride;
#endif
    mptorch_mps::gemm(mpt_args(), g, load, e, store_at, map_at, C, prec_idx, As, Bs, l);
}
