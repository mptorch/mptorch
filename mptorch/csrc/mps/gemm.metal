// The GEMM kernel, for all eight GEMM ops: one thread per output element,
// which runs that element's whole K-reduction through the Accumulator the
// CPU and CUDA kernels use (common/gemm_policy.h), in the same order, with
// the same casts and, under RoundMode::SR, the same Philox stream: keyed on
// the element's global linear index into [batch, M, N] and seeded from the
// CPU generator's draw (mps/gemm_backend.cpp), which is what makes a result
// bit-identical to the CPU op's.
//
// Compiled once per instantiation, after prelude.metal and a few lines the
// backend writes (gemm_backend.cpp's launch_source) that define
//
//   mpt_storage_t     float, half or bfloat: the operands' and C's dtype
//   MPT_ROUND_MODE    the RoundMode, as the template argument it is on the
//                     other backends (gemm_dtype.h's dispatch_round_mode)
//   mpt_args()        the op's Args (common/gemm_args.h), as literals
//
// The policies are built here, on the device, from mpt_args() and by the
// same with_accumulator the host backends call. Because the Args are
// literals, every format constant folds into the kernel: the casts' masks,
// their range bounds and their fast-path gates are immediates, and a branch
// on a gate is gone rather than predicted. Reading the same policy from a
// buffer measured 1.7-1.8x slower (dev/benchmarks/mps_backend.py). Folding
// is also what keeps the constants exact: the compiler folds in IEEE
// arithmetic, where the GPU flushes a subnormal result (bit_helper.h's
// make_normal_range_params computes its one subnormal constant on the word,
// so it would not depend on this even unfolded).
//
// The mixed ops pick each element's formats with the Args' with_slot rather
// than with_palette: the element's slot widths are selected from the
// literal lists, and one Mac is built from them. A palette of built Macs,
// selected from, is what the CPU and CUDA kernels take, and here it cost
// 1.5-2.9x the single-format kernel on the same formats. A split binaryK
// mac's cost grew by the same step with each slot, whatever the map (one
// uniform format, rows, columns, random), which is a policy re-selected at
// its every use rather than held in registers. Built from selected widths, it is computed once,
// and a mixed op costs 1.2-1.8x its single-format twin at 1024^3, about
// what reading the constants as run-time values rather than immediates
// costs.
//
// The tile. A threadgroup of GEMM_TILE x GEMM_TILE threads (launch_params.h)
// computes that block of C, and walks K GEMM_TILE steps at a time: each
// thread copies one element of op(A)'s tile and one of op(B)'s into
// threadgroup memory, and each then reads its row of the first and its
// column of the second from there. Each element of A and B is loaded from
// device memory once per threadgroup instead of once per thread, which took
// an unquantized sum at 1024^3 from 7.6 to 3.5 ms; a quantized one is bound
// by its casts, so the gain there is smaller (binaryK fma 8.6 -> 6.9 ms, a
// split superfp mac about none). 16 x 16 matches or beats every other shape
// tried (8x32, 16x32, 32x32, K steps of 16 and 32), and is the CUDA kernel's
// block too. A tile whose K-steps are all inside K runs a loop of fixed
// length, which the compiler unrolls, and the last tile a loop of what is
// left; the fixed loop is worth 2x on the unquantized sum. Stochastic
// rounding is the exception: its step carries a Philox block and two casts,
// and a second copy of that body costs more in instruction fetch than the
// unrolling saves, so SR runs the one loop of variable length for every tile
// (60 ms at 1024^3 for a split binaryK mac, where two loops took 86).
//
// What the GPU's arithmetic cannot match. The Apple GPU flushes binary32
// subnormals to zero, operands and results, in every Metal math mode. The
// casts themselves are immune (they round on the word, and bit_helper.h
// spells the few operations they apply to the value so the flush cannot
// reach them), but the arithmetic between them is not: a product a*b, a
// running sum, or a fused a*b + c whose exact value is below 2^-126 in
// magnitude, or whose operand is, becomes a zero before the format sees it. Under RZ such a value rounds
// to zero on the CPU too, and under RNE and RNA it does whenever the format's
// smallest value is at least 2^-125, which is every SUBNORMALS format the
// binary32 carrier holds (a NORMALS or EXTENDED_NORMALS floor can sit at
// 2^-126, with its rounding boundary below it). Under RU, RD, RO and SR the
// CPU can round it to the format's smallest value where the GPU gives zero,
// and an unquantized sum (accumulate_quant=False) can keep a subnormal the
// GPU drops. tests/test_mps.py pins both halves of that statement.

namespace mptorch_mps
{

// One output element: its coordinates, whether it is inside C (a tile at
// C's edge has threads past it, which stage their share of the tile and
// wait at its barriers but compute nothing), and its linear index into
// [batch, M, N].
struct Element
{
    uint32_t i, j, bt;
    bool inside;
    uint64_t out;
};

// One thread's share of staging an operand's tiles: the same element of
// every tile along K, which is one element per tile at GEMM_TILE^2 threads
// and a GEMM_TILE^2 tile.
struct Staged
{
    device const mpt_storage_t *src; // this thread's element of the next tile
    uint64_t step;                   // from one tile's element to the next's
    uint dst;                        // its place in the threadgroup's copy
    uint32_t k;                      // its position along K in the first tile
    bool live;                       // its row of op(A), or column of op(B), is in the operand
};

// op(X) is logically [rows, cols], and stored as its transpose when
// `trans`; the tiles start at (r0, c0) and walk along K, which is op(X)'s
// columns for A and its rows for B. Adjacent threads take adjacent elements
// of X's storage, along a row of the tile when X is stored as is and down a
// column when it is stored transposed, so that a SIMD group's loads are one
// contiguous run either way.
inline Staged stager(device const mpt_storage_t *X, uint64_t off, bool trans, uint32_t rows,
                     uint32_t cols, uint32_t r0, uint32_t c0, bool k_along_cols, uint2 l)
{
    const uint r = trans ? l.x : l.y, c = trans ? l.y : l.x;
    const uint32_t gr = r0 + r, gc = c0 + c;
    Staged s;
    s.src = X + off + (trans ? (uint64_t)gc * rows + gr : (uint64_t)gr * cols + gc);
    // The next tile is GEMM_TILE further along K: that many elements on in
    // X's storage when K runs along the dimension X stores contiguously, and
    // that many rows of the storage on when it runs across it.
    const bool k_contiguous = k_along_cols != trans;
    s.step = k_contiguous ? GEMM_TILE : (uint64_t)GEMM_TILE * (trans ? rows : cols);
    s.dst = r * GEMM_TILE + c;
    s.k = k_along_cols ? gc : gr;
    s.live = k_along_cols ? gr < rows : gc < cols;
    return s;
}

// Copies this thread's element of the tile at K-offset k0, converted to the
// carrier, or a zero where the tile hangs past the operand (never read: the
// K-loop stops at K, and no thread past C accumulates), and moves on to the
// next tile's.
inline void stage(thread Staged &s, threadgroup float *tile, uint32_t k0, uint32_t K)
{
    tile[s.dst] = s.live && s.k + k0 < K ? to_carrier(*s.src) : 0.0f;
    s.src += s.step;
}

template <class Acc>
inline void gemm_tile(Acc acc, Element e, constant GemmLaunch &g, device const mpt_storage_t *A,
                      device const mpt_storage_t *B, device mpt_storage_t *C,
                      threadgroup float *As, threadgroup float *Bs, uint3 group, uint2 l)
{
    if (e.inside && g.use_rng)
        acc.seed_rng(g.seed, e.out);
    Staged sa = stager(A, (uint64_t)e.bt * g.stride_a, g.trans_a, g.M, g.K,
                       group.y * GEMM_TILE, 0, true, l);
    Staged sb = stager(B, (uint64_t)e.bt * g.stride_b, g.trans_b, g.K, g.N, 0,
                       group.x * GEMM_TILE, false, l);
    threadgroup const float *a = As + l.y * GEMM_TILE; // row i of op(A)'s tile
    threadgroup const float *b = Bs + l.x;             // column j of op(B)'s tile
    for (uint32_t k0 = 0; k0 < g.K; k0 += GEMM_TILE)
    {
        stage(sa, As, k0, g.K);
        stage(sb, Bs, k0, g.K);
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (e.inside)
        {
            if constexpr (MPT_ROUND_MODE == RoundMode::SR)
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
        C[e.out] = from_carrier<mpt_storage_t>(acc.finalize());
}

// gemm_tile with everything but the policy bound, for the Args' factories
// to call with the policy they build. A struct rather than a lambda because
// the library is MSL 3.1 (metal_runtime.mm), and Metal has lambdas only
// from 3.2.
struct TileRunner
{
    Element e;
    constant GemmLaunch *g;
    device const mpt_storage_t *A;
    device const mpt_storage_t *B;
    device mpt_storage_t *C;
    threadgroup float *As;
    threadgroup float *Bs;
    uint3 group;
    uint2 l;

    template <class Acc>
    void operator()(Acc acc) const
    {
        gemm_tile(acc, e, *g, A, B, C, As, Bs, group, l);
    }
};

// Builds the element's policy from the op's Args and runs its tile. A
// template only so that `if constexpr` discards the branch the Args type
// has no member for.
template <class Args>
inline void gemm(const thread Args &args, constant GemmLaunch &g, device const mpt_storage_t *A,
                 device const mpt_storage_t *B, device mpt_storage_t *C,
                 device const int32_t *prec_idx, threadgroup float *As, threadgroup float *Bs,
                 uint3 group, uint2 l)
{
    Element e;
    e.i = group.y * GEMM_TILE + l.y;
    e.j = group.x * GEMM_TILE + l.x;
    e.bt = group.z;
    e.inside = e.i < g.M && e.j < g.N;
    e.out = ((uint64_t)e.bt * g.M + e.i) * g.N + e.j;
    TileRunner run{e, &g, A, B, C, As, Bs, group, l};
    if constexpr (Args::mixed)
    {
        const int32_t idx = e.inside ? prec_idx[e.bt * g.idx_batch_stride + e.i * g.idx_row_stride +
                                                e.j * g.idx_col_stride]
                                     : 0;
        args.template with_slot<float, MPT_ROUND_MODE>(idx, run);
    }
    else
        args.template with_accumulator<float, MPT_ROUND_MODE>(run);
}

} // namespace mptorch_mps

// The grid is (ceil(N / GEMM_TILE), ceil(M / GEMM_TILE), batch) threadgroups
// of GEMM_TILE x GEMM_TILE threads, so a SIMD group spans two rows of a tile
// and 16 adjacent columns. prec_idx is bound on every op (to C where there is
// no map) and read only on the mixed ones.
[[max_total_threads_per_threadgroup(mptorch_mps::GEMM_TILE * mptorch_mps::GEMM_TILE)]]
kernel void mpt_gemm(device const mpt_storage_t *A [[buffer(0)]],
                     device const mpt_storage_t *B [[buffer(1)]],
                     device mpt_storage_t *C [[buffer(2)]],
                     constant mptorch_mps::GemmLaunch &g [[buffer(3)]],
                     device const int32_t *prec_idx [[buffer(4)]],
                     uint3 group [[threadgroup_position_in_grid]],
                     uint3 local [[thread_position_in_threadgroup]])
{
    threadgroup float As[mptorch_mps::GEMM_TILE * mptorch_mps::GEMM_TILE];
    threadgroup float Bs[mptorch_mps::GEMM_TILE * mptorch_mps::GEMM_TILE];
    mptorch_mps::gemm(mpt_args(), g, A, B, C, prec_idx, As, Bs, group, local.xy);
}
