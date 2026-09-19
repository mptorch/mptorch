// The GEMM kernel, for all eight GEMM ops: one thread per output element,
// which runs that element's whole K-reduction through the Accumulator the
// CPU and CUDA kernels use (common/gemm_policy.h), in the same order, with
// the same casts and, under RoundMode::SR, the same Philox stream: keyed on
// the element's global linear index into [batch, M, N] and seeded from the
// CPU generator's draw (mps/gemm_backend.mm), which is what makes a result
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
// same with_accumulator / with_palette the host backends call. Because the
// Args are literals, every format constant folds into the kernel: the casts'
// masks, their range bounds and their fast-path gates are immediates, and a
// branch on a gate is gone rather than predicted. Reading the same policy
// from a buffer measured 1.7-1.8x slower (dev/benchmarks/mps_backend.py).
// Folding is also what keeps the constants exact: the compiler folds in IEEE
// arithmetic, where the GPU flushes a subnormal result (bit_helper.h's
// make_normal_range_params computes its one subnormal constant on the word,
// so it would not depend on this even unfolded).
//
// What the GPU's arithmetic cannot match. The Apple GPU flushes binary32
// subnormals to zero, operands and results, in every Metal math mode. The
// casts themselves are immune (they round on the word, and bit_helper.h
// spells the few operations they apply to the value so the flush cannot
// reach them), but the arithmetic between them is not: a product a*b, a running sum, or a fused
// a*b + c whose exact value is below 2^-126 in magnitude, or whose operand
// is, becomes a zero before the format sees it. Under RZ such a value rounds
// to zero on the CPU too, and under RNE and RNA it does whenever the format's
// smallest value is at least 2^-125, which is every SUBNORMALS format the
// binary32 carrier holds (a NORMALS or EXTENDED_NORMALS floor can sit at
// 2^-126, with its rounding boundary below it). Under RU, RD, RO and SR the
// CPU can round it to the format's smallest value where the GPU gives zero,
// and an unquantized sum (accumulate_quant=False) can keep a subnormal the
// GPU drops. tests/test_mps.py pins both halves of that statement.

namespace mptorch_mps
{

// One output element. `pal` is the palette on the mixed ops and an empty
// NoPalette on the others, whose instantiation never reads it.
template <bool MIXED, class Acc, class Pal>
inline void gemm_element(Acc acc, const thread Pal &pal, constant GemmLaunch &g,
                         device const mpt_storage_t *A, device const mpt_storage_t *B,
                         device mpt_storage_t *C, device const int32_t *prec_idx, uint3 gid)
{
    const uint32_t j = gid.x, i = gid.y, bt = gid.z;
    if (i >= g.M || j >= g.N || bt >= g.batch)
        return;
    const uint64_t out = ((uint64_t)bt * g.M + i) * g.N + j;
    if (g.use_rng)
        acc.seed_rng(g.seed, out);
    if constexpr (MIXED)
        acc.mac = pal.slot(prec_idx[bt * g.idx_batch_stride + i * g.idx_row_stride + j * g.idx_col_stride]);

    // op(A)[i, k] and op(B)[k, j] walked along k; the trans flags only
    // change the starting point and the step, as in the other kernels.
    device const mpt_storage_t *a = A + bt * g.stride_a + (g.trans_a ? (uint64_t)i : (uint64_t)i * g.K);
    device const mpt_storage_t *b = B + bt * g.stride_b + (g.trans_b ? (uint64_t)j * g.K : (uint64_t)j);
    const uint64_t step_a = g.trans_a ? g.M : 1, step_b = g.trans_b ? 1 : g.N;
    for (uint32_t k = 0; k < g.K; ++k, a += step_a, b += step_b)
        acc.accumulate(to_carrier(*a), to_carrier(*b));
    C[out] = from_carrier<mpt_storage_t>(acc.finalize());
}

// Builds the op's policies from its Args and runs the element. A template
// only so that `if constexpr` discards the branch the Args type has no
// member for.
template <class Args>
inline void gemm(const thread Args &args, constant GemmLaunch &g, device const mpt_storage_t *A,
                 device const mpt_storage_t *B, device mpt_storage_t *C,
                 device const int32_t *prec_idx, uint3 gid)
{
    if constexpr (Args::mixed)
        args.template with_palette<float, MPT_ROUND_MODE>(
            [&](auto acc, const thread auto &pal)
            { gemm_element<true>(acc, pal, g, A, B, C, prec_idx, gid); });
    else
        args.template with_accumulator<float, MPT_ROUND_MODE>(
            [&](auto acc) { gemm_element<false>(acc, NoPalette{}, g, A, B, C, prec_idx, gid); });
}

} // namespace mptorch_mps

// The grid is (N, M, batch), so the 32 threads of a SIMD group share a row
// of A, which they read as one broadcast, and read 32 adjacent columns of B.
// prec_idx is bound on every op (to C where there is no map) and read only
// on the mixed ones.
kernel void mpt_gemm(device const mpt_storage_t *A [[buffer(0)]],
                     device const mpt_storage_t *B [[buffer(1)]],
                     device mpt_storage_t *C [[buffer(2)]],
                     constant mptorch_mps::GemmLaunch &g [[buffer(3)]],
                     device const int32_t *prec_idx [[buffer(4)]],
                     uint3 gid [[thread_position_in_grid]])
{
    mptorch_mps::gemm(mpt_args(), g, A, B, C, prec_idx, gid);
}
