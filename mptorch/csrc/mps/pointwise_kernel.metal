// The fused pointwise and table ops (binaryK_pointwise, superfp_pointwise,
// fixedpoint_pointwise and their _pwl siblings) on an Apple GPU: element j
// of the result is
//
//     o[j] = storage(cast(step(x[j], y[j]))),
//
// common/pointwise.h's statement, as the CPU kernels compute it
// (cpu/pointwise_kernel.h). The step is pointwise.h's ArithStep or
// PwlStep in binary32, MPS's only carrier, whose Metal arms keep the GPU's
// flush of binary32 subnormals away from it (bit_helper.h's
// metal_sub_rn, metal_mul_rn and metal_div_rn, carrier_odd_metal.h's
// round-to-odd primitives, and exact compares in the table's range tests).
// The cast is quantize.metal's quantize_one, the MPS quantizer's, which
// decides on the word. So the result is the CPU's bit for bit.
//
// Compiled after prelude.metal, pointwise.metal and the instantiation's
// lines (pointwise_entry.cpp): quantize.metal's (the storage type, the round
// mode, the format), then mpt_index_t (uint32_t when every element's index
// fits, uint64_t otherwise) and MPT_PWL (1 for a table op). The op, the
// carrier rounding, the number operand and a table's constants are launch
// parameters (pointwise_params.h), as the CPU's step takes them, so that a
// format compiles once for every op.
//
// The operands are read through the geometry (pointwise::offsets, with
// FastDivmod, when it is not flat; the host keeps such a geometry under 2^31
// elements) and the result is written densely. Each thread takes four
// consecutive elements, the four that share a Philox block under
// RoundMode::SR: element j takes word j & 3 of block j >> 2 of the stream
// keyed on the CPU generator's seed, as quantize.metal's elements do, so an
// SR result is the CPU's.
//
// `b` and `entries` are bound to the first operand when the call has no
// second tensor or no table; the kernel then reads neither.

#include "quantize.metal"

kernel void mpt_pointwise(device const mpt_storage_t *a [[buffer(0)]],
                          device const mpt_storage_t *b [[buffer(1)]],
                          device mpt_storage_t *o [[buffer(2)]],
                          device const float *entries [[buffer(3)]],
                          constant mptorch::pointwise::PointwiseGeom &geom [[buffer(4)]],
                          constant mptorch_mps::PointwiseLaunch &q [[buffer(5)]],
                          uint t [[thread_position_in_grid]])
{
    namespace pw = mptorch::pointwise;
    const mpt_index_t n = mpt_index_t(q.n);
    const mpt_index_t first = mpt_index_t(t) * 4;
    if (first >= n)
        return;
    PhiloxBlock block;
    if constexpr (MPT_ROUND_MODE == RoundMode::SR)
        block = philox_block(q.seed, t, 0);
    // pw::offsets takes the geometry by a thread reference
    const pw::PointwiseGeom g = geom;
#if MPT_PWL
    const pw::PwlStep<float> step_of{entries, q.segments, q.lo, q.hi, q.scale, q.tails, q.rsqrt != 0};
#else
    const pw::ArithStep step_of{q.op, q.odd != 0};
#endif
    for (int l = 0; l < 4; ++l)
    {
        const mpt_index_t j = first + l;
        if (j >= n)
            return;
        int64_t oa = int64_t(j), ob = int64_t(j);
        if (!g.flat)
            pw::offsets(g, int32_t(j), oa, ob);
        const float x = mptorch_mps::to_carrier(a[oa]);
        const float y = g.has_b ? mptorch_mps::to_carrier(b[ob]) : q.y;
        mptorch_mps::store_carrier(o, j, mptorch_mps::quantize_one(step_of(x, y), block.word(l)));
    }
}
