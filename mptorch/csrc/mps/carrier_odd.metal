// _carrier_odd on an Apple GPU: carrier_odd_metal.h's primitives applied
// elementwise, so the tests can see what the format casts after them erase
// (zero signs, the largest finite value, subnormal parity), word for word
// against the CPU's. A test op, binary32 only.
//
// Compiled after prelude.metal, odd.metal and a line the backend writes
// (mps/carrier_odd_entry.cpp): MPT_ODD_OP, 0 for fma, 1 for add, 2 for mul.
// The operands are read and the result written as words, so that no load or
// store of a subnormal can be a float operation the GPU flushes.

kernel void mpt_carrier_odd(device const uint32_t *a [[buffer(0)]],
                            device const uint32_t *b [[buffer(1)]],
                            device const uint32_t *c [[buffer(2)]],
                            device uint32_t *o [[buffer(3)]],
                            uint i [[thread_position_in_grid]])
{
    namespace odd = mptorch::gemm::odd;
    const float x = as_type<float>(a[i]), y = as_type<float>(b[i]);
#if MPT_ODD_OP == 0
    const float r = odd::fma(x, y, as_type<float>(c[i]));
#elif MPT_ODD_OP == 1
    const float r = odd::add(x, y);
#else
    const float r = odd::mul(x, y);
#endif
    o[i] = as_type<uint32_t>(r);
}
