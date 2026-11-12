// The CUDA kernels of block_pack, block_unpack, block_quant and block_quant_
// for float64 tensors, which round in binary64: block_kernel.cuh's templates
// instantiated for double, whose binary32 instantiations are
// block_kernel.cu's. A file of its own, like the GEMM's *_f64 twins, so that
// MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out. The
// entry points are in block_entry.cpp.

#include "block_kernel.cuh"

namespace mptorch::block_cuda
{
    void launch_block_pack(const BlockPackJob &j, const BlockFormatParamsT<double> &p, const BlockCastT<double> &c,
                           at::PhiloxCudaState rng, cudaStream_t stream)
    {
        launch_pack_dtype<double, double>(j, p, c, rng, stream);
    }

    void launch_block_unpack(const BlockUnpackJob &j, const BlockFormatParamsT<double> &p, cudaStream_t stream)
    {
        const unsigned grid = grid_of((j.batch * j.rows * j.cols + THREADS - 1) / THREADS);
        block_unpack_kernel<double, double><<<grid, THREADS, 0, stream>>>(j, p);
    }
} // namespace mptorch::block_cuda
