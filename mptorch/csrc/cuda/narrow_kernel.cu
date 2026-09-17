#include "../common/narrow_binary64.h"
#include "narrow_kernel.h"
#include <algorithm>
#include <cuda_runtime.h>

// narrow_float64 on the device: common/narrow_binary64.h's word-level
// rounding, applied to two elements per thread through one 128-bit load and
// one store of the resulting pair, the way quant_kernel_all (utils.cuh) moves
// its float4s. The op is bandwidth bound like every elementwise kernel here,
// and the integer path means the device computes the same bits as the CPU
// kernel by construction, with no dependence on the conversion hardware.

namespace mptorch::narrow_cuda
{
    namespace
    {
        constexpr int BLOCK_SIZE = 256;

        // Rounds `size` doubles at `a` onto the (ExpBits, ManBits) format
        // and stores the words at `o`. `Pair` is the store type for two
        // words: ushort2 for float16 and bfloat16, uint2 for float32. `a`
        // must be 16-byte aligned for the ulonglong2 load (the entry point
        // guarantees it). The loop is grid-stride, so a tensor whose thread
        // count outruns grid.x's 2^31 - 1 is still covered; an odd last
        // element is done by thread 0 through a scalar load.
        template <class Word, int ExpBits, int ManBits, class Pair>
        __global__ __launch_bounds__(256, 2)
        void narrow_kernel(const double *__restrict__ a, Word *__restrict__ o, int64_t size)
        {
            const int64_t pairs = size / 2;
            const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
            const int64_t stride = (int64_t)gridDim.x * blockDim.x;

            for (int64_t i = idx; i < pairs; i += stride)
            {
                const ulonglong2 in = reinterpret_cast<const ulonglong2 *>(a)[i];
                Pair out;
                out.x = narrow_binary64<Word, ExpBits, ManBits>(in.x);
                out.y = narrow_binary64<Word, ExpBits, ManBits>(in.y);
                reinterpret_cast<Pair *>(o)[i] = out;
            }

            if (idx == 0 && (size & 1))
                o[size - 1] = narrow_binary64<Word, ExpBits, ManBits>(
                    reinterpret_cast<const unsigned long long *>(a)[size - 1]);
        }

        template <class Word, int ExpBits, int ManBits, class Pair>
        void launch_as(const double *a, void *o, int64_t size, cudaStream_t stream)
        {
            // The same grid computation as utils.cuh's grid_for, restated
            // because that header pulls in ATen, which this object must not.
            const int64_t threads = std::max<int64_t>(size / 2, size & 1);
            const int64_t grid = std::clamp<int64_t>((threads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1,
                                                     2147483647);
            narrow_kernel<Word, ExpBits, ManBits, Pair><<<(int)grid, BLOCK_SIZE, 0, stream>>>(
                a, static_cast<Word *>(o), size);
        }
    } // namespace

    // (5, 10) is float16, (8, 7) bfloat16 and (8, 23) float32.
    void launch(const double *a, void *o, int64_t size, Target target, cudaStream_t stream)
    {
        switch (target)
        {
        case Target::Float16:
            launch_as<uint16_t, 5, 10, ushort2>(a, o, size, stream);
            break;
        case Target::BFloat16:
            launch_as<uint16_t, 8, 7, ushort2>(a, o, size, stream);
            break;
        default:
            launch_as<uint32_t, 8, 23, uint2>(a, o, size, stream);
            break;
        }
    }
} // namespace mptorch::narrow_cuda
