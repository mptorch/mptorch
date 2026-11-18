#include "../common/narrow_binary64.h"
#include "narrow_kernel.h"
#include <algorithm>
#include <cuda_runtime.h>

// narrow_float64 on the device: common/narrow_binary64.h's rounding on the
// word, applied to two elements per iteration of a thread through one 128-bit
// load and one store of the resulting pair, as quant_kernel_all
// (cuda/utils.cuh) moves its float4s. Like every elementwise kernel here, the
// op is bandwidth bound, and because the rounding is integer arithmetic, the
// device computes the same bits as the CPU kernel whatever the conversion
// hardware does.

namespace mptorch::narrow_cuda
{
    namespace
    {
        constexpr int BLOCK_SIZE = 256;

        // Rounds the `size` doubles at `a` to the (ExpBits, ManBits) format
        // and stores the words at `o`. `Pair` is the store type for two
        // words: ushort2 for float16 and bfloat16, uint2 for float32. `a`
        // must be 16-byte aligned for the ulonglong2 load (the entry point
        // ensures it). The loop strides by the grid's size, so a tensor that
        // would need more than gridDim.x's limit of 2^31 - 1 blocks is still
        // covered; thread 0 handles an odd last element through a scalar
        // load.
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
            // The same grid computation as cuda/utils.cuh's grid_for, written
            // out again because that header includes ATen, which this object
            // must not.
            const int64_t threads = std::max<int64_t>(size / 2, size & 1);
            const int64_t grid = std::clamp<int64_t>((threads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1,
                                                     2147483647);
            narrow_kernel<Word, ExpBits, ManBits, Pair><<<(int)grid, BLOCK_SIZE, 0, stream>>>(
                a, static_cast<Word *>(o), size);
        }
    } // namespace

    // (ExpBits, ManBits): (5, 10) is float16, (8, 7) bfloat16 and (8, 23)
    // float32.
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
