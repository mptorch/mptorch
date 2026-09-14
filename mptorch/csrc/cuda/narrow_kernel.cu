#include "../common/narrow_binary64.h"
#include "narrow_kernel.h"
#include <algorithm>
#include <cuda_runtime.h>

// narrow_float64 on the device: common/narrow_binary64.h, two elements a
// thread through one 128-bit load and one store of the pair, the way
// quant_kernel_all (utils.cuh) moves its float4s -- the op is bandwidth bound
// like every elementwise kernel here.

namespace mptorch::narrow_cuda
{
    namespace
    {
        constexpr int BLOCK_SIZE = 256;

        // `Pair` is the store: two float16 or bfloat16 words are a ushort2,
        // two float32 words a uint2. The grid-stride loop covers a tensor
        // whose thread count outruns grid.x's 2^31 - 1, as quant_kernel_all's
        // does.
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
            // utils.cuh's grid_for, which this object cannot include without
            // ATen
            const int64_t threads = std::max<int64_t>(size / 2, size & 1);
            const int64_t grid = std::clamp<int64_t>((threads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1,
                                                     2147483647);
            narrow_kernel<Word, ExpBits, ManBits, Pair><<<(int)grid, BLOCK_SIZE, 0, stream>>>(
                a, static_cast<Word *>(o), size);
        }
    } // namespace

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
