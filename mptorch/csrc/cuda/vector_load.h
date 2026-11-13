#pragma once

// The input a vector-loading CUDA kernel can read in place, and the tensor one
// can write in place.
//
// quant_kernel_all and quant_kernel_all_sr (utils.cuh) and narrow_kernel
// (narrow_kernel.cu) read their input 16 bytes at a time through an `int4`
// or a `ulonglong2`. Such a load requires a 16-byte-aligned address, and a
// misaligned one raises `CUDA error: misaligned address`, which is sticky:
// the context is lost and every later CUDA call in the process fails.
// `.contiguous()` alone does not rule that out. A contiguous view need not
// start on a 16-byte boundary (`x[1:]` of a float32 tensor starts 4 bytes
// into its storage, of a float16 one 2, and `w[1:]` of a [n, 3] float32
// matrix 12), and `.contiguous()` hands such a view back as it is. So a view
// whose data pointer is off the boundary is cloned into an allocation of its
// own, which the CUDA caching allocator starts on a 512-byte boundary.
//
// Only such a view pays for this, and the cost is one device copy of the
// input for the duration of the call, the same copy `.contiguous()` already
// makes of a strided input. Every other input (a kernel's own output,
// anything `.contiguous()` had to copy, a tensor at the start of its
// storage) is read in place after one pointer test. The GEMM kernels load
// one element at a time and need none of this.

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <cstdint>

namespace mptorch
{

  // `a` made contiguous and, if its first element is off a 16-byte
  // boundary, copied. The assertion documents what the allocator promises.
  inline at::Tensor vector_loadable(const at::Tensor &a)
  {
    at::Tensor c = a.contiguous();
    if (reinterpret_cast<uintptr_t>(c.data_ptr()) % 16 != 0)
    {
      c = c.clone();
      TORCH_INTERNAL_ASSERT(reinterpret_cast<uintptr_t>(c.data_ptr()) % 16 == 0,
                            "a fresh CUDA allocation that is not 16-byte aligned");
    }
    return c;
  }

  // The in-place ops' side of the same requirement. An op that writes its
  // argument cannot take it through vector_loadable: the copy that function
  // makes of a strided or off-boundary view is what the kernel would write,
  // and the caller's tensor would come back untouched. So such a tensor is
  // refused, with a message naming the out-of-place op, which does copy.
  inline void check_vector_loadable_in_place(const at::Tensor &a, const char *op,
                                             const char *out_of_place_op)
  {
    TORCH_CHECK(a.is_contiguous(), op, " writes its argument in place and needs a contiguous "
                "tensor, got strides ", a.strides(), " for sizes ", a.sizes(),
                ": use ", out_of_place_op, ", which copies a strided input");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0, op,
                " writes its argument in place and its CUDA kernel moves 16 bytes at a time, "
                "so the tensor must start on a 16-byte boundary; this one is a view that "
                "starts ", reinterpret_cast<uintptr_t>(a.data_ptr()) % 16,
                " bytes past one (x[1:] is such a view): use ", out_of_place_op,
                ", which copies such an input");
  }

} // namespace mptorch
