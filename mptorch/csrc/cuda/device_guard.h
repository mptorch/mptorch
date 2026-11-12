#pragma once
// Every CUDA op runs on its tensors' device (since cb8cf52).
//
// PyTorch's own operators get a device guard from its code generator; an
// operator registered through TORCH_LIBRARY_IMPL gets none. mptorch's CUDA
// entry points take the *current* device's stream
// (at::cuda::getCurrentCUDAStream()) and generator
// (getDefaultCUDAGenerator()) and launch there, so without a guard a tensor
// on cuda:1 with cuda:0 current ran on cuda:0 against cuda:1's memory: an
// illegal address without peer access, and an SR draw from cuda:0's
// generator. cuda_ops.cpp registers every op through OnOperandDevice, which
// makes the first CUDA tensor argument's device current for the call and
// gives the caller's back on return or throw; the entry points, which never
// name a device, follow it. Operands on different devices are each op's own
// check (check_matmul_inputs in gemm_host.h, gemm_conv_host.h, block_host.h):
// the guard only picks the device. It is a registration-side wrapper so that
// no entry point or kernel changes, and so that registering a new op the way
// the others are registered is what guards it.

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>
#include <utility>

namespace mptorch
{
  inline void note_cuda_device(std::optional<c10::Device> &device, const at::Tensor &t)
  {
    if (!device && t.defined() && t.is_cuda())
      device = t.device();
  }

  // Every other argument (formats, flags, shapes) names no device.
  template <typename T>
  inline void note_cuda_device(std::optional<c10::Device> &, const T &)
  {
  }

  template <auto F>
  struct OnOperandDevice;

  template <typename R, typename... Args, R (*F)(Args...)>
  struct OnOperandDevice<F>
  {
    [[gnu::noinline]] static R call(Args... args)
    {
      std::optional<c10::Device> device;
      (note_cuda_device(device, args), ...);
      const c10::cuda::OptionalCUDAGuard guard(device);
      return F(std::forward<Args>(args)...);
    }
  };
} // namespace mptorch
