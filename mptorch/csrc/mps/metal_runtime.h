#pragma once

// The MPS backend's one door into Metal, as the C++ files see it: compile a
// kernel for one instantiation, cache it, and encode one launch of it on
// torch's MPS stream. The implementation (metal_runtime.mm) is the backend's
// only Objective-C++; everything else in csrc/mps is plain C++ that builds
// the lines naming an instantiation (its "tail") and the per-launch
// parameters (launch_params.h), and hands over the tensors.
//
// A kernel's source is prelude.metal, then the tail, then gemm.metal or
// quantize.metal. The tail holds everything that is a constant of the
// instantiation: the storage dtype, the rounding mode and the format, which
// the kernel builds its cast constants from so that they fold (gemm.metal
// says why). Each distinct tail is one compile, 20-300 ms the first time a
// process meets it (less once the system's Metal cache has seen it), then a
// hash lookup. A layer's formats are fixed, so a model compiles a handful.

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace mptorch_mps
{

  // Which kernel source follows the tail, and so which kernel is launched.
  enum class KernelSource
  {
    Gemm,     // gemm.metal's mpt_gemm
    Quantize, // quantize.metal's mpt_quantize
  };

  // One argument of a launch, in buffer-index order: an MPS tensor, bound as
  // its storage's MTLBuffer at the tensor's byte offset, or a few bytes of
  // launch parameters, copied into the command stream.
  struct KernelArg
  {
    const at::Tensor *tensor = nullptr;
    const void *bytes = nullptr;
    size_t size = 0;

    KernelArg(const at::Tensor &t) : tensor(&t) {}
    template <class Params>
    static KernelArg params(const Params &p)
    {
      KernelArg a;
      a.bytes = &p;
      a.size = sizeof(Params);
      return a;
    }

  private:
    KernelArg() = default;
  };

  // Encodes one launch of the kernel `source` instantiated by `tail` on the
  // current MPS stream, after whatever torch has already encoded there and
  // before whatever comes next. Compiles the instantiation first if this
  // process has not; a compile error is raised with the Metal compiler's
  // message. Nothing is waited for.
  //
  // `launch` runs `threads` threads, one-dimensional, in threadgroups the
  // runtime sizes (quantize.metal). `launch_groups` runs (gx, gy, gz) whole
  // threadgroups of tx x ty threads, for a kernel whose threadgroup shares
  // its work and so must be complete (gemm.metal's tiles).
  void launch(KernelSource source, const std::string &tail, std::initializer_list<KernelArg> args,
              uint64_t threads);
  void launch_groups(KernelSource source, const std::string &tail,
                     std::initializer_list<KernelArg> args, uint64_t gx, uint64_t gy, uint64_t gz,
                     uint32_t tx, uint32_t ty);

  // The Metal name of a tensor's dtype, for a tail's `mpt_storage_t`, or a
  // NotImplementedError naming `op` for a dtype the kernels do not load.
  const char *metal_storage_type(at::ScalarType t, const char *op);

  // The RoundMode enumerator a mode's integer names, for a tail, spelled out
  // so that a compile error reads as the source would. `rm` must be a valid
  // RoundMode.
  const char *metal_round_mode(int64_t rm);

} // namespace mptorch_mps
