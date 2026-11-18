#pragma once

// The MPS backend's one interface to Metal, as the C++ files see it: compile
// a kernel for one instantiation, cache it, and encode one launch of it on
// torch's MPS stream. The implementation (metal_runtime.mm) is the backend's
// only Objective-C++; every other file in csrc/mps is plain C++ that writes
// the lines naming an instantiation (its "tail"), fills the per-launch
// parameters (launch_params.h), and hands over the tensors.
//
// A kernel's source is prelude.metal, then the tail, then gemm.metal,
// quantize.metal or block.metal. The odd carrier step's kernels put
// odd.metal between the prelude and the tail, and end in gemm.metal or
// carrier_odd.metal. The tail holds everything that is a constant of the
// instantiation: the storage dtype, the rounding mode and the format, from
// which the kernel builds its cast constants at compile time (gemm.metal
// says why). Each distinct tail is one compile, 20-300 ms the first time a
// process meets it (less once the system's Metal cache has seen it), and a
// hash lookup after that. A layer's formats are fixed, so a model compiles
// a handful.

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mptorch_mps
{

  // Which kernel source follows the tail, and so which kernel is launched.
  // The two odd carrier step sources also put odd.metal, the odd headers,
  // between the prelude and the tail. New sources go at the end, since the
  // cache key starts with the source's number.
  enum class KernelSource
  {
    Gemm,          // gemm.metal's mpt_gemm
    Quantize,      // quantize.metal's mpt_quantize
    BlockPack,     // block.metal's mpt_block_pack
    BlockPackTile, // block.metal's mpt_block_pack_tile
    BlockUnpack,   // block.metal's mpt_block_unpack
    GemmOdd,       // gemm.metal's mpt_gemm, after odd.metal: an odd Args
    CarrierOdd,    // carrier_odd.metal's mpt_carrier_odd, after odd.metal
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

  // Encodes one launch of the kernel `source`, instantiated by `tail`, on
  // the current MPS stream, after whatever torch has already encoded there
  // and before whatever comes next. Compiles the instantiation first if this
  // process has not; a compile error is raised with the Metal compiler's
  // message. Nothing is waited for.
  //
  // `launch` runs `threads` threads in one dimension, in threadgroups the
  // runtime sizes (quantize.metal). `launch_groups` runs (gx, gy, gz) whole
  // threadgroups of tx x ty threads, for a kernel whose threadgroup shares
  // its work and so must be complete (gemm.metal's tiles).
  void launch(KernelSource source, const std::string &tail, const std::vector<KernelArg> &args,
              uint64_t threads);
  void launch_groups(KernelSource source, const std::string &tail, const std::vector<KernelArg> &args,
                     uint64_t gx, uint64_t gy, uint64_t gz, uint32_t tx, uint32_t ty);

  // The Metal name of a tensor's dtype, for a tail's `mpt_storage_t`. A dtype
  // the kernels do not load raises a NotImplementedError naming `op`.
  const char *metal_storage_type(at::ScalarType t, const char *op);

  // The RoundMode enumerator that a mode's integer names, written out for a
  // tail so that a compile error shows the enumerator, as the source would.
  // `rm` must be a valid RoundMode.
  const char *metal_round_mode(int64_t rm);

} // namespace mptorch_mps
