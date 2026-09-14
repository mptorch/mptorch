// What the ops do when they are handed a tensor that requires grad.
//
// Every op in this extension is registered for CPU and CUDA only. Without an
// Autograd kernel, calling one on an operand that requires grad *runs*,
// returns a tensor carrying a grad_fn, and leaves that operand's `.grad` as
// None after `.backward()` -- with one warning, at call time, from a
// deprecation path that a future PyTorch will remove:
//
//   UserWarning: mptorch::custom_matmul_binaryK: an autograd kernel was not
//   registered to the Autograd key(s) but we are trying to backprop through
//   it. This may lead to silently incorrect behavior.
//
// So anyone hand-rolling a quantized attention block out of the raw ops today
// gets a silently untrained model. These ops have no derivative the dispatcher
// could infer either: an op that simulates its own arithmetic is not
// differentiable in the sense autograd means, and what its gradient *should*
// be -- which format each gradient pass runs in -- is exactly what
// `mptorch.quant.qmatmul` / `QMatmul` and `mptorch.quant.Quantizer` exist to
// let a caller say.
//
// This kernel therefore raises, and names the entry point that does carry a
// gradient. A fallthrough would be quieter but keeps the silent-gradient-loss
// shape, which is the bug. See dev/gemm_roadmap.md (X1).
//
// The check is `GradMode` *and* requires_grad, which is what makes this safe
// inside the layers: a torch.autograd.Function's forward runs with grad mode
// disabled, so `CustomArithLinear` / `CustomArithMatmul` / `Quantizer` call
// straight through on operands that require grad, exactly as they did before.

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

namespace
{

  // Every op this is registered on takes its tensors as leading arguments, so
  // the whole stack is worth scanning; a schema with no tensor arguments at
  // all would simply never trigger.
  bool any_requires_grad(const c10::OperatorHandle &op, const torch::jit::Stack &stack)
  {
    const auto n = op.schema().arguments().size();
    for (const c10::IValue &v : torch::jit::last(stack, n))
      if (v.isTensor() && v.toTensor().requires_grad())
        return true;
    return false;
  }

  void raise_on_grad(const c10::OperatorHandle &op, c10::DispatchKeySet ks,
                     torch::jit::Stack *stack)
  {
    if (at::GradMode::is_enabled() && any_requires_grad(op, *stack))
    {
      const std::string &name = op.schema().operator_name().name;
      const bool matmul = name.find("custom_matmul") != std::string::npos;
      const bool narrow = name.find("narrow_float64") != std::string::npos;
      TORCH_CHECK(
          false, name, " is not differentiable: ",
          matmul ? "it simulates the arithmetic of its own reduction, so which format each "
                   "gradient pass runs in is a choice rather than a derivative. Use "
                   "mptorch.quant.qmatmul (or QMatmul), which computes each gradient in the "
                   "format its own hook names"
          : narrow ? "it is the store of a result mptorch.quant computed in binary64, which "
                     "its own entry points call where no gradient flows. Use Tensor.to for a "
                     "differentiable conversion"
                   : "rounding to a coarse format has a zero derivative almost everywhere, so a "
                     "gradient through it is a modelling choice. Use mptorch.quant.Quantizer, "
                     "which applies one format in the forward pass and one of your choosing in "
                     "the backward",
          ". Detach the operand if no gradient was wanted.");
    }
    op.redispatchBoxed(ks & c10::after_autograd_keyset, stack);
  }

} // namespace

TORCH_LIBRARY_IMPL(mptorch, Autograd, m)
{
  const auto kernel = []
  { return torch::CppFunction::makeFromBoxedFunction<&raise_on_grad>(); };
  m.impl("binaryK_quant", kernel());
  m.impl("superfp_quant", kernel());
  m.impl("narrow_float64", kernel());
  m.impl("custom_matmul_binaryK", kernel());
  m.impl("custom_matmul_superfp", kernel());
  m.impl("custom_matmul_binaryK_fma", kernel());
  m.impl("custom_matmul_superfp_fma", kernel());
  m.impl("custom_matmul_binaryK_mixed", kernel());
  m.impl("custom_matmul_superfp_mixed", kernel());
  m.impl("custom_matmul_binaryK_fma_mixed", kernel());
  m.impl("custom_matmul_superfp_fma_mixed", kernel());
}
