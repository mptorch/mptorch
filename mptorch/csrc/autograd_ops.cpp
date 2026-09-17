// What the ops do when they are handed a tensor that requires grad: the
// Autograd dispatch key of every op in this extension, one boxed kernel that
// raises.
//
// The ops have CPU and CUDA kernels only. With no Autograd kernel at all,
// calling one on an operand that requires grad under grad mode still runs,
// returns a tensor that carries a grad_fn, and leaves that operand's `.grad`
// as None after `.backward()`, with only a call-time UserWarning ("an
// autograd kernel was not registered to the Autograd key(s) but we are
// trying to backprop through it") from a deprecation path a future PyTorch
// will remove. Anyone assembling a quantized block out of the raw ops would
// get a silently untrained model. The dispatcher cannot infer a derivative
// either: an op that simulates its own arithmetic is not differentiable in
// autograd's sense, and what its gradient should be, which format each
// gradient pass runs in, is a modelling choice that
// `mptorch.quant.qmatmul` / `QMatmul` and `mptorch.quant.Quantizer` exist to
// let a caller make.
//
// The kernel therefore raises and names the entry point that does carry a
// gradient. A silent fallthrough to the backend would keep the
// gradient-loss shape, which is the bug.
//
// The check is `GradMode::is_enabled()` *and* requires_grad, which is what
// makes this safe inside the layers: a torch.autograd.Function's forward
// runs with grad mode disabled (its autograd keys are excluded), so
// `CustomArithLinear` / `CustomArithMatmul` / `Quantizer` call straight
// through to the backend on operands that require grad, and their own
// backward supplies the gradient.

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>

namespace
{

  // True when any tensor argument of the call on the stack requires grad.
  // The op's arguments are the last schema().arguments().size() values of
  // the stack; the tensors are the leading ones, but the whole argument
  // list is scanned so the check holds for any schema, and an op with no
  // tensor arguments would simply never trigger.
  bool any_requires_grad(const c10::OperatorHandle &op, const torch::jit::Stack &stack)
  {
    const auto n = op.schema().arguments().size();
    for (const c10::IValue &v : torch::jit::last(stack, n))
      if (v.isTensor() && v.toTensor().requires_grad())
        return true;
    return false;
  }

  // The boxed Autograd kernel. Boxed (arguments on the stack, one kernel for
  // every schema) so the same function serves all eleven ops. It raises when
  // a gradient would be expected, with a message specific to what the op is
  // (a GEMM, the narrowing store, or an elementwise quantizer), and
  // otherwise redispatches to the backend kernel below the autograd keys,
  // which is what a plain fallthrough would have done.
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

// Every op the library declares (quant_ops.cpp) gets the same kernel under
// the Autograd key. An op added there without a line here falls back to the
// warn-and-run behaviour described at the top of this file.
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
