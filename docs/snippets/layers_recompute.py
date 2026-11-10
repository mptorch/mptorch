"""What a quantized layer keeps for backward: a quantized copy, or the raw operand."""

import torch
from torch import nn

from mptorch import BinaryK, RoundMode
from mptorch.quant import QAffineFormats, QConv2d, QLinear, Quant

q = Quant(BinaryK(8, 4, prng_bits=8), RoundMode.SR)


def held_for_backward(model, x):
    """The bytes the graph keeps for backward that nothing else keeps alive:
    every tensor any node saves, once per storage, less the parameters and
    the input, which the caller holds anyway."""
    saved = {}

    def pack(t):
        saved[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(x)
    alive = {p.untyped_storage().data_ptr() for p in model.parameters()}
    alive.add(x.untyped_storage().data_ptr())
    return sum(n for ptr, n in saved.items() if ptr not in alive), out


def stack(recompute):
    """Four conv -> ReLU layers: nn.Conv2d, or QConv2d with SR on every signal."""
    torch.manual_seed(1)
    layers = []
    for _ in range(4):
        if recompute == "nn":
            layers += [nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()]
            continue
        f = QAffineFormats(input_quant=q, weight_quant=q, igrad_quant=q, wgrad_quant=q)
        if recompute is not None:
            f.recompute_input = f.recompute_weight = recompute
        layers += [QConv2d(16, 16, 3, padding=1, formats=f), nn.ReLU()]
    return nn.Sequential(*layers)


x = torch.randn(8, 16, 32, 32)
activation = x.numel() * x.element_size()
grads = {}
for name, recompute in (("nn.Conv2d", "nn"), ("QConv2d, copies kept", False), ("QConv2d", None)):
    model = stack(recompute)
    torch.manual_seed(2)
    held, out = held_for_backward(model, x)
    out.sum().backward()
    grads[name] = [p.grad for p in model.parameters()]
    print(f"{name:21s} holds {held / activation:.2f} activations for backward")
same = all(map(torch.equal, grads["QConv2d"], grads["QConv2d, copies kept"]))
print("the two QConv2d stacks' gradients, under SR, are identical:", same)


# A Linear keeps its input as a quantized copy and its weight raw.
def shares(saved, raw):
    return saved.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()


layer = QLinear(64, 64, formats=QAffineFormats(input_quant=q, weight_quant=q))
x = torch.randn(32, 64, requires_grad=True)
saved_input, saved_weight = layer(x).grad_fn.saved_tensors
print("\nQLinear keeps its input raw:", shares(saved_input, x))
print("QLinear keeps its weight raw:", shares(saved_weight, layer.weight))

# A quantizer mptorch does not know to be stateless keeps its result, unless
# the flag says it may run twice.
formats = QAffineFormats(weight_quant=lambda w: q(w))
layer = QLinear(64, 64, formats=formats)
print(
    "\nwith a callable of its own, the weight raw:",
    shares(layer(x).grad_fn.saved_tensors[1], layer.weight),
)
formats.recompute_weight = True
print("and with recompute_weight=True:", shares(layer(x).grad_fn.saved_tensors[1], layer.weight))
