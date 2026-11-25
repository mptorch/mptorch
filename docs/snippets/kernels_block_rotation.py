"""Rotating both operands along K before block quantization."""

import torch
from torch import nn

from mptorch import MXFP4_E2M1, MXFP8_E4M3, NVFP4
from mptorch.quant import BlockMac, GemmFormats, Hadamard, QLinear

torch.manual_seed(0)
x = torch.randn(64, 768)
x[:, [7, 100, 300]] *= 30  # three outlier channels, as in an LLM's activations
w = torch.randn(256, 768) / 768**0.5
ref = x.double() @ w.double().T


def error(fmt, rotation):
    """The relative error of x @ w.T through a block GEMM layer in fmt."""
    gf = GemmFormats(fwd_mac=BlockMac(fmt), rotation=rotation)
    layer = QLinear(768, 256, bias=False, formats=gf.affine())
    with torch.no_grad():
        layer.weight.copy_(w)
        y = layer(x)
    return ((y.double() - ref).norm() / ref.norm()).item()


print("relative error of x @ w.T")
print(f"{'format':<8}{'none':>8}{'DH32':>8}{'DH128':>8}")
for name, fmt in (("MXFP8", MXFP8_E4M3), ("MXFP4", MXFP4_E2M1), ("NVFP4", NVFP4)):
    row = [error(fmt, r) for r in (None, Hadamard(32), Hadamard(128))]
    print(f"{name:<8}" + "".join(f"{e:8.4f}" for e in row))

# The rotation is orthonormal and the gradients are rotated back, so without
# quantization a rotated layer is nn.Linear, up to the rotation's rounding.
lin = nn.Linear(768, 256)
q = QLinear(768, 256, formats=GemmFormats(rotation=Hadamard(128)).affine())
q.load_state_dict(lin.state_dict())
xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
lin(xa).sum().backward()
q(xb).sum().backward()
assert xa.grad is not None and xb.grad is not None
gap = (xa.grad - xb.grad).abs().max() / xa.grad.abs().max()
print(f"input gradient against nn.Linear's, relative: {gap.item():.1e}")
