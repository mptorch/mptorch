"""Lookup-table activations: the three fits, and a table in an activation's formats."""

import torch

from mptorch import BinaryK
from mptorch.quant import ActivationFormats, PWLTable, Quant, gelu_table, qsilu, silu_table

print("largest error on [-8, 8], on 1024 points a segment")
print(f"{'':<6}{'segments':>9}{'chord':>11}{'balanced':>11}{'minimax':>11}")
for name, preset in (("SiLU", silu_table), ("GELU", gelu_table)):
    for n in (8, 16, 64):
        errors = [preset(n, fit=fit).max_error() for fit in ("chord", "balanced", "minimax")]
        print(f"{name:<6}{n:>9}" + "".join(f"{e:11.2e}" for e in errors))

print("\n1/sqrt over [1, 4), largest relative error")
for n in (16, 32, 64):
    errors = [
        PWLTable.rsqrt(n, fit=fit).max_error(relative=True)
        for fit in ("chord", "balanced", "minimax")
    ]
    print(f"{'':<6}{n:>9}" + "".join(f"{e:11.2e}" for e in errors))

# SiLU through its 16-segment table, the result rounded to bfloat16's
# precision; the gradient is SiLU's own derivative, not the table's slope
x = torch.linspace(-6, 6, 7, requires_grad=True)
f = ActivationFormats(table=silu_table(), out=Quant(BinaryK(14, 8)))
y = qsilu(x, f)
y.sum().backward()
print("\nsilu table:", [round(v, 4) for v in y.tolist()])
print("gradient:  ", [round(v, 4) for v in x.grad.tolist()])
