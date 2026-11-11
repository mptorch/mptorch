"""The kernels under torch.export and torch.compile."""

import warnings

import torch

from mptorch import RoundMode
from mptorch.quant import QLinear, binaryK_gemm_formats, binaryK_matmul, binaryK_quantize

torch.manual_seed(0)
layer = QLinear(32, 16, formats=binaryK_gemm_formats(8, 4, acc_K=16, acc_P=11))

# Export with a dynamic batch. The program is the forward alone: the layer's
# autograd.Function is inlined and its GEMM operator is in the graph.
x = torch.randn(8, 32)
ep = torch.export.export(layer, (x,), dynamic_shapes=({0: torch.export.Dim.AUTO},))
print([str(n.target) for n in ep.graph.nodes if n.op == "call_function"])
program = ep.module()
with torch.no_grad():  # the forward alone has no gradient to give
    for batch in (8, 5):
        x = torch.randn(batch, 32)
        print(batch, torch.equal(program(x), layer(x)))

# Compile two GEMMs and a quantizer, here under stochastic rounding. The flat
# functions trace into one graph; with fallback_random the random operators
# run in program order, so the compiled result is eager's, draw for draw.
# Dynamo traces through the format checks' lru_cache, which it says once per
# function; the checks are pure, so their cache only saves time.
warnings.filterwarnings("ignore", message="Dynamo detected a call to a `functools.lru_cache`")
SR = RoundMode.SR


def step(a, b):
    kw = dict(mul_K=8, mul_P=4, mul_prng_bits=8, acc_K=16, acc_P=11, acc_prng_bits=8)
    q = binaryK_quantize(a, 8, 4, prng_bits=8, rounding_mode=SR)
    h = binaryK_matmul(q, b, rounding_mode=SR, **kw)
    return binaryK_matmul(torch.relu(h), b, trans_b=True, rounding_mode=SR, **kw)


a, b = torch.randn(8, 16), torch.randn(16, 16)
explanation = torch._dynamo.explain(step)(a, b)
print(f"{explanation.graph_count} graph, {explanation.graph_break_count} breaks")
compiled = torch.compile(step)
with torch._inductor.config.patch(fallback_random=True):
    torch.manual_seed(1)
    got = compiled(a, b)
torch.manual_seed(1)
print("compiled == eager:", torch.equal(got, step(a, b)))
