"""Compiled code keeps the result of an in-place quantizer.

The in-place quantizer ops (``binaryK_quant_`` and its siblings) write their
argument and return nothing. While they returned it, as ATen's own in-place
ops do, Inductor took the returned alias for a buffer of its own and handed the
argument's memory to a later kernel while that result was still in use, so a
compiled function returned values the later kernel had written (torch 2.15).
Each case keeps a rounded result live past a second rounding and holds the
compiled function to eager, word for word, through every path that rounds in
place: the public in-place quantizers, a pointwise op's rounding, a layer's
output rounding and ``ScaledQuant``.
"""

import pytest
import torch
from torch import nn

from mptorch import BinaryK
from mptorch.number import MXFP8_E4M3
from mptorch.quant import (
    GemmFormats,
    Quant,
    Recipe,
    ScaledQuant,
    binaryK_quantize_,
    block_quantize_,
    convert,
    fixedpoint_quantize_,
    qadd,
    qmul,
    superfp_quantize_,
)
from tests.markers import available_devices

Q = Quant(BinaryK(8, 4))


def _compiled(f):
    torch._dynamo.reset()
    return torch.compile(f, backend="inductor", fullgraph=True)


def _same_words(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g.view(torch.int32), w.view(torch.int32))


# The flat quantizers look their format up per call, so Dynamo says once per
# function that it traces through the format checks' lru_cache (and block.py's
# _code_fields); the checks are pure, so the cache only saves time.
@pytest.mark.filterwarnings("ignore:Dynamo detected a call to a `functools.lru_cache`:UserWarning")
@pytest.mark.parametrize("device", available_devices)
def test_the_in_place_quantizers(device):
    """Each family's in-place quantizer, with its first result returned."""

    def f(x):
        a = binaryK_quantize_(x + 1, 8, 4)
        b = superfp_quantize_(x * 2, 3, 4, 8, 7)
        c = fixedpoint_quantize_(x - 1, 8, 4)
        d = block_quantize_(x * 5, MXFP8_E4M3)
        return (
            a,
            binaryK_quantize_(a * 3, 8, 4),
            b,
            superfp_quantize_(b * 3, 3, 4, 8, 7),
            c,
            fixedpoint_quantize_(c * 3, 8, 4),
            d,
            block_quantize_(d * 3, MXFP8_E4M3),
        )

    x = torch.randn(64, 64, device=device)
    _same_words(_compiled(f)(x), f(x))


@pytest.mark.parametrize("device", available_devices)
def test_a_pointwise_result(device):
    """A pointwise op's rounded result, returned and also rounded on."""

    def f(x, y):
        h = qadd(x, y, Q)
        return h, qmul(h, y, Q)

    x = torch.randn(64, 64, device=device)
    y = torch.randn(64, 64, device=device)
    _same_words(_compiled(f)(x, y), f(x, y))


class _Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64)
        self.b = nn.Linear(64, 64)

    def forward(self, x):
        y = self.a(x)
        return y, y + self.b(y)


@pytest.mark.parametrize("device", available_devices)
def test_a_rounded_layer_output(device):
    """A converted layer's rounded output, fed to a residual and returned."""
    model, _ = convert(_Residual(), Recipe(linear=GemmFormats(output_quant=Q)))
    model = model.to(device)
    x = torch.randn(64, 64, device=device)
    with torch.no_grad():
        _same_words(_compiled(model)(x), model(x))


@pytest.mark.parametrize("device", available_devices)
def test_scaled_quant_applied_twice(device):
    """``ScaledQuant`` rounds its carrier tensor in place. On CUDA, Inductor's
    division by the scale can differ from eager's in the last place, so the
    results are held close rather than equal: the defect left the first
    result holding the second call's product, three times the value."""
    sq = ScaledQuant(BinaryK(8, 4))

    def f(x):
        a = sq(x)
        return a, sq(a * 3)

    x = torch.randn(64, 64, device=device)
    got, want = _compiled(f)(x), f(x)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w, rtol=2**-20, atol=0)
