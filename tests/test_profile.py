"""
``profile``: each layer's multiply-accumulates, by the formats it computes in.

``mptorch/quant/profile.py`` records every call of a Linear, a convolution or
a ``QMatmul`` during one forward pass, as the GEMM the kernels run. These
tests count the MACs by hand for each kind of layer and shape (leading
dimensions, groups, stride, dilation and padding, an unbatched input, a
broadcast matmul), then check what a record is filed under, how a cost table
prices it, and that the hooks are gone afterwards, after an error too.
"""

import math

import pytest
import torch
from torch import nn

from mptorch import MXFP8_E4M3, BinaryK
from mptorch.quant import (
    BlockMac,
    FusedMac,
    GemmFormats,
    QLinear,
    QMatmul,
    Recipe,
    SplitMac,
    binaryK_gemm_formats,
    convert_,
    presets,
    profile,
)

P8 = BinaryK(8, 4)
P16 = BinaryK(16, 11)


def _one(module, *inputs):
    (record,) = profile(module, *inputs).records
    return record


def test_a_linear_layer():
    r = _one(nn.Linear(16, 8), torch.randn(3, 5, 16))
    assert (r.kind, r.batch, r.M, r.N, r.K, r.macs) == ("linear", 1, 15, 8, 16, 15 * 8 * 16)
    assert _one(nn.Linear(16, 8), torch.randn(16)).M == 1


@pytest.mark.parametrize(
    "conv,shape",
    [
        (nn.Conv1d(4, 6, 3, stride=2, padding=1), (2, 4, 11)),
        (nn.Conv2d(4, 6, (3, 2), stride=(2, 1), dilation=(1, 2), groups=2), (3, 4, 9, 8)),
        (nn.Conv3d(6, 6, 3, padding=1, groups=6), (1, 6, 4, 4, 4)),
        (nn.Conv2d(4, 6, 3, padding="same"), (4, 7, 7)),  # unbatched
    ],
    ids=str,
)
def test_a_convolution(conv, shape):
    """MACs = batch * Cout * (output positions) * (Cin / groups) * (kernel taps),
    one GEMM per sample and group."""
    x = torch.randn(*shape)
    out = conv(x)
    nd = len(conv.kernel_size)
    batch = shape[0] if len(shape) == nd + 2 else 1
    positions = math.prod(out.shape[-nd:])
    taps = math.prod(conv.kernel_size)
    expected = batch * conv.out_channels * positions * (conv.in_channels // conv.groups) * taps
    r = _one(conv, x)
    assert r.kind == "conv" and r.macs == expected
    assert (r.batch, r.M, r.N) == (batch * conv.groups, conv.out_channels // conv.groups, positions)
    assert r.K == conv.in_channels // conv.groups * taps


@pytest.mark.parametrize(
    "a_shape,b_shape,batch,M,N,K",
    [
        ((5, 12), (12, 7), 1, 5, 7, 12),
        ((3, 5, 12), (12, 7), 3, 5, 7, 12),
        ((2, 1, 5, 12), (3, 12, 7), 6, 5, 7, 12),
        ((12,), (12, 7), 1, 1, 7, 12),
        ((5, 12), (12,), 1, 5, 1, 12),
    ],
)
def test_a_matmul(a_shape, b_shape, batch, M, N, K):
    r = _one(QMatmul(), torch.randn(*a_shape), torch.randn(*b_shape))
    assert (r.kind, r.batch, r.M, r.N, r.K, r.macs) == ("matmul", batch, M, N, K, batch * M * N * K)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(torch.relu(self.fc(self.conv(x).flatten(1))))


def test_keys_formats_and_costs():
    model = _Net()
    gf = GemmFormats.uniform(SplitMac(P8, P16))
    recipe = Recipe(gemm=gf).override("fc", linear=GemmFormats(fwd_mac=FusedMac(P16)))
    convert_(model, recipe.override("head", linear=None))
    p = profile(model, torch.randn(2, 3, 4, 4), cost={P8: 1.0, FusedMac(P16): 2.0, None: 4.0})
    assert [r.name for r in p.records] == ["conv", "fc", "head"]
    conv, fc, head = p.records
    assert conv.formats is gf and conv.key == SplitMac(P8, P16)
    assert fc.key == FusedMac(P16) and head.formats is None and head.key is None
    assert p.by_format() == {SplitMac(P8, P16): conv.macs, FusedMac(P16): fc.macs, None: head.macs}
    # the mac first, then its product format: SplitMac(P8, P16) is priced as P8
    assert p.total_cost == conv.macs * 1.0 + fc.macs * 2.0 + head.macs * 4.0
    assert p.macs == conv.macs + fc.macs + head.macs
    table = p.table()
    assert "conv" in table and "total cost" in table and "not counted" in table


def test_a_missing_cost_and_no_table():
    p = profile(nn.Linear(4, 4), torch.randn(2, 4), cost={P8: 1.0})
    with pytest.raises(KeyError, match="no entry for None"):
        p.total_cost  # noqa: B018 -- the property raises
    with pytest.raises(ValueError, match="cost="):
        profile(nn.Linear(4, 4), torch.randn(2, 4)).total_cost  # noqa: B018
    assert "total cost" not in profile(nn.Linear(4, 4), torch.randn(2, 4)).table()


def test_hand_built_and_block_layers():
    custom = QLinear(4, 4, formats=binaryK_gemm_formats(mul_K=8, mul_P=4))
    assert _one(custom, torch.randn(2, 4)).key == "custom"
    assert _one(QLinear(4, 4), torch.randn(2, 4)).key is None  # no hooks: PyTorch's product
    mx = nn.Linear(64, 8)
    convert_(mx, presets.mx())
    p = profile(mx, torch.randn(2, 64), cost={MXFP8_E4M3: 0.5})
    assert p.records[0].key == BlockMac(MXFP8_E4M3) and p.total_cost == 0.5 * 2 * 64 * 8


def test_each_call_is_a_record_and_the_hooks_are_removed():
    shared = nn.Linear(4, 4)
    model = nn.Sequential(shared, nn.ReLU(), shared)
    p = profile(model, torch.randn(3, 4))
    assert [r.name for r in p.records] == ["0", "0"]  # one module, two calls
    assert not shared._forward_hooks

    class Fails(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            self.fc(x)
            raise RuntimeError("boom")

    failing = Fails()
    with pytest.raises(RuntimeError, match="boom"):
        profile(failing, torch.randn(2, 4))
    assert not failing.fc._forward_hooks


def test_keyword_inputs_and_no_grad():
    class Kw(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.grad_enabled = None

        def forward(self, x, *, scale):
            self.grad_enabled = torch.is_grad_enabled()
            return self.fc(x) * scale

    model = Kw()
    p = profile(model, torch.randn(2, 4), scale=2.0)
    assert p.records[0].macs == 2 * 4 * 4 and model.grad_enabled is False
