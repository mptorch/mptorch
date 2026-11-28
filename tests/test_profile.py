"""
``profile``: a model's multiply-accumulates and elements, by the formats each computes in.

``mptorch/quant/instrument/profiling.py`` counts what a ``simulate`` sees during one
forward pass, and the converted layers from their formats: each product as
the GEMM the kernels run, each sum of a softmax, a norm or a reduction as a
GEMM with ones, each elementwise call as its elements. These tests count the
MACs by hand for each kind of layer and shape (leading dimensions, groups,
stride, dilation and padding, an unbatched input, a broadcast matmul), for
the mini models and for attention, then check what a record is filed under,
how a cost table prices it, that a converted model and a simulated one count
alike, that a fake pass counts as a real one, the backward's steps, and that
the hooks, the buffers and the generators are as they were afterwards.
"""

import collections
import math

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

from mptorch import MXFP8_E4M3, NVFP4, BinaryK, RoundMode
from mptorch.quant import (
    AttentionFormats,
    BlockMac,
    BlockQuant,
    FusedMac,
    GemmFormats,
    NormFormats,
    QLinear,
    QMatmul,
    QMultiheadAttention,
    QSoftmax,
    Quant,
    Recipe,
    SoftmaxFormats,
    SplitMac,
    binaryK_gemm_formats,
    convert,
    convert_,
    presets,
    profile,
    register,
)
from mptorch.quant.instrument.conversion import _REGISTRY
from tests.models import MiniGPT2, MiniLlama

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
    assert "conv" in table and "total cost" in table
    assert "torch.relu" in table  # the element table


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


# --- profile on the mode: what a simulate sees ------------------------------------------

AF = AttentionFormats.uniform(SplitMac(P8, P16))


def _llama_macs(T=16, B=2, L=8, hidden=32, layers=2, heads=4, kv=2, inter=64, vocab=64):
    """MiniLlama's forward MACs by hand: per layer seven projections, the core
    (scores, the softmax's sum, the output, over the query heads) and two
    RMSNorm means; then the last norm and lm_head."""
    d = hidden // heads
    projections = T * hidden * (2 * hidden + 2 * kv * d) + 3 * T * hidden * inter
    core = B * heads * (L * L * d + L * L + L * L * d)
    norms = 2 * T * hidden
    return layers * (projections + core + norms) + T * hidden + T * hidden * vocab


@pytest.mark.parametrize("backward", [False, True])
def test_the_mini_llama_by_hand_and_its_eager_twin(backward):
    """SDPA under the attention kind and eager attention under attention_parts
    count the same MACs, each under the same key, forward and backward; GQA's
    keys are counted per query head, as they are repeated."""
    ids = torch.randint(0, 64, (2, 8))
    profiles = {}
    for attn in ("sdpa", "eager"):
        recipe = presets.uniform(P8)
        recipe = (
            recipe.override("*attn", attention=AF)
            if attn == "sdpa"
            else recipe.attention_parts("*attn", AF)
        )
        profiles[attn] = profile(MiniLlama(attn=attn), ids, recipe=recipe, backward=backward)
    sdpa, eager = profiles["sdpa"], profiles["eager"]
    if not backward:
        assert sdpa.macs == _llama_macs()
    assert sdpa.by_format() == eager.by_format()
    core = [r for r in sdpa.records if r.kind == "attention" and r.step == "fwd"]
    assert [(r.site, r.batch, r.M, r.N, r.K) for r in core[:3]] == [
        ("qk", 8, 8, 8, 8),
        ("softmax", 1, 1, 64, 8),
        ("pv", 8, 8, 8, 8),
    ]
    assert AF.qk is not None and AF.softmax is not None
    assert core[0].key == AF.qk.fwd_mac and core[1].key == AF.softmax.acc
    norms = [r for r in sdpa.records if r.kind == "reduction"]  # LlamaRMSNorm's mean
    assert len(norms) == 5 and all(r.key is None for r in norms)


def test_the_mini_gpt2s_conv1d_and_tied_head():
    """GPT-2's Conv1D is an addmm over its weight, counted as the linear it is;
    the head, tied to the embedding, is counted under its own name."""
    p = profile(MiniGPT2(), torch.randint(0, 64, (2, 8)), recipe=presets.uniform(P8))
    by_name = {r.name: r for r in p.records if r.kind == "linear"}
    c_attn = by_name["transformer.h.0.attn.c_attn"]
    assert (c_attn.batch, c_attn.M, c_attn.N, c_attn.K, c_attn.call) == (
        1,
        16,
        96,
        32,
        "torch.addmm",
    )
    head = by_name["lm_head"]
    assert (head.M, head.N, head.K) == (16, 64, 32) and head.key == SplitMac(P8, None)


def _key_counts(p):
    return collections.Counter((r.kind, r.macs, r.key, r.step) for r in p.records)


@pytest.mark.parametrize("name", ["resnet18", "vit_b_16"])
def test_converted_and_simulated_count_alike(name):
    """On torchvision's models, built on the meta device and counted without
    computing, a converted model and the simulated one count the same
    products under the same keys, forward and backward."""
    torchvision = pytest.importorskip("torchvision")
    with torch.device("meta"):
        model = getattr(torchvision.models, name)()
    recipe = presets.uniform(P8, kinds=("linear", "conv", "attention"))
    x = torch.empty(2, 3, 224, 224, device="meta")
    simulated = profile(model, x, recipe=recipe, backward=True, run=False)
    converted, _ = convert(model, recipe)
    assert _key_counts(profile(converted, x, backward=True, run=False)) == _key_counts(simulated)
    assert all(r.formats is not None for r in simulated.records if r.kind == "conv")


@pytest.mark.parametrize("backward", [False, True])
def test_a_fake_pass_counts_as_a_real_one(backward):
    """run=False counts what run=True does, under stochastic rounding too; so
    does a model built on the meta device."""
    sr = BinaryK(8, 4, prng_bits=8)
    gf = GemmFormats.uniform(SplitMac(sr, BinaryK(16, 11, prng_bits=8), rounding=RoundMode.SR))
    recipe = Recipe(gemm=gf, attention=AttentionFormats(qk=gf, pv=gf))
    ids = torch.randint(0, 64, (2, 8))
    torch.manual_seed(0)
    model = MiniLlama(attn="sdpa")
    real = profile(model, ids, recipe=recipe, backward=backward)
    fake = profile(model, ids, recipe=recipe, backward=backward, run=False)
    assert real.records == fake.records and real.elements == fake.elements
    with torch.device("meta"):
        meta = MiniLlama(attn="sdpa")
    assert profile(meta, ids, recipe=recipe, backward=backward, run=False).records == real.records


class _Frozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 8)
        self.head = nn.Linear(8, 2)
        self.side = nn.Linear(8, 8)

    def forward(self, x):
        h = self.fc(self.conv(x).flatten(1))
        with torch.no_grad():
            self.side(h)  # no gradient flows here
        return self.head(h)


def test_the_backwards_steps():
    """Each gradient is counted only for an operand that needs one: not the
    first convolution's input (the data), not a frozen layer's weight, nothing
    inside no_grad; and each step is filed under its own mac."""
    model = _Frozen()
    model.fc.weight.requires_grad_(False)
    gf = GemmFormats(fwd_mac=SplitMac(P8, P16), igrad_mac=FusedMac(P16), wgrad_mac=FusedMac(P8))
    p = profile(model, torch.randn(2, 3, 4, 4), recipe=Recipe(gemm=gf), backward=True)
    steps = {(r.name, r.step) for r in p.records}
    assert ("conv", "igrad") not in steps and ("conv", "wgrad") in steps
    assert ("fc", "wgrad") not in steps and ("fc", "igrad") in steps
    assert {s for n, s in steps if n == "side"} == {"fwd"}
    keys = {r.step: r.key for r in p.records if r.name == "head"}
    assert keys == {"fwd": SplitMac(P8, P16), "igrad": FusedMac(P16), "wgrad": FusedMac(P8)}
    fwd = {r.name: r.macs for r in p.records if r.step == "fwd"}
    assert {r.name: r.macs for r in p.records if r.step == "wgrad"}["head"] == fwd["head"]


def _brute_igrad(size, kernel, stride, padding, dilation):
    """The input gradient's MACs per (input, output) channel pair as the
    kernels run it: every input position with every tap whose residue class
    of the stride is the position's."""
    total = 1
    for n, k, s, p, d in zip(size, kernel, stride, padding, dilation, strict=True):
        total *= sum((h + p - j * d) % s == 0 for h in range(n) for j in range(k))
    return total


@pytest.mark.parametrize(
    "size,kernel,stride,padding,dilation",
    [
        ((32,), (3,), (1,), (0,), (1,)),  # valid: 96 against the forward's 90
        ((9, 8), (3, 3), (1, 1), (1, 1), (1, 1)),
        ((9, 8), (3, 3), (2, 2), (1, 1), (1, 1)),
        ((8, 8), (1, 1), (2, 2), (0, 0), (1, 1)),
        ((11, 10), (3, 2), (2, 2), (0, 1), (2, 2)),
        ((5, 6, 7), (3, 3, 3), (2, 1, 3), (1, 0, 1), (1, 2, 1)),
    ],
    ids=str,
)
def test_a_convolutions_input_gradient(size, kernel, stride, padding, dilation):
    """The input gradient is one GEMM per residue class of the stride, over
    the taps that reach its positions: as many MACs as a brute-force count."""
    conv_cls = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[len(size)]
    conv = conv_cls(4, 6, kernel, stride=stride, padding=padding, dilation=dilation, groups=2)
    x = torch.randn(3, 4, *size, requires_grad=True)
    p = profile(conv, x, backward=True)
    igrad = sum(r.macs for r in p.records if r.step == "igrad")
    per_pair = _brute_igrad(size, kernel, stride, padding, dilation)
    assert igrad == 3 * 2 * (4 // 2) * (6 // 2) * per_pair


_MHA_CASES = {
    "self": ({}, "same"),
    "k_is_v": ({}, "kv"),
    "different": ({}, "none"),
    "kdim_vdim": ({"kdim": 12, "vdim": 10}, "none"),
    "bias_kv_zero_attn": ({"add_bias_kv": True, "add_zero_attn": True}, "same"),
    "batch_first": ({"batch_first": True}, "same"),
}


@pytest.mark.parametrize("case", list(_MHA_CASES))
@pytest.mark.parametrize("need_weights", [False, True])
def test_multihead_attention_converted_and_simulated(case, need_weights):
    """nn.MultiheadAttention's projections and core count alike converted and
    simulated, whichever of torch's paths the call takes."""
    kwargs, same = _MHA_CASES[case]
    torch.manual_seed(0)
    model = nn.Sequential()
    model.add_module("attn", nn.MultiheadAttention(16, 4, **kwargs))

    class Call(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, q, k, v):
            if same == "same":
                return self.inner(q, q, q, need_weights=need_weights)[0]
            if same == "kv":
                return self.inner(q, k, k, need_weights=need_weights)[0]
            return self.inner(q, k, v, need_weights=need_weights)[0]

    m = Call(model.attn)
    q = torch.randn(5, 2, 16)
    k = torch.randn(7, 2, kwargs.get("kdim", 16))
    v = torch.randn(7, 2, kwargs.get("vdim", 16))
    if kwargs.get("batch_first"):
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
    recipe = presets.uniform(P8, kinds=("linear", "attention"))
    simulated = profile(m, q, k, v, recipe=recipe, backward=True)
    converted, _ = convert(m, recipe)
    assert _key_counts(profile(converted, q, k, v, backward=True)) == _key_counts(simulated)
    in_proj = [r for r in simulated.records if r.kind == "linear" and r.name == "inner"]
    expected = {"same": 1, "kv": 2, "none": 3}[same]
    assert len([r for r in in_proj if r.step == "fwd"]) == expected
    keys = 7 + 2 if case == "bias_kv_zero_attn" else 7
    if same == "same":
        keys = 5 + 2 if case == "bias_kv_zero_attn" else 5
    (scores,) = [r for r in simulated.records if r.site == "qk" and r.step == "fwd"]
    assert (scores.batch, scores.M, scores.N, scores.K) == (2 * 4, 5, keys, 4)


def test_nothing_counted_twice():
    """A converted layer without formats calls torch where the mode sees it, and
    is counted once, by its hook; each spelling of a product is one call."""
    soft = nn.Sequential(QSoftmax(dim=-1))
    p = profile(soft, torch.randn(3, 5))
    assert len(p.records) == 1 and len(p.elements) == 1

    class SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = QMultiheadAttention(16, 4)  # no formats: torch's forward, seen by the mode

        def forward(self, x):
            return self.mha(x, x, x)[0]

    p = profile(SelfAttention(), torch.randn(5, 2, 16))
    assert len([r for r in p.records if r.site == "qk"]) == 1

    class Spellings(nn.Module):
        def forward(self, a, b):
            return a @ b, torch.matmul(a, b), a.matmul(b), torch.mm(a, b)

    p = profile(Spellings(), torch.randn(3, 4), torch.randn(4, 5))
    # `a @ b` reaches the mode as Tensor.matmul
    assert [r.call for r in p.records] == [
        "Tensor.matmul",
        "torch.matmul",
        "Tensor.matmul",
        "torch.mm",
    ]


class _Einsum(nn.Module):
    def forward(self, a, b):
        return torch.einsum("ij,jk->ik", a, b)


def test_what_is_not_counted():
    """An einsum is listed, not counted; so is a module simulated whole, whose
    elements are counted and whose children are counted once."""
    p = profile(_Einsum(), torch.randn(3, 4), torch.randn(4, 5))
    assert p.records == () and [u.call for u in p.uncounted] == ["torch.einsum"]
    assert "not counted: torch.einsum" in p.table()

    from tests.test_simulate import _Block, _block_forward

    register(_Block, kind="norm", forward=_block_forward)
    try:
        model = nn.Sequential(_Block())
        norm = NormFormats.uniform(SplitMac(P16, P16), Quant(P16))
        recipe = Recipe(linear=GemmFormats.uniform(SplitMac(P8, P16)), norm=norm)
        for counted in (
            profile(model, torch.randn(3, 8), recipe=recipe),
            profile(convert(model, recipe)[0], torch.randn(3, 8)),
        ):
            assert [r.name for r in counted.records] == ["0.lin"]
            assert [(e.name, e.kind) for e in counted.elements] == [("0", "norm")]
            assert [u.name for u in counted.uncounted] == ["0"]
    finally:
        del _REGISTRY[_Block]


def test_custom_steps_and_a_fused_mac_without_products():
    """A hand-built layer's steps are custom where it has hooks; a sum on a
    FusedMac with no fma is priced as the carrier's arithmetic, torch's."""
    custom = QLinear(4, 4, formats=binaryK_gemm_formats(mul_K=8, mul_P=4))
    p = profile(custom, torch.randn(2, 4, requires_grad=True), backward=True)
    assert {r.step: r.key for r in p.records} == {
        "fwd": "custom",
        "igrad": "custom",
        "wgrad": "custom",
    }
    soft = SoftmaxFormats(acc=FusedMac(None))
    p = profile(nn.Softmax(-1), torch.randn(3, 5), recipe=Recipe(softmax=soft), cost={None: 4.0})
    assert p.records[0].key == FusedMac(None) and p.total_cost == 15 * 4.0


def test_the_model_and_the_generators_are_left_alone():
    """A batch norm's running statistics and the random generators are as they
    were after both kinds of pass; a block weight's packing is not memoized
    from a fake pass; a value a fake pass cannot read raises; and with
    backward under inference_mode, profile refuses."""
    bn = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).train()
    before = {k: v.clone() for k, v in bn.state_dict().items()}
    sr = BinaryK(8, 4, prng_bits=8)
    recipe = Recipe(
        gemm=GemmFormats.uniform(SplitMac(sr, BinaryK(16, 11, prng_bits=8), rounding=RoundMode.SR))
    )
    x = torch.randn(8, 4)
    state = torch.get_rng_state()
    for run in (True, False):
        profile(bn, x, recipe=recipe, run=run)
        assert all(torch.equal(before[k], v) for k, v in bn.state_dict().items())
        assert torch.equal(torch.get_rng_state(), state)

    q = BlockQuant(MXFP8_E4M3, axis=1)
    w = torch.randn(8, 64)
    with FakeTensorMode(allow_non_fake_inputs=True):
        q.pack(w)  # as a fake pass packs a block layer's real weight
    assert not is_fake(q.pack(w).data)

    nv = nn.Linear(64, 8)
    convert_(nv, Recipe(linear=GemmFormats.uniform(BlockMac(NVFP4))))
    x = torch.randn(2, 64)
    with pytest.raises(RuntimeError, match="run=True"):
        profile(nv, x, run=False)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        profile(nv, x, backward=True)
    assert not torch._C._is_torch_function_mode_enabled()
