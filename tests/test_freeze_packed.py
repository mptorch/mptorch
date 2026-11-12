"""A ``QLinear`` that holds only its block codes: ``freeze_packed_``.

A frozen layer must return what the layer returned before the freeze, word
for word, on every backend and in both carriers. It must also hold nothing of
the float weight, follow ``.to()``, and load from a frozen checkpoint into a
model built for it. Layers that are never frozen must not change at all.
"""

import copy
import dataclasses
import gc
import io
import weakref

import pytest
import torch
from torch import nn

from mptorch import (
    E8M0,
    MXFP4_E2M1,
    MXFP6_E2M3,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    RoundMode,
    SuperFP,
)
from mptorch.quant import (
    BlockPacked,
    BlockQuant,
    QLinear,
    QPackedLinear,
    binaryK_gemm_formats,
    block_gemm_formats,
    block_unpack,
    freeze_packed_,
)

from .markers import available_devices, float64_devices, requires_cuda

ACC = BinaryK(16, 11)


def _square(fmt: BlockFormat) -> BlockFormat:
    return dataclasses.replace(fmt, block_rows=fmt.block_size)


SFP8 = BlockFormat(SuperFP(3, 4, 2, 7), E8M0, 32, elem_max=448.0, nan_code=0x7F)

# weight formats, 1D and square; the input is packed in the 1D one
FORMATS = {
    "mxfp8_e4m3": MXFP8_E4M3,
    "mxfp8_e5m2": MXFP8_E5M2,
    "mxfp6_e2m3": MXFP6_E2M3,
    "mxfp4": MXFP4_E2M1,
    "nvfp4": NVFP4,
    "sq_mxfp8": _square(MXFP8_E4M3),
    "sq_nvfp4": _square(NVFP4),
    "superfp": SFP8,
}

# how the unfrozen layer's weight slot is set, the three cases a freeze packs
# from: unset (the recorded BlockQuant's packing of the weight), a BlockQuant's
# `.pack` (its memoized packing) and any other tensor-valued slot (the
# recorded quant's packing of its result; here the BlockQuant as a fake
# quantizer)
WEIGHT_SLOTS = ["unset", "pack", "fake"]


def _formats(fmt: BlockFormat, slot: str = "unset", *, quant: BlockQuant | None = None, **kw):
    """``block_gemm_formats`` over ``fmt``, its weight slot set as ``slot``
    says, with the weight quantized by ``quant`` (``fmt`` along axis 1 by
    default)."""
    formats = block_gemm_formats(dataclasses.replace(fmt, block_rows=1), fmt, acc=ACC, **kw)
    q = BlockQuant(fmt, axis=1) if quant is None else quant
    if slot == "pack":
        formats.weight_quant = q.pack
    elif slot == "fake":
        formats.weight_quant = q
    return formats


def _layer(formats, device, dtype=torch.float32, *, seed=0, out_features=32, in_features=64):
    torch.manual_seed(seed)
    return QLinear(in_features, out_features, formats=formats).to(device=device, dtype=dtype)


def _input(device, dtype=torch.float32, *, seed=1, batch=4, in_features=64, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, in_features, generator=g, dtype=torch.float32).to(device, dtype)
    return x.requires_grad_(requires_grad)


def _frozen_equals_unfrozen(layer, x):
    with torch.no_grad():
        before = layer(x)
        layer.freeze_packed_()
        after = layer(x)
    assert type(layer) is QPackedLinear
    assert torch.equal(after, before)


# --- the forward ----------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("slot", WEIGHT_SLOTS)
@pytest.mark.parametrize("name", list(FORMATS))
def test_the_frozen_forward_is_the_unfrozen_one(device, name, slot):
    layer = _layer(_formats(FORMATS[name], slot), device)
    _frozen_equals_unfrozen(layer, _input(device))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("epilogue", [False, True])
@pytest.mark.parametrize("tensor_scale", [None, 0.0123])
def test_nvfp4_tensor_scales(device, tensor_scale, epilogue):
    """The derived and a static tensor scale, applied in the decode or to the
    result: the frozen layer records the packing's and reads it back."""
    fmt = _square(NVFP4)
    quant = BlockQuant(fmt, axis=1, tensor_scale=tensor_scale)
    formats = _formats(fmt, "pack", quant=quant, tensor_scale_epilogue=epilogue)
    layer = _layer(formats, device)
    _frozen_equals_unfrozen(layer, _input(device))
    assert layer.packed_weight.tensor_scale != 1.0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "alg, block_size, outer",
    [
        (AccumulateAlgorithm.KAHAN, None, None),
        (AccumulateAlgorithm.BLOCK, 16, BinaryK(16, 11)),
        (AccumulateAlgorithm.TREE, 4, None),
    ],
)
def test_accumulate_algorithms(device, alg, block_size, outer):
    formats = _formats(
        MXFP8_E4M3, accumulate_algorithm=alg, block_size=block_size, outer=outer
    )  # fmt: skip
    _frozen_equals_unfrozen(_layer(formats, device), _input(device))


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("slot", ["unset", "pack"])
def test_binary64(device, slot):
    """A float64 layer packs and multiplies in binary64, and so does a
    float32 one whose formats name that carrier; the packing records it."""
    fmt = _square(NVFP4)
    layer = _layer(_formats(fmt, slot), device, torch.float64)
    _frozen_equals_unfrozen(layer, _input(device, torch.float64))
    assert layer.packed_weight.carrier is torch.float64

    formats = _formats(fmt, carrier=torch.float64)
    layer = _layer(formats, device)
    _frozen_equals_unfrozen(layer, _input(device))
    assert layer.get_extra_state()["carrier"] == "float64"


@pytest.mark.parametrize("device", available_devices)
def test_a_stochastic_packing_is_the_one_the_last_forward_used(device):
    """With the packed-once recipe the freeze takes the memoized packing, so
    under SR the frozen layer returns the last forward's result, draws and
    all, and the memo is cleared so as not to keep the codes alive."""
    fmt = _square(MXFP8_E4M3)
    quant = BlockQuant(fmt, RoundMode.SR, axis=1)
    layer = _layer(_formats(fmt, "pack", quant=quant), device)
    _frozen_equals_unfrozen(layer, _input(device))
    assert quant._memo[0] is None


# --- gradients ------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_a_square_frozen_weight_serves_the_input_gradient(device):
    fmt = _square(NVFP4)
    layer = _layer(_formats(fmt, "pack"), device)
    x = _input(device, requires_grad=True)
    layer(x).square().sum().backward()
    want, x.grad = x.grad, None
    bias_grad = layer.bias.grad
    layer.zero_grad(set_to_none=True)

    layer.freeze_packed_()
    assert layer.weight is None
    assert [name for name, _ in layer.named_parameters()] == ["bias"]
    layer(x).square().sum().backward()
    assert x.grad is not None and want is not None
    assert layer.bias.grad is not None and bias_grad is not None
    assert torch.equal(x.grad, want)
    assert torch.equal(layer.bias.grad, bias_grad)


@pytest.mark.parametrize("device", available_devices)
def test_a_layer_below_a_frozen_one_trains_as_below_its_unfrozen_twin(device):
    fmt = _square(MXFP8_E4M3)
    torch.manual_seed(5)
    below = nn.Linear(16, 64).to(device)
    unfrozen = nn.Sequential(below, _layer(_formats(fmt, "pack"), device))
    frozen = copy.deepcopy(unfrozen)
    freeze_packed_(frozen)
    x = _input(device, in_features=16)
    grads = []
    for model in (unfrozen, frozen):
        model(x).square().sum().backward()
        grads.append([p.grad for p in model[0].parameters()])
    for want, got in zip(*grads, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize("device", available_devices)
def test_a_1d_frozen_weight_refuses_the_input_gradient(device):
    layer = _layer(_formats(MXFP8_E4M3), device)
    layer.freeze_packed_()
    x = _input(device, requires_grad=True)
    out = layer(x)
    with pytest.raises(ValueError, match="freeze it in a square-tile format"):
        out.sum().backward()


# --- state ----------------------------------------------------------------------------


def _mlp(fmt=None, device=None, out_features=8):
    fmt = _square(NVFP4) if fmt is None else fmt
    return nn.Sequential(
        QLinear(64, 64, device=device, formats=_formats(fmt)),
        nn.ReLU(),
        QLinear(64, out_features, device=device, formats=_formats(fmt)),
    )


def _saved(model) -> dict:
    """``model``'s state through a file, loaded as an untrusted one would be."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


def test_the_state_dict_keys():
    model = _mlp()
    assert freeze_packed_(model) == 2
    assert sorted(model[0].state_dict()) == [
        "_extra_state",
        "bias",
        "weight_codes",
        "weight_scales",
    ]
    codes = model[0].weight_codes
    assert codes.dtype == torch.uint8 and model[0].weight_scales.dtype == torch.uint8
    assert model[0].packed_weight.nbytes == codes.numel() + model[0].weight_scales.numel()


@pytest.mark.parametrize("device", available_devices)
def test_a_frozen_checkpoint_loads_into_a_skeleton(device):
    torch.manual_seed(0)
    model = _mlp().to(device)
    x = _input(device)
    freeze_packed_(model)
    with torch.no_grad():
        want = model(x)
    state = _saved(model)

    skeleton = _mlp().to(device)
    assert freeze_packed_(skeleton, pack=False) == 2
    assert not skeleton[0].weight_codes.any()
    skeleton.load_state_dict(state)
    with torch.no_grad():
        assert torch.equal(skeleton(x), want)


def test_a_meta_model_never_allocates_the_float_weights():
    torch.manual_seed(0)
    model = _mlp()
    freeze_packed_(model)
    x = _input("cpu")
    with torch.no_grad():
        want = model(x)

    skeleton = _mlp(device="meta")
    freeze_packed_(skeleton, pack=False)
    assert skeleton[0].weight_codes.is_meta
    skeleton.load_state_dict(_saved(model), assign=True)
    assert skeleton[0].weight_codes.device.type == "cpu"
    with torch.no_grad():
        assert torch.equal(skeleton(x), want)


def test_a_meta_weight_has_nothing_to_pack():
    with pytest.raises(ValueError, match="pack=False"):
        freeze_packed_(_mlp(device="meta"))


def test_a_mismatched_checkpoint_raises_naming_both():
    model = _mlp()
    freeze_packed_(model)
    state = _saved(model)

    other_format = _mlp(_square(MXFP8_E4M3))
    freeze_packed_(other_format, pack=False)
    with pytest.raises(ValueError, match="format .* this layer's"):
        other_format.load_state_dict(state)

    other_shape = _mlp(out_features=16)
    freeze_packed_(other_shape, pack=False)
    with pytest.raises(ValueError, match="shape"):
        other_shape.load_state_dict(state)

    newer = dict(state)
    newer["0._extra_state"] = {**state["0._extra_state"], "layout": 99}
    skeleton = _mlp()
    freeze_packed_(skeleton, pack=False)
    with pytest.raises(ValueError, match="layout version 99"):
        skeleton.load_state_dict(newer)


def test_float_and_frozen_checkpoints_do_not_mix():
    float_model, frozen = _mlp(), _mlp()
    freeze_packed_(frozen)
    with pytest.raises(RuntimeError, match="Unexpected key.*weight_codes"):
        float_model.load_state_dict(frozen.state_dict())
    with pytest.raises(RuntimeError, match='Unexpected key.*"0.weight"'):
        frozen.load_state_dict(_mlp().state_dict())


def test_deepcopy():
    model = _mlp()
    freeze_packed_(model)
    twin = copy.deepcopy(model)
    x = _input("cpu")
    with torch.no_grad():
        assert torch.equal(twin(x), model(x))
    assert twin[0].weight_codes.data_ptr() != model[0].weight_codes.data_ptr()


# --- devices and dtypes ---------------------------------------------------------------


@requires_cuda
def test_the_codes_follow_the_layer_between_devices():
    layer = _layer(_formats(_square(NVFP4), "pack"), "cpu")
    x = _input("cpu")
    _frozen_equals_unfrozen(layer, x)
    with torch.no_grad():
        want = layer(x)
        layer.cuda()
        assert layer.weight_codes.is_cuda and layer.packed_weight.device.type == "cuda"
        assert torch.equal(layer(x.cuda()).cpu(), want)
        layer.cpu()
        assert torch.equal(layer(x), want)


@pytest.mark.filterwarnings("ignore::mptorch.FormatRangeWarning")
def test_half_leaves_the_codes_alone():
    """``.half()`` converts the bias, not the codes; a layer halved before
    the freeze freezes to what it computes in half."""
    layer = _layer(_formats(MXFP8_E4M3), "cpu").half()
    x = _input("cpu", torch.float16)
    _frozen_equals_unfrozen(layer, x)

    frozen = _layer(_formats(MXFP8_E4M3), "cpu")
    frozen.freeze_packed_()
    frozen.half()
    assert frozen.weight_codes.dtype == torch.uint8 and frozen.bias.dtype == torch.float16
    with torch.no_grad():
        assert frozen(x).dtype == torch.float16


# --- memory ---------------------------------------------------------------------------


@pytest.mark.parametrize("pack", [True, False])
def test_the_float_weight_is_freed(pack):
    quant = BlockQuant(MXFP8_E4M3, axis=1)
    layer = _layer(_formats(MXFP8_E4M3, "pack", quant=quant), "cpu")
    with torch.no_grad():
        layer(_input("cpu"))  # fills the BlockQuant's memo
    ref = weakref.ref(layer.weight)
    layer.freeze_packed_(pack=pack)
    gc.collect()
    assert ref() is None
    assert quant._memo[0] is None


@requires_cuda
def test_on_cuda_the_layer_holds_its_codes_and_bias():
    def allocated(*tensors):
        return sum(-(-t.nbytes // 512) * 512 for t in tensors)

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    layer = QLinear(1024, 1024, device="cuda", formats=_formats(_square(NVFP4)))
    assert torch.cuda.memory_allocated() - base == allocated(layer.weight, layer.bias)
    layer.freeze_packed_()
    gc.collect()
    held = allocated(layer.weight_codes, layer.weight_scales, layer.bias)
    assert torch.cuda.memory_allocated() - base == held
    assert held < 4 * 1024 * 1024 / 7


# --- unfreezing -----------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_unfreeze_gives_the_decoded_weight(device):
    layer = _layer(_formats(_square(NVFP4)), device)
    layer.freeze_packed_()
    decoded = block_unpack(layer.packed_weight)
    x = _input(device)
    with torch.no_grad():
        frozen_out = layer(x)
    back = layer.unfreeze_packed_()
    assert back is layer and type(layer) is QLinear
    assert sorted(layer.state_dict()) == ["bias", "weight"]
    assert isinstance(layer.weight, nn.Parameter) and torch.equal(layer.weight, decoded)
    with torch.no_grad():
        # the decoded weight's values are in the format, so packing them again
        # gives them back, and the GEMM sees what the frozen layer saw
        assert torch.equal(layer(x), frozen_out)
    layer(x).sum().backward()
    assert layer.weight.grad is not None


# --- what is left alone ---------------------------------------------------------------


def test_unfrozen_layers_are_untouched():
    layer = _layer(_formats(MXFP8_E4M3, "pack"), "cpu")
    assert type(layer) is QLinear
    assert sorted(layer.state_dict()) == ["bias", "weight"]
    twin = _layer(_formats(MXFP8_E4M3, "pack"), "cpu", seed=3)
    twin.load_state_dict(layer.state_dict())  # strict
    assert torch.equal(twin.weight, layer.weight)


def test_freeze_packed_skips_other_layers():
    model = nn.Sequential(
        QLinear(64, 64, formats=binaryK_gemm_formats(8, 4)),
        QLinear(64, 64),
        nn.Linear(64, 64),
        QLinear(64, 8, formats=_formats(MXFP8_E4M3)),
    )
    assert freeze_packed_(model) == 1
    assert [type(m).__name__ for m in model] == ["QLinear", "QLinear", "Linear", "QPackedLinear"]


# --- errors ---------------------------------------------------------------------------


def test_formats_that_are_not_block_formats_raise():
    layer = QLinear(64, 64, formats=binaryK_gemm_formats(8, 4))
    with pytest.raises(ValueError, match="block_gemm_formats"):
        layer.freeze_packed_()


def test_a_second_freeze_raises():
    layer = _layer(_formats(MXFP8_E4M3), "cpu")
    layer.freeze_packed_()
    with pytest.raises(TypeError, match="frozen already"):
        layer.freeze_packed_()


def test_a_subclass_is_not_switched():
    class Mine(QLinear):
        pass

    layer = Mine(64, 32, formats=_formats(MXFP8_E4M3))
    with pytest.raises(TypeError, match="Mine"):
        layer.freeze_packed_()
    assert freeze_packed_(layer) == 0


def test_a_weight_quant_the_hooks_cannot_take_raises():
    formats = _formats(MXFP8_E4M3)
    formats.weight_quant = lambda w: 3
    with pytest.raises(ValueError, match="returns a int"):
        _layer(formats, "cpu").freeze_packed_()

    formats.weight_quant = BlockQuant(MXFP8_E4M3, axis=0).pack
    with pytest.raises(ValueError, match="axis 1"):
        _layer(formats, "cpu").freeze_packed_()


def test_the_class_is_not_built_directly():
    with pytest.raises(TypeError, match="freeze_packed_"):
        QPackedLinear(64, 32)
    layer = _layer(_formats(MXFP8_E4M3), "cpu")
    layer.freeze_packed_()
    with pytest.raises(RuntimeError, match="unfreeze_packed_"):
        layer.reset_parameters()


def test_repr_names_the_format_and_the_bytes():
    layer = _layer(_formats(MXFP8_E4M3), "cpu")
    layer.freeze_packed_()
    text = repr(layer)
    assert text.startswith("QPackedLinear(") and "packed_bytes=2112" in text
    assert isinstance(layer.packed_weight, BlockPacked)


# --- the class switch under torch.export and torch.compile --------------------------------


def test_a_frozen_layer_exports():
    """In MXFP8: an NVFP4 input's derived tensor scale is read with
    ``.item()``, which export refuses frozen or not."""
    layer = _layer(_formats(_square(MXFP8_E4M3)), "cpu")
    layer.freeze_packed_()
    x = _input("cpu", batch=8)
    ep = torch.export.export(layer, (x,), dynamic_shapes=({0: torch.export.Dim.AUTO},))
    targets = {
        n.target.name()
        for n in ep.graph.nodes
        if n.op == "call_function" and getattr(n.target, "namespace", None) == "mptorch"
    }
    assert "mptorch::custom_matmul_block" in targets
    program = ep.module()
    for batch in (8, 5):
        xb = _input("cpu", batch=batch)
        with torch.no_grad():
            assert torch.equal(program(xb), layer(xb))


# Dynamo warns that it traces through the lru_cache of the format descriptor's
# helpers (block.py's _code_fields); an unfrozen block layer warns the same.
@pytest.mark.filterwarnings("ignore:Dynamo detected a call to a `functools.lru_cache`")
def test_a_frozen_layer_compiles():
    torch._dynamo.reset()
    layer = _layer(_formats(_square(NVFP4)), "cpu")
    layer.freeze_packed_()
    compiled = torch.compile(layer, backend="aot_eager")
    x = _input("cpu")
    with torch.no_grad():
        assert torch.equal(compiled(x), layer(x))
