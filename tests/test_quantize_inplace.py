"""
The in-place elementwise quantizers, ``binaryK_quantize_`` and
``superfp_quantize_`` (``mptorch/quant/ops.py``), and ``Quant(inplace=True)``.

An in-place quantizer is its out-of-place op with the output pointer set to
the input's: the same kernels, the same casts, and under ``RoundMode.SR`` the
same random word per element, since the word is keyed on the element's index.
So the reference for every value is the out-of-place op, word for word, and
what is left to check is everything the out-of-place op does *around* its
kernel that an in-place one cannot do, or has to do itself:

* **No copy.** The out-of-place ops make a strided input contiguous, clone a
  CUDA view that does not start on a 16-byte boundary (the kernel's vector
  loads fault on one), and widen a narrower tensor for
  ``carrier=torch.float64``. Each of those hands the kernel a copy, which an
  in-place op would round while the caller's tensor stayed as it was. The
  in-place ops refuse all three, and leave the tensor untouched when they do.
* **The version counter.** The kernels write through ``data_ptr()`` and go
  around ATen, so nothing bumps the tensor's version unless the ops are
  registered under ``ADInplaceOrView`` (``csrc/autograd_ops.cpp``). Without the
  bump, a tensor saved for backward and then quantized in place is silently
  used with its new values.
* **No allocation**, which is the point of the ops.
* **MPS.** There is no Metal kernel yet; the ops raise and name the
  out-of-place op (``dev/continuation_plan.md``, phase H).
"""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from mptorch import BinaryK, SuperFP
from mptorch.number import FormatRangeWarning, RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import (
    Quant,
    binaryK_quantize,
    binaryK_quantize_,
    superfp_quantize,
    superfp_quantize_,
)
from tests.markers import available_devices, float64_devices, requires_cuda, requires_mps

# 100_003 is prime, so no vector width (4 floats, 2 doubles, 8 halves) divides
# it and the kernels' scalar tail runs on every backend and dtype.
SIZE = 100_003

DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
_WORDS = {
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float64: torch.int64,
}

# (out-of-place, in-place, format arguments): binaryK K=8, P=4 and superfp
# m3e4n1b7, both of which binary32 carries.
OPS: dict[str, tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor], dict[str, Any]]] = {
    "binaryK": (binaryK_quantize, binaryK_quantize_, {"K": 8, "P": 4}),
    "superfp": (
        superfp_quantize,
        superfp_quantize_,
        {"man_bits": 3, "exp_bits": 4, "normal_binades": 1, "bias": 7},
    ),
}
OP_NAMES = list(OPS)

# The in-place ops run on these; on MPS they raise (test_mps_raises).
_devices = [d for d in available_devices if "mps" not in str(d)]


def _same_words(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit equality, which sees a NaN's payload and a zero's sign."""
    return torch.equal(a.view(_WORDS[a.dtype]), b.view(_WORDS[b.dtype]))


def _input(device: str, dtype: torch.dtype, size: int = SIZE) -> torch.Tensor:
    """Values across and past both formats' ranges, with the specials."""
    x = torch.randn(size, device=device) * torch.logspace(-6, 6, size, device=device)
    x = x.to(dtype)
    x[:6] = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 448.0])
    return x


# --- the values: the out-of-place op's, word for word ------------------------------


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("saturation_mode", list(SaturationMode), ids=lambda m: m.name)
@pytest.mark.parametrize("rounding_mode", list(RoundMode), ids=lambda m: m.name)
def test_matches_out_of_place(device, dtype, op, saturation_mode, rounding_mode):
    """Every dtype, round mode and saturation mode, SR under one seed: the
    tensor ends up holding what the out-of-place op returns, vector body and
    scalar tail alike, and is the tensor returned."""
    quantize, quantize_, fmt = OPS[op]
    kw = dict(
        fmt,
        prng_bits=4 if rounding_mode is RoundMode.SR else 0,
        rounding_mode=rounding_mode,
        saturation_mode=saturation_mode,
    )
    x = _input(device, dtype)
    torch.manual_seed(7)
    expected = quantize(x, **kw)
    y = x.clone()
    torch.manual_seed(7)
    out = quantize_(y, **kw)
    assert out is y
    assert _same_words(y, expected)


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("subnormals_mode", list(SubnormalsMode), ids=lambda m: m.name)
def test_binaryK_subnormals_modes(device, subnormals_mode):
    """The one schema argument superfp does not have reaches the kernel."""
    x = _input(device, torch.float32)
    expected = binaryK_quantize(x, 8, 4, subnormals_mode=subnormals_mode)
    assert _same_words(
        binaryK_quantize_(x.clone(), 8, 4, subnormals_mode=subnormals_mode), expected
    )


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_float64_rounds_in_binary64(device, op):
    """A float64 tensor is rounded in place in binary64, with no narrowing on
    the way: to a format past binary32's precision the result keeps bits a
    float32 could not hold."""
    fmt = (
        {"K": 40, "P": 30}
        if op == "binaryK"
        else {"man_bits": 29, "exp_bits": 10, "normal_binades": 1000, "bias": 511}
    )
    quantize, quantize_, _ = OPS[op]
    x = (1.0 + torch.rand(SIZE, dtype=torch.float64, device=device)) * 3.0
    expected = quantize(x, **fmt)
    y = x.clone()
    quantize_(y, **fmt)
    assert _same_words(y, expected)
    assert not torch.equal(y, y.float().double())


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_shapes(device, op):
    """A multi-dimensional tensor, an empty one, and one shorter than a
    vector."""
    quantize, quantize_, fmt = OPS[op]
    for shape in [(7, 11, 13), (0,), (3, 0, 5), (1,), (3,)]:
        x = torch.randn(shape, device=device)
        expected = quantize(x, **fmt)
        y = x.clone()
        assert quantize_(y, **fmt) is y
        assert y.shape == x.shape and _same_words(y, expected)


# --- what the out-of-place op copies, the in-place op refuses ------------------------


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_non_contiguous_raises(device, op):
    """A strided tensor raises, names the out-of-place op, and is untouched."""
    _, quantize_, fmt = OPS[op]
    base = torch.randn(16, 16, device=device)
    before = base.clone()
    for view in (base.t(), base[:, ::2], base[::2]):
        with pytest.raises(RuntimeError, match=f"contiguous.*use {op}_quant,"):
            quantize_(view, **fmt)
    assert _same_words(base, before)


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("op", OP_NAMES)
def test_cuda_misaligned_view_raises(dtype, op):
    """``x[1:]`` is contiguous but starts off a 16-byte boundary, which the
    CUDA kernel's vector loads cannot address. The out-of-place op clones such
    a view; in place that would round the clone, so the op raises instead,
    before any launch (a launch would fault, and that error is sticky)."""
    _, quantize_, fmt = OPS[op]
    x = _input("cuda", dtype, 1024)
    before = x.clone()
    with pytest.raises(RuntimeError, match=f"16-byte boundary.*use {op}_quant,"):
        quantize_(x[1:], **fmt)
    assert _same_words(x, before)
    # The context survived: a misaligned launch would have cost it.
    torch.cuda.synchronize()


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("op", OP_NAMES)
def test_aligned_view_writes_only_the_view(device, dtype, op):
    """A contiguous view the kernel can address is rounded in place, and the
    rest of its storage is left alone: ``x[16:-3]`` starts on a 16-byte
    boundary in every dtype, and ends mid-vector."""
    quantize, quantize_, fmt = OPS[op]
    x = _input(device, dtype, 1024)
    before = x.clone()
    expected = quantize(x[16:-3], **fmt)
    quantize_(x[16:-3], **fmt)
    assert _same_words(x[16:-3], expected)
    assert _same_words(x[:16], before[:16]) and _same_words(x[-3:], before[-3:])


def test_cpu_view_at_any_offset():
    """The CPU kernel loads one element at a time, so any contiguous view
    does, ``x[1:]`` included."""
    x = _input("cpu", torch.float32, 1024)
    before = x.clone()
    binaryK_quantize_(x[1:], 8, 4)
    assert _same_words(x[1:], binaryK_quantize(before[1:], 8, 4))
    assert _same_words(x[:1], before[:1])


# --- the carrier ---------------------------------------------------------------------


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("op", OP_NAMES)
def test_widening_carrier_raises(device, dtype, op):
    """``carrier=torch.float64`` on a narrower tensor rounds a float64 copy and
    narrows the result, which cannot happen in place: refused, naming the
    out-of-place spelling, with the tensor untouched."""
    _, quantize_, fmt = OPS[op]
    x = _input(device, dtype, 64)
    before = x.clone()
    with pytest.raises(ValueError, match=rf"in place.*use {op}_quantize\(x"):
        quantize_(x, **fmt, carrier=torch.float64)
    assert _same_words(x, before)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_own_carrier_is_accepted(device, op):
    """Naming the carrier a tensor already has changes nothing, and a carrier
    narrower than the tensor, or no carrier at all, raises as it does out of
    place."""
    quantize, quantize_, fmt = OPS[op]
    for dtype, carrier in [(torch.float32, torch.float32), (torch.float64, torch.float64)]:
        x = _input(device, dtype, 64)
        expected = quantize(x, **fmt)
        assert _same_words(quantize_(x, **fmt, carrier=carrier), expected)
    x = _input(device, torch.float64, 64)
    with pytest.raises(ValueError, match="narrower than float64"):
        quantize_(x, **fmt, carrier=torch.float32)
    with pytest.raises(TypeError, match="torch.dtype"):
        quantize_(x, **fmt, carrier="float64")


@pytest.mark.parametrize("device", _devices)
def test_format_checks_run_first(device):
    """The per-call format checks are the out-of-place op's: a format the
    carrier cannot hold raises before anything is written, and one whose edge
    lands off a narrow storage dtype's grid warns."""
    x = _input(device, torch.float32, 64)
    before = x.clone()
    with pytest.raises(ValueError):
        binaryK_quantize_(x, 40, 30)  # 30 bits of precision in binary32
    assert _same_words(x, before)
    with pytest.warns(FormatRangeWarning, match="float16"):
        superfp_quantize_(x.half(), 2, 5, 1, 15)


# --- autograd --------------------------------------------------------------------------


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_requires_grad_raises(device, op):
    """Not differentiable, like the out-of-place op: raises under grad mode and
    names ``Quantizer``; under ``no_grad`` it is a plain write."""
    quantize, quantize_, fmt = OPS[op]
    x = torch.randn(64, device=device, requires_grad=True)
    expected = quantize(x.detach(), **fmt)
    with pytest.raises(RuntimeError, match="not differentiable.*Quantizer"):
        quantize_(x, **fmt)
    with torch.no_grad():
        quantize_(x, **fmt)
    assert _same_words(x.detach(), expected)


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("op", OP_NAMES)
def test_version_counter(device, op):
    """One call is one version. The kernels write through ``data_ptr()``, so
    this is the ``ADInplaceOrView`` registration and nothing else."""
    _, quantize_, fmt = OPS[op]
    x = torch.randn(64, device=device)
    version = x._version
    quantize_(x, **fmt)
    assert x._version == version + 1
    quantize_(x[16:], **fmt)  # a view shares its base's counter
    assert x._version == version + 2


@pytest.mark.parametrize("device", _devices)
def test_saved_tensor_modified_in_place_is_caught(device):
    """What the version counter is for: ``a * w`` saves ``w`` for ``a``'s
    gradient, and quantizing ``w`` in place before ``backward`` must raise
    rather than differentiate against the rounded values."""
    a = torch.randn(64, device=device, requires_grad=True)
    w = torch.randn(64, device=device)
    y = (a * w).sum()
    binaryK_quantize_(w, 8, 4)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.backward()


# --- memory ------------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("op", OP_NAMES)
@pytest.mark.parametrize("rounding_mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
def test_cuda_allocates_nothing(op, rounding_mode):
    """The out-of-place op's peak is one more tensor; the in-place op's is
    none, SR included (its draws are generated in the kernel)."""
    quantize, quantize_, fmt = OPS[op]
    kw = dict(fmt, rounding_mode=rounding_mode, prng_bits=2 if rounding_mode is RoundMode.SR else 0)
    x = torch.randn(1 << 20, device="cuda")
    nbytes = x.numel() * x.element_size()

    def peak_over(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        out = fn()
        torch.cuda.synchronize()
        del out
        return torch.cuda.max_memory_allocated() - start

    assert peak_over(lambda: quantize(x, **kw)) >= nbytes
    assert peak_over(lambda: quantize_(x, **kw)) == 0


# --- Quant(inplace=True) ------------------------------------------------------------------


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize(
    "fmt", [BinaryK(8, 4), BinaryK(8, 3, prng_bits=4), SuperFP(3, 4, 8, 7)], ids=str
)
@pytest.mark.parametrize(
    "rounding", [RoundMode.RNE, RoundMode.RZ, RoundMode.SR], ids=lambda m: m.name
)
def test_quant_inplace(device, fmt, rounding):
    """``Quant(fmt, rounding, inplace=True)`` rounds its argument to what
    ``Quant(fmt, rounding)`` returns, and returns the argument."""
    x = _input(device, torch.float32, 4099)
    torch.manual_seed(11)
    expected = Quant(fmt, rounding)(x)
    y = x.clone()
    torch.manual_seed(11)
    out = Quant(fmt, rounding, inplace=True)(y)
    assert out is y
    assert _same_words(y, expected)
    assert not _same_words(x, expected)  # the out-of-place call left x alone


def test_quant_inplace_is_part_of_the_value():
    """``inplace`` is a field like ``carrier``: two ``Quant``s that differ in
    it are different keys, and the widening carrier is refused at the call."""
    fmt = BinaryK(8, 4)
    assert Quant(fmt, inplace=True) != Quant(fmt)
    assert Quant(fmt, inplace=True) == Quant(fmt, inplace=True)
    assert Quant(fmt, inplace=False) == Quant(fmt)
    assert len({Quant(fmt), Quant(fmt, inplace=True)}) == 2
    with pytest.raises(ValueError, match="in place"):
        Quant(fmt, carrier=torch.float64, inplace=True)(torch.randn(8))


# --- MPS -------------------------------------------------------------------------------------


@requires_mps
@pytest.mark.parametrize("op", OP_NAMES)
def test_mps_raises(op):
    """No Metal kernel yet: the op says so, names the out-of-place op, and
    leaves the tensor as it was."""
    _, quantize_, fmt = OPS[op]
    x = torch.randn(64, device="mps")
    before = x.clone()
    with pytest.raises(RuntimeError, match=f"no MPS kernel yet.*use {op}_quant,"):
        quantize_(x, **fmt)
    assert torch.equal(x, before)
