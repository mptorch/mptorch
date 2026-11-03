"""
TensorQuant: a per-tensor power-of-two scale around an elementwise quantizer.

The scale exponent is the largest ``s`` with ``max |x| * 2**s`` no larger than
the format's largest finite value (Perez et al., arXiv:2309.17224, Eq. 3), over
the tensor's finite elements. Both multiplications by the scale are exact, so a
call is the format's cast of ``x * 2**s``, times ``2**-s``, word for word. The
tests hold it to that reference, with ``s`` found by search and the powers of
two applied in float64 on the CPU, across magnitudes that put ``s`` past
binary32's exponent range. ``max_finite`` is held to the format's own cast.

On MPS the Apple GPU flushes binary32 subnormals in arithmetic, so the cases
whose scaling passes through them run on the CPU and CUDA only.
"""

import math

import pytest
import torch

from mptorch import (
    E4M3,
    E5M2,
    MXFP8_E4M3,
    BinaryK,
    FixedPoint,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
    SuperFP,
)
from mptorch.quant import QAffineFormats, QConv2d, Quant, TensorQuant, max_finite
from mptorch.quant.ops import _narrowed
from mptorch.quant.tensor import _mul_pow2, _scale_exponent
from tests.markers import available_devices, float64_devices

# Formats with and without infinities, signed and unsigned, a binaryK with no
# mantissa, superfp with two normal binades and with eight, and fixed point in
# two's complement, symmetric, unsigned, and with a step coarser than 1.
FORMATS = [
    E4M3,
    E5M2,
    BinaryK(8, 4),
    BinaryK(6, 3),
    BinaryK(6, 1),
    BinaryK(8, 4, is_signed=False),
    BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE),
    BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE_EXTENDED),
    BinaryK(8, 4, bias=7, subnormals=SubnormalsMode.NORMALS),
    SuperFP(3, 4, 2, 7),
    SuperFP(3, 4, 8, 7),
    SuperFP(3, 4, 2, 7, saturation=SaturationMode.SAT_FINITE),
    FixedPoint(8, 4),
    FixedPoint(8, 4, symmetric=True),
    FixedPoint(8, 4, is_signed=False),
    FixedPoint(16, -2),
]
SIGNED_FORMATS = [f for f in FORMATS if f.is_signed]

# Exponents of the tensors' magnitudes: 2**-140 is a binary32 subnormal, and
# both ends put the scale exponent past binary32's normal exponents.
MAGNITUDES = [-140, -40, 0, 40, 120]


def _search(amax: float, top: float) -> int:
    """Eq. 3 by search: the largest ``s`` with ``amax * 2**s <= top``, the
    products taken with ``math.ldexp``, exact near ``top``."""
    s = math.floor(math.log2(top) - math.log2(amax))
    while math.ldexp(amax, s + 1) <= top:
        s += 1
    while math.ldexp(amax, s) > top:
        s -= 1
    return s


def _exponent(x: torch.Tensor, top: float) -> int:
    """:func:`_search` over the finite elements of ``x``."""
    a = x.double().abs()
    return _search(a[torch.isfinite(a)].max().item(), top)


def _reference(x: torch.Tensor, fmt, rounding: RoundMode = RoundMode.RNE) -> torch.Tensor:
    """``Quant(fmt)(x * 2**s) * 2**-s`` on the CPU, the powers of two applied
    in float64 and the result stored in ``x``'s dtype."""
    x = x.cpu()
    s = _exponent(x, max_finite(fmt))
    scaled = (x.double() * 2.0**s).to(x.dtype)
    return (Quant(fmt, rounding)(scaled).double() * 2.0**-s).to(x.dtype)


def _tensor(magnitude: int, n: int = 4096) -> torch.Tensor:
    """Gaussian values times ``2**magnitude`` in float32, spread over 24 more
    binades, with every 7th element zero.

    The spread is wider than an 8-bit format's range, so a scale one binade off
    changes the result."""
    spread = torch.exp2(torch.randint(-24, 1, (n,), dtype=torch.float64))
    x = torch.randn(n, dtype=torch.float64) * spread * 2.0**magnitude
    x[::7] = 0.0
    return x.float()


def _smallest(fmt) -> float:
    """The format's smallest positive value: the tiniest binary64 rounded up."""
    return Quant(fmt, RoundMode.RU)(torch.tensor([2.0**-1074], dtype=torch.float64)).item()


_WORD = {8: torch.int64, 4: torch.int32, 2: torch.int16}


def _assert_same_words(got: torch.Tensor, ref: torch.Tensor, *, nan_payloads: bool = True) -> None:
    """``got`` holds exactly ``ref``'s words: bits rather than values, so that a
    -0.0 and a NaN are compared too.

    ``nan_payloads=False`` accepts any NaN where ``ref`` has one, for a NaN
    that passes through the scaling's multiplications: a GPU's float unit
    returns its canonical NaN, where the CPU's keeps the operand's payload."""
    got, ref = got.cpu(), ref.cpu()
    assert got.dtype == ref.dtype and got.shape == ref.shape
    word = _WORD[ref.element_size()]
    differ = ref.reshape(-1).view(word) != got.reshape(-1).view(word)
    if not nan_payloads:
        differ &= ~(ref.reshape(-1).isnan() & got.reshape(-1).isnan())
    if differ.any():
        i = int(differ.nonzero()[0])
        r, g = ref.reshape(-1)[i], got.reshape(-1)[i]
        raise AssertionError(
            f"{int(differ.sum())} of {differ.numel()} words differ; the first, at {i}: "
            f"want {r.item()!r} ({int(r.view(word)):#x}), got {g.item()!r} ({int(g.view(word)):#x})"
        )


# --- max_finite ------------------------------------------------------------------


def test_max_finite_known_values():
    """The OCP layouts' tops under P3109's conventions, as the tutorial states
    them: E4M3's 448 (its 480 code is the infinity outside the finite domain)
    and E5M2's 98304."""
    assert max_finite(E4M3) == 448.0
    assert max_finite(E5M2) == 98304.0
    assert max_finite(BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE)) == 480.0
    assert max_finite(BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE_EXTENDED)) == 448


@pytest.mark.parametrize("fmt", FORMATS, ids=str)
def test_max_finite_is_the_casts_largest(fmt):
    """``max_finite`` is a value of the format, and no input from half of it to
    twice it rounds to a larger finite value, in binary64."""
    top = max_finite(fmt)
    grid = torch.cat(
        [
            torch.linspace(top / 2, 2 * top, 100_001, dtype=torch.float64),
            torch.tensor([top], dtype=torch.float64),
        ]
    )
    out = Quant(fmt)(grid)
    assert out[-1].item() == top
    assert out[torch.isfinite(out)].max().item() == top


def test_max_finite_refuses_other_formats():
    with pytest.raises(TypeError):
        max_finite(MXFP8_E4M3)


# --- the scale ---------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fmt", SIGNED_FORMATS, ids=str)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
def test_matches_reference(device, fmt, magnitude):
    """A call equals the format's cast of the scaled tensor, unscaled, word for
    word, at every magnitude, including scale exponents past binary32's."""
    if device == "mps" and magnitude < -100:
        pytest.skip("the Apple GPU flushes binary32 subnormals")
    x = _tensor(magnitude)
    _assert_same_words(TensorQuant(fmt)(x.to(device)), _reference(x, fmt))


@pytest.mark.parametrize("top", [max_finite(f) for f in FORMATS] + [1.0, 1.5, 2.0**-100])
def test_scale_exponent(top):
    """``_scale_exponent`` is the search's ``s`` and puts ``amax`` in the top
    binade, ``(top / 2, top]``: for powers of two, for ``top`` itself and the
    values beside it, and for random mantissas across binary32's exponents and
    beyond, binary64's subnormals included."""
    amaxes = [top, math.nextafter(top, 0), math.nextafter(top, math.inf), 5e-324, 1e300]
    amaxes += [2.0**e for e in range(-149, 128, 7)]
    amaxes += [m * 2.0**e for m, e in zip(torch.rand(200).add(0.5).tolist(), range(-200, 200, 2))]
    for amax in amaxes:
        s = _scale_exponent(amax, top)
        assert s == _search(amax, top), amax
        assert top / 2 < math.ldexp(amax, s) <= top, amax


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("s", [-275, -149, -140, -127, -126, 0, 127, 128, 150, 275])
def test_mul_pow2_rounds_once(device, s):
    """``_mul_pow2`` is ``x * 2**s`` rounded once into binary32, for exponents
    past one power of two binary32 holds, both ways: the reference applies
    ``2**s`` exactly in float64 and rounds the product to float32 once. The
    values carry all 24 significand bits, so results that leave binary32's
    normal range have bits to round."""
    if device == "mps" and s != 0:
        pytest.skip("the Apple GPU flushes binary32 subnormals")
    # going down, results near 2**-140, among the subnormals; going up, results
    # in [2**0, 2**127), from inputs as small as binary32's subnormals
    e = min(127, -s - 140) if s < 0 else max(-s, -149)
    x = (torch.rand(4096, dtype=torch.float64).mul(0.5).add(1) * 2.0**e).float()
    ref = (x.double() * 2.0**s).float()
    _assert_same_words(_mul_pow2(x.to(device), s), ref)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fmt", [E4M3, E5M2, SuperFP(3, 4, 2, 7)], ids=str)
def test_exact_power_of_two_boundary(device, fmt):
    """When ``max |x|`` is exactly the format's top times a power of two, the
    scale puts it on the top, not one binade below. When it is one float32
    step above that, the scale is one binade lower.

    ``s`` isn't exposed, so each tensor also holds the format's smallest value
    at the expected scale. At the right ``s`` it comes back unchanged; one
    binade short it halves and rounds to zero. Rounding up catches a scale one
    binade too high: the largest element would round past the format's top."""
    top, smallest = max_finite(fmt), _smallest(fmt)
    at = torch.tensor([top * 2.0**-20, -top * 2.0**-21, smallest * 2.0**-20], device=device)
    _assert_same_words(TensorQuant(fmt)(at), at)
    above = torch.nextafter(at[:1].cpu(), torch.tensor([math.inf]))
    above = torch.cat([above, torch.tensor([smallest * 2.0**-19])])
    for rounding in (RoundMode.RNE, RoundMode.RU):
        got = TensorQuant(fmt, rounding)(above.to(device))
        _assert_same_words(got, _reference(above, fmt, rounding))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "fmt", [BinaryK(8, 4, is_signed=False), FixedPoint(8, 4, is_signed=False)], ids=str
)
def test_unsigned_format(device, fmt):
    """An unsigned format's negative values go to zero, as its cast sends them,
    and the scale is still taken from ``max |x|``."""
    x = _tensor(-20)
    out = TensorQuant(fmt)(x.to(device)).cpu()
    _assert_same_words(out, _reference(x, fmt))
    assert (out[x < 0] == 0).all()


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "fmt", [BinaryK(8, 4, bias=7, prng_bits=16), FixedPoint(8, 4, prng_bits=16)], ids=str
)
def test_stochastic_rounding_is_bracketed_and_unbiased(device, fmt):
    """Under ``RoundMode.SR`` every draw of one value is one of its two
    neighbours at the scale, both are drawn, and the mean over many draws is
    the value, as in tests/test_accumulators.py."""
    n = 20000
    x = torch.full((n,), 1.3 * 2.0**-20)  # off the grid at the scale of both formats
    value = x[0].item()
    s = _exponent(x, max_finite(fmt))
    scaled = x[:1].double() * 2.0**s
    lo = Quant(fmt, RoundMode.RD)(scaled).item() * 2.0**-s
    hi = Quant(fmt, RoundMode.RU)(scaled).item() * 2.0**-s
    out = TensorQuant(fmt, RoundMode.SR)(x.to(device)).cpu()
    assert set(out.unique().tolist()) == {lo, hi}
    assert abs(out.double().mean().item() - value) < 3 * (hi - lo) / (2 * n**0.5)


# --- dtypes and carriers --------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
def test_narrow_dtypes(device, dtype):
    """A float16 or bfloat16 tensor is scaled in float32 and its result stored
    back in its dtype once."""
    x = _tensor(-12).to(dtype)
    out = TensorQuant(E4M3)(x.to(device))
    assert out.dtype is dtype
    _assert_same_words(out, _reference(x.float(), E4M3).to(dtype))


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("magnitude", [-800, 0, 800])
def test_float64(device, magnitude):
    """A float64 tensor is scaled and rounded in binary64, at magnitudes far
    past binary32's exponents."""
    fmt = BinaryK(16, 8)
    x = torch.randn(4096, dtype=torch.float64) * 2.0**magnitude
    out = TensorQuant(fmt)(x.to(device))
    assert out.dtype is torch.float64
    _assert_same_words(out, _reference(x, fmt))


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str)
def test_float64_carrier(device, dtype):
    """``carrier=torch.float64`` is the call on the widened tensor, narrowed
    back once, as in tests/test_block_binary64.py. The format has 26 bits of
    precision, which only binary64 holds, so a call that dropped to binary32
    would be refused rather than return plausible numbers."""
    fmt = BinaryK(32, 26)
    x = _tensor(-20).to(dtype).to(device)
    out = TensorQuant(fmt, carrier=torch.float64)(x)
    assert out.dtype is dtype
    _assert_same_words(out, _narrowed(TensorQuant(fmt)(x.double()), dtype))
    with pytest.raises(ValueError, match="binary64 holds"):
        TensorQuant(fmt)(x)


def test_float32_carrier_refuses_float64():
    with pytest.raises(ValueError):
        TensorQuant(E4M3, carrier=torch.float32)(torch.ones(4, dtype=torch.float64))


# --- tensors with nothing to scale -------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_zero_and_empty(device):
    """A tensor of zeros and an empty tensor are passed to the cast unscaled."""
    zeros = torch.zeros(8, device=device)
    _assert_same_words(TensorQuant(E4M3)(zeros), zeros)
    empty = torch.empty(0, 3, device=device)
    assert TensorQuant(E4M3)(empty).shape == (0, 3)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "fmt", [E4M3, BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE), FixedPoint(8, 4)]
)
def test_non_finite_elements(device, fmt):
    """Infinities and NaNs are left out of ``max |x|``: the finite elements are
    scaled as they would be without them, and the non-finite ones go through
    the cast as they would unscaled."""
    x = _tensor(-20)
    x[:3] = torch.tensor([math.inf, -math.inf, math.nan])
    _assert_same_words(TensorQuant(fmt)(x.to(device)), _reference(x, fmt), nan_payloads=False)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "fmt, ends",
    [
        (E4M3, [math.inf, -math.inf]),
        (BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE), [480.0, -480.0]),
        (FixedPoint(8, 4), [7.9375, -8.0]),
    ],
    ids=["E4M3", "SAT_FINITE", "FixedPoint"],
)
def test_only_non_finite_elements(device, fmt, ends):
    """A tensor with no finite element is not scaled. Where the format
    saturates an infinity it comes back as the format's own end, which any
    scale would move, and a NaN stays a NaN."""
    x = torch.tensor([math.inf, -math.inf, math.nan], device=device)
    out = TensorQuant(fmt)(x).cpu()
    assert out[:2].tolist() == ends
    assert math.isnan(out[2].item())


# --- refusals and the value type ------------------------------------------------------


def test_refuses_formats_without_an_elementwise_scale():
    with pytest.raises(TypeError, match="BlockFormat"):
        TensorQuant(MXFP8_E4M3)
    with pytest.raises(TypeError):
        TensorQuant("E4M3")  # type: ignore[arg-type]


def test_refuses_bad_carriers():
    with pytest.raises(TypeError):
        TensorQuant(E4M3, carrier="float64")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TensorQuant(E4M3, carrier=torch.float16)


def test_equal_by_value():
    """Equal and hashable by value, as :class:`Quant` is."""
    assert TensorQuant(E4M3) == TensorQuant(E4M3)
    assert hash(TensorQuant(E4M3)) == hash(TensorQuant(E4M3))
    assert TensorQuant(E4M3) != TensorQuant(E4M3, RoundMode.RZ)


# --- in a layer -------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_in_a_conv_layer(device):
    """In a ``QConv2d``'s slots the layer computes what a hand-written script
    does with the same quantizers, as tests/test_qconv2d.py's tier 3 does: the
    forward is ``conv2d`` of the scaled input and weight, and the gradients
    are ``torch.nn.grad``'s of the scaled output gradient. The output gradient
    is small enough that E5M2 would flush it to zero unscaled."""
    fwd, bwd = TensorQuant(E4M3), TensorQuant(E5M2)
    formats = QAffineFormats(
        input_quant=fwd, weight_quant=fwd, igrad_quant=bwd, wgrad_quant=bwd, bgrad_quant=bwd
    )
    conv = QConv2d(3, 8, 3, padding=1, formats=formats).to(device)
    assert conv.bias is not None
    x = (torch.randn(2, 3, 8, 8) * 1e-4).to(device).requires_grad_()
    out = conv(x)
    g_out = torch.randn_like(out) * 1e-6
    with torch.no_grad():
        qx, qw, qg = fwd(x), fwd(conv.weight), bwd(g_out)
        ref = torch.nn.functional.conv2d(qx, qw, conv.bias, padding=1)
        igrad = torch.nn.grad.conv2d_input(x.shape, qw, qg, padding=1)
        wgrad = torch.nn.grad.conv2d_weight(qx, conv.weight.shape, qg, padding=1)
        bgrad = qg.sum(dim=(0, 2, 3))
    _assert_same_words(out.detach(), ref)
    out.backward(g_out)
    # tier 3's margin for the backward reducing in another order, taken relative
    # to each gradient's size, since these are far below 1
    for got, want in [(x.grad, igrad), (conv.weight.grad, wgrad), (conv.bias.grad, bgrad)]:
        assert got is not None
        atol = 1e-3 * want.abs().max().item()
        assert torch.allclose(got, want, rtol=1e-3, atol=atol)
