"""
``Quant`` resolves its format once, and must then behave as the flat quantizers do.

``mptorch.quant.Quant`` builds its op's arguments and the format's findings
for every carrier and storage dtype when it is made (``ops/_spec.py``'s
``_QuantSpec``, ``ops/block.py``'s ``_BlockQuantSpec``), where
``binaryK_quantize`` and its siblings look them up on every call. These tests
hold the two paths to each other over every family, dtype, carrier and
in-place choice: the same result bit for bit (stochastic rounding under the
same seed), the same warnings, attributed to the same line, and the same
error. The last test checks what the resolution is for under
``torch.compile``: a layer with ``Quant`` slots traces through no
``lru_cache``-wrapped function.
"""

import warnings
from functools import partial

import pytest
import torch

from mptorch import (
    E8M0,
    MXFP8_E4M3,
    NVFP4,
    BinaryK,
    BlockFormat,
    FixedPoint,
    RoundMode,
    SuperFP,
)
from mptorch.quant import (
    QLinear,
    Quant,
    binaryK_gemm_formats,
    binaryK_quantize,
    binaryK_quantize_,
    block_quantize,
    block_quantize_,
    fixedpoint_quantize,
    fixedpoint_quantize_,
    superfp_quantize,
    superfp_quantize_,
)
from tests.markers import available_devices

SR = RoundMode.SR


def _flat(fmt, rounding, carrier, inplace):
    """The flat quantizer call that ``Quant(fmt, rounding, carrier=carrier,
    inplace=inplace)`` stands for. It is a ``partial``, which adds no Python
    frame, so the flat function's warnings name the line that calls it, as
    ``Quant``'s do."""
    if isinstance(fmt, BinaryK):
        return partial(
            binaryK_quantize_ if inplace else binaryK_quantize,
            K=fmt.K,
            P=fmt.P,
            bias=fmt.bias,
            prng_bits=fmt.prng_bits,
            is_signed=fmt.is_signed,
            rounding_mode=rounding,
            saturation_mode=fmt.saturation,
            subnormals_mode=fmt.subnormals,
            carrier=carrier,
        )
    if isinstance(fmt, SuperFP):
        return partial(
            superfp_quantize_ if inplace else superfp_quantize,
            man_bits=fmt.man_bits,
            exp_bits=fmt.exp_bits,
            normal_binades=fmt.normal_binades,
            bias=fmt.bias,
            prng_bits=fmt.prng_bits,
            is_signed=fmt.is_signed,
            rounding_mode=rounding,
            saturation_mode=fmt.saturation,
            carrier=carrier,
        )
    if isinstance(fmt, FixedPoint):
        return partial(
            fixedpoint_quantize_ if inplace else fixedpoint_quantize,
            wl=fmt.wl,
            fl=fmt.fl,
            prng_bits=fmt.prng_bits,
            is_signed=fmt.is_signed,
            symmetric=fmt.symmetric,
            rounding_mode=rounding,
            carrier=carrier,
        )
    return partial(
        block_quantize_ if inplace else block_quantize,
        fmt=fmt,
        axis=-1,
        rounding=rounding,
        carrier=carrier,
    )


# (format, rounding). Each family has a format both carriers hold, one that
# binary32 refuses or warns about and binary64 holds, and one under
# stochastic rounding; the binaryK E5M2 layout warns about float16 storage.
CASES = [
    (BinaryK(8, 4), RoundMode.RNE),
    (BinaryK(8, 4, prng_bits=8), SR),
    (BinaryK(8, 3, bias=15), RoundMode.RU),
    (BinaryK(16, 8, bias=255), RoundMode.RNE),
    (BinaryK(40, 30), RoundMode.RNE),
    (SuperFP(3, 4, 2, 7), RoundMode.RNE),
    (SuperFP(3, 4, 2, 7, prng_bits=8), SR),
    (SuperFP(4, 4, 1, 7), RoundMode.RD),
    (FixedPoint(8, 4), RoundMode.RNE),
    (FixedPoint(8, 4, prng_bits=8), SR),
    (FixedPoint(40, 4), RoundMode.RNE),
    (MXFP8_E4M3, RoundMode.RNE),
    (NVFP4, RoundMode.RNE),
    (BlockFormat(BinaryK(11, 2), E8M0, 32), RoundMode.RNE),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
CARRIERS = [None, torch.float32, torch.float64]


def _seeded(device):
    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)


def _outcome(fn, x, device):
    """Run ``fn(x)`` from this one line, so that a per-call warning names the
    same line whichever path issued it, and return what it did: its result or
    its error, and its warnings."""
    _seeded(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fn(x)
        except Exception as e:  # noqa: BLE001 -- the two paths' errors are compared
            result = (type(e), str(e))
    seen = [(w.category, str(w.message), w.filename, w.lineno) for w in caught]
    return result, seen


def _same(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        return a == b
    return a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fmt,rounding", CASES, ids=lambda v: str(v)[:40])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("carrier", CARRIERS, ids=str)
@pytest.mark.parametrize("inplace", [False, True])
def test_quant_is_the_flat_quantizer(device, fmt, rounding, dtype, carrier, inplace):
    if device == "mps" and (dtype is torch.float64 or carrier is torch.float64):
        pytest.skip("MPS has no float64")
    q = Quant(fmt, rounding, carrier=carrier, inplace=inplace)
    flat = _flat(fmt, rounding, carrier, inplace)
    torch.manual_seed(7)
    x = (torch.randn(6, 64, dtype=torch.float64) * 3).to(dtype).to(device)
    flat_result, flat_warnings = _outcome(flat, x.clone(), device)
    quant_result, quant_warnings = _outcome(q, x.clone(), device)
    assert _same(flat_result, quant_result)
    assert flat_warnings == quant_warnings


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("fmt,rounding", CASES[:1] + CASES[5:6] + CASES[8:9], ids=str)
def test_a_strided_tensor(device, fmt, rounding):
    """Out of place a strided tensor is copied, in place it is refused, by
    both paths alike."""
    x = torch.randn(64, 6, device=device).mT
    for inplace in (False, True):
        q = Quant(fmt, rounding, inplace=inplace)
        flat = _flat(fmt, rounding, None, inplace)
        a, wa = _outcome(flat, x.clone().mT.contiguous().mT, device)
        b, wb = _outcome(q, x.clone().mT.contiguous().mT, device)
        assert _same(a, b) and wa == wb


def test_the_call_reads_no_lru_cache_under_compile(monkeypatch):
    """A layer with ``Quant`` slots traces through no ``lru_cache``-wrapped
    function, so Dynamo has no cache to warn about (it warns once per call
    site and process, so its warning function is recorded directly)."""
    import torch._dynamo.utils

    noticed = []
    monkeypatch.setattr(torch._dynamo.utils, "warn_once", lambda msg, *a, **k: noticed.append(msg))
    torch._dynamo.reset()
    formats = binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=16, acc_P=11)
    formats.input_quant = Quant(BinaryK(8, 4))
    formats.weight_quant = Quant(SuperFP(3, 4, 2, 7))
    formats.igrad_quant = formats.wgrad_quant = Quant(FixedPoint(12, 8), RoundMode.RZ)
    layer = QLinear(16, 8, formats=formats)
    x = torch.randn(4, 16, requires_grad=True)
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    compiled(x).sum().backward()
    assert not [m for m in noticed if "lru_cache" in m]
