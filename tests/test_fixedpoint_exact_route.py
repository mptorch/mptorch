"""
The exact integer route of the fixed-point GEMM (``mptorch/quant/_exact.py``).

A recipe's pass whose operands are values of fixed-point formats of at most 8
bits, and whose mac holds every product and partial sum exactly, runs on a
vendor GEMM: ``torch._int_mm``, or torch's float matmul on a CPU without
oneDNN. These tests hold both engines to the kernel word for word (the CPU
cases run each engine, forced):

- **Layers**: a ``QLinear`` and a ``QMatmul`` built from ``GemmFormats``,
  forward and backward, over operand formats (signed, unsigned, symmetric,
  mixed steps), split and fused macs, wide and narrow results, ragged and
  batched shapes, and NaN operands, on the CPU and CUDA. Each case checks
  that the route ran, so a route that never fires cannot pass.
- **The bound on the sum's length**: the longest exact sum takes the route
  and one step longer takes the kernel; a route forced past the bound gives
  other words (the negative control).
- **What keeps the kernel**: stochastic rounding, wrap-around, an operand
  quantizer that is not a fixed-point ``Quant``, wider codes, the carrier
  path, float16 and bfloat16 operands, GEMMs below the size threshold.
- **The float engine**: binary64 past float32's exact range, and no -0.0.
- **``torch.compile``**: a compiled layer on the route gives the eager
  kernel's words.
"""

import pytest
import torch

import mptorch.quant._exact as exact
from mptorch import FixedPoint, Overflow, RoundMode
from mptorch.quant import (
    FusedMac,
    GemmFormats,
    QLinear,
    QMatmul,
    Quant,
    ScaledQuant,
    SplitMac,
)
from mptorch.quant.mac import spec_for_mac

from .markers import requires_cuda, requires_mps

# "cpu-int" and "cpu-float" force the CPU's engine (`_exact._CPU_ENGINE`).
DEVICES = [
    "cpu-int",
    "cpu-float",
    pytest.param("cuda", marks=requires_cuda),
    pytest.param("mps", marks=requires_mps),
]

Q07 = FixedPoint(8, 7)
NARROW = SplitMac(FixedPoint(16, 14), FixedPoint(25, 14))  # every sum of up to 1,023 Q0.7 products
WIDE = SplitMac(FixedPoint(16, 14), FixedPoint(32, 14))  # a float64 result


@pytest.fixture
def calls(monkeypatch):
    """Every GEMM size takes the route, and each engine call is recorded (the
    engine, its left operand's dtype and shape)."""
    monkeypatch.setattr(exact, "_MIN_STEPS", dict.fromkeys(exact._MIN_STEPS, 0))
    seen: list[tuple[str, torch.dtype, tuple[int, ...]]] = []
    for name in ("_int_mm", "_float_mm"):
        real = getattr(exact, name)

        def counted(a, b, real=real, name=name):
            seen.append((name, a.dtype, tuple(a.shape)))
            return real(a, b)

        monkeypatch.setattr(exact, name, counted)
    return seen


def _on(monkeypatch, device: str) -> str:
    """The torch device of a ``DEVICES`` entry, forcing the CPU's engine for
    the two CPU entries."""
    if device.startswith("cpu-"):
        monkeypatch.setattr(exact, "_CPU_ENGINE", device.removeprefix("cpu-"))
        return "cpu"
    return device


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int64)


def _same(u: torch.Tensor, v: torch.Tensor) -> bool:
    return u.dtype == v.dtype and torch.equal(_words(u), _words(v))


def _both(monkeypatch, calls, run):
    """``run()`` on the route and on the kernel: the two results and how
    many ``_int_mm`` calls each made."""
    out = {}
    for on in (True, False):
        monkeypatch.setattr(exact, "_ENABLED", on)
        calls.clear()
        out[on] = (run(), len(calls))
    return out[True], out[False]


def _linear_step(formats: GemmFormats, x: torch.Tensor, n_out: int, device: str):
    """A ``QLinear``'s output, input gradient and weight gradient."""
    gen = torch.Generator().manual_seed(1)
    layer = QLinear(x.shape[-1], n_out, bias=True, formats=formats.affine())
    with torch.no_grad():
        layer.weight.copy_(torch.rand(layer.weight.shape, generator=gen) * 2 - 1)
    layer = layer.to(device, x.dtype)

    def run():
        layer.zero_grad(set_to_none=True)
        xg = x.detach().clone().requires_grad_()
        y = layer(xg)
        y.backward(torch.full_like(y, 0.25))
        return y.detach(), xg.grad, layer.weight.grad

    return run


def _inputs(shape, device, dtype=torch.float32, lo=-0.5, hi=0.5):
    gen = torch.Generator().manual_seed(0)
    return (torch.rand(shape, generator=gen, dtype=torch.float64) * (hi - lo) + lo).to(
        device, dtype
    )


# Each case: the GemmFormats, the input's range, and what the route covers.
CASES = {
    "Q0.7 into (25, 14)": (GemmFormats.uniform(NARROW, Quant(Q07)), (-0.5, 0.5)),
    "a wide forward, rounded once": (
        GemmFormats(
            input_quant=Quant(Q07),
            weight_quant=Quant(Q07),
            igrad_quant=Quant(Q07),
            wgrad_quant=Quant(Q07),
            fwd_mac=WIDE,
            igrad_mac=NARROW,
            wgrad_mac=NARROW,
            output_quant=FixedPoint(16, 8),
        ),
        (-0.5, 0.5),
    ),
    "mixed steps": (
        GemmFormats(
            input_quant=Quant(FixedPoint(8, 4)),
            weight_quant=Quant(Q07),
            igrad_quant=Quant(FixedPoint(8, 4)),
            wgrad_quant=Quant(FixedPoint(8, 4)),
            fwd_mac=SplitMac(FixedPoint(16, 11), FixedPoint(25, 11)),
            igrad_mac=SplitMac(FixedPoint(16, 11), FixedPoint(25, 11)),
            wgrad_mac=SplitMac(FixedPoint(16, 8), FixedPoint(25, 8)),
        ),
        (-6.0, 6.0),
    ),
    "an unsigned input": (
        GemmFormats(
            input_quant=Quant(FixedPoint(7, 7, is_signed=False)),
            weight_quant=Quant(Q07),
            igrad_quant=Quant(Q07),
            wgrad_quant=Quant(Q07),
            fwd_mac=NARROW,
            igrad_mac=NARROW,
            wgrad_mac=SplitMac(FixedPoint(16, 14), FixedPoint(25, 14)),
        ),
        (0.0, 1.0),
    ),
    "symmetric operands, a fused mac": (
        GemmFormats.uniform(FusedMac(FixedPoint(25, 14)), Quant(FixedPoint(8, 7, symmetric=True))),
        (-1.0, 1.0),
    ),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("shape", [(5, 64), (17, 100), (40, 257), (2, 9, 64)])
def test_a_linear_layer_on_the_route_is_the_kernel(monkeypatch, calls, device, case, shape):
    device = _on(monkeypatch, device)
    if device == "mps" and case == "a wide forward, rounded once":
        pytest.skip("MPS has no float64, so a wide result is refused there")
    formats, (lo, hi) = CASES[case]
    run = _linear_step(formats, _inputs(shape, device, lo=lo, hi=hi), 33, device)
    (on, n_on), (off, n_off) = _both(monkeypatch, calls, run)
    assert n_on == 3 and n_off == 0
    for u, v in zip(on, off, strict=True):
        assert _same(u, v)


@pytest.mark.parametrize("device", DEVICES)
def test_nan_operands_reach_the_same_outputs(monkeypatch, calls, device):
    device = _on(monkeypatch, device)
    x = _inputs((24, 96), device)
    x[3, 5] = float("nan")
    x[7, 0] = float("nan")
    run = _linear_step(GemmFormats.uniform(NARROW, Quant(Q07)), x, 40, device)
    (on, n_on), (off, _) = _both(monkeypatch, calls, run)
    assert n_on == 3
    assert on[0].isnan().any()
    for u, v in zip(on, off, strict=True):
        assert _same(u, v)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "shapes",
    [
        ((6, 70), (70, 20)),
        ((3, 6, 70), (3, 70, 20)),
        ((3, 6, 70), (70, 20)),
        ((6, 70), (2, 70, 20)),
    ],
)
def test_a_matmul_on_the_route_is_the_kernel(monkeypatch, calls, device, shapes):
    device = _on(monkeypatch, device)
    formats = GemmFormats.uniform(NARROW, Quant(Q07)).matmul()
    a0 = _inputs(shapes[0], device)
    b0 = _inputs(shapes[1], device, lo=-1, hi=1)

    def run():
        a, b = a0.clone().requires_grad_(), b0.clone().requires_grad_()
        y = QMatmul(formats)(a, b)
        y.backward(torch.full_like(y, 0.25))
        return y.detach(), a.grad, b.grad

    (on, n_on), (off, n_off) = _both(monkeypatch, calls, run)
    assert n_on >= 3 and n_off == 0
    for u, v in zip(on, off, strict=True):
        assert _same(u, v)


def test_a_float64_layer_takes_the_route_in_binary64(monkeypatch, calls):
    # binary64 holds an exact sum of up to 53 bits in the carrier, so only an
    # accumulate format past that span runs the integer accumulator
    mac = SplitMac(FixedPoint(16, 14), FixedPoint(54, 14))
    run = _linear_step(
        GemmFormats.uniform(mac, Quant(Q07)), _inputs((20, 80), "cpu", torch.float64), 30, "cpu"
    )
    (on, n_on), (off, _) = _both(monkeypatch, calls, run)
    assert n_on == 3 and on[0].dtype == torch.float64
    for u, v in zip(on, off, strict=True):
        assert _same(u, v)


def _route():
    """The route of a Q0.7 x Q0.7 pass into (25, 14)."""
    return exact.exact_route(NARROW, Quant(Q07), Quant(Q07))


def test_the_bound_on_the_sum_length():
    # Q0.7 codes give products in [-16256, 16384]; (25, 14) holds sums up to
    # 2^24 - 1 on the products' grid, so 1,023 products of 16384
    assert _route().k_max == (2**24 - 1) // 16384 == 1023
    assert _route().dtypes == frozenset({torch.float32})


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("K, routed", [(1023, True), (1024, False)])
def test_the_longest_exact_sum_takes_the_route(monkeypatch, calls, device, K, routed):
    device = _on(monkeypatch, device)
    spec = spec_for_mac(NARROW)
    a = torch.full((40, K), -1.0, device=device)  # the largest products, every one
    b = torch.full((K, 24), -1.0, device=device)

    def run():
        return exact.run_exact(_route(), spec, a, b, False, False)

    (on, n_on), (off, _) = _both(monkeypatch, calls, run)
    assert (n_on > 0) is routed
    assert _same(on, off)


@pytest.mark.parametrize("device", DEVICES)
def test_a_route_past_the_bound_gives_other_words(monkeypatch, calls, device):
    device = _on(monkeypatch, device)
    # The negative control: 1,024 products of 1.0 saturate (25, 14) at
    # 1024 - 2^-14 in the kernel, while the integer sum is 1024 exactly.
    spec = spec_for_mac(NARROW)
    a = torch.full((40, 1024), -1.0, device=device)
    b = torch.full((1024, 24), -1.0, device=device)
    forced = _route()._replace(k_max=10**6)
    routed = exact.run_exact(forced, spec, a, b, False, False)
    kernel = exact._run_gemm(spec, a, b, False, False)
    assert calls and not _same(routed, kernel)
    assert routed.max().item() == 1024.0 and kernel.max().item() == 1024.0 - 2.0**-14


@pytest.mark.parametrize(
    "mac, a, b",
    [
        (
            SplitMac(
                FixedPoint(16, 14, prng_bits=4),
                FixedPoint(25, 14, prng_bits=4),
                rounding=RoundMode.SR,
            ),
            Quant(Q07),
            Quant(Q07),
        ),
        (
            SplitMac(FixedPoint(16, 14), FixedPoint(25, 14), overflow=Overflow.WRAP),
            Quant(Q07),
            Quant(Q07),
        ),
        (NARROW, ScaledQuant(Q07), Quant(Q07)),
        (NARROW, None, Quant(Q07)),
        (NARROW, Quant(FixedPoint(9, 7)), Quant(Q07)),  # codes past int8
        (NARROW, Quant(FixedPoint(8, 7, is_signed=False)), Quant(Q07)),  # 0..255
        (NARROW, Quant(FixedPoint(8, 8)), Quant(Q07)),  # products finer than the formats' grid
        (
            SplitMac(FixedPoint(16, 14), FixedPoint(16, 14)),
            Quant(Q07),
            Quant(Q07),
        ),  # the carrier path
    ],
    ids=[
        "SR",
        "wrap",
        "a ScaledQuant",
        "no quantizer",
        "9-bit codes",
        "8-bit unsigned codes",
        "a coarser product grid",
        "the carrier path",
    ],
)
def test_what_keeps_the_kernel(mac, a, b):
    assert exact.exact_route(mac, a, b) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_narrower_operands_keep_the_kernel(monkeypatch, calls, dtype):
    kernel = []
    monkeypatch.setattr(exact, "_run_gemm", lambda *args: kernel.append(args) or args[1])
    a = _inputs((40, 64), "cpu", dtype)
    exact.run_exact(_route(), spec_for_mac(NARROW), a, a.mT, False, False)
    assert kernel and not calls


@pytest.mark.parametrize("engine", ["int", "float"])
def test_a_compiled_layer_on_the_route_gives_the_kernels_words(monkeypatch, engine):
    monkeypatch.setattr(exact, "_MIN_STEPS", dict.fromkeys(exact._MIN_STEPS, 0))
    monkeypatch.setattr(exact, "_CPU_ENGINE", engine)
    x = _inputs((32, 128), "cpu")
    layer = QLinear(128, 48, formats=GemmFormats.uniform(NARROW, Quant(Q07)).affine())
    with torch.no_grad():
        layer.weight.copy_(_inputs((48, 128), "cpu", lo=-1, hi=1))
    monkeypatch.setattr(exact, "_ENABLED", False)
    eager = layer(x)
    monkeypatch.setattr(exact, "_ENABLED", True)
    compiled = torch.compile(layer, fullgraph=True)(x)
    assert _same(compiled, eager)


@pytest.mark.parametrize("engine", ["int", "float"])
@pytest.mark.parametrize("routed", [False, True])
def test_small_gemms_keep_the_kernel(monkeypatch, engine, routed):
    # each CPU engine's threshold: a 64 x 64 x N call of exactly its mac
    # steps takes the route, and one column fewer keeps the kernel
    monkeypatch.setattr(exact, "_CPU_ENGINE", engine)
    seen = []
    for name in ("_int_mm", "_float_mm"):
        real = getattr(exact, name)
        monkeypatch.setattr(exact, name, lambda a, b, real=real: seen.append(1) or real(a, b))
    n = exact._MIN_STEPS[f"cpu-{engine}"] // (64 * 64) - (0 if routed else 1)
    a = Quant(Q07)(_inputs((64, 64), "cpu"))
    b = Quant(Q07)(_inputs((64, n), "cpu"))
    out = exact.run_exact(_route(), spec_for_mac(NARROW), a, b, False, False)
    assert bool(seen) is routed
    assert _same(out, exact._run_gemm(spec_for_mac(NARROW), a, b, False, False))


@pytest.mark.parametrize("device", DEVICES)
def test_a_sum_of_zero_products_is_the_unsigned_zero(monkeypatch, calls, device):
    # rows of zeros against negative weights: every product is -0.0 in
    # floating point, and the kernel's sum is +0.0
    device = _on(monkeypatch, device)
    spec = spec_for_mac(NARROW)
    a = torch.zeros(40, 64, device=device)
    a[1::2] = -0.5
    b = torch.full((64, 24), -0.5, device=device)
    (on, n_on), (off, _) = _both(
        monkeypatch, calls, lambda: exact.run_exact(_route(), spec, a, b, False, False)
    )
    assert n_on > 0 and _same(on, off)
    assert not torch.signbit(on[::2]).any()


def test_the_float_engine_sums_in_binary64_past_float32s_exact_range(monkeypatch, calls):
    # 2,000 Q0.7 products can sum past 2^24 steps, which float32 does not
    # hold exactly: the float engine takes binary64 for them
    monkeypatch.setattr(exact, "_CPU_ENGINE", "float")
    spec, q = spec_for_mac(WIDE), Quant(Q07)
    route = exact.exact_route(WIDE, q, q)
    assert route is not None
    a = q(_inputs((30, 2000), "cpu", lo=-1, hi=1))
    b = q(_inputs((2000, 20), "cpu", lo=-1, hi=1))
    (on, _), (off, _) = _both(
        monkeypatch, calls, lambda: exact.run_exact(route, spec, a, b, False, False)
    )
    monkeypatch.setattr(exact, "_ENABLED", True)
    calls.clear()
    exact.run_exact(route, spec, a, b, False, False)
    assert calls == [("_float_mm", torch.float64, (30, 2000))]
    assert on.dtype == torch.float64 and _same(on, off)


def test_the_float_engine_needs_full_float32_precision(monkeypatch, calls):
    # a lowered float32 matmul precision may multiply in fewer bits: the
    # float engine then takes binary64
    monkeypatch.setattr(exact, "_CPU_ENGINE", "float")
    a = Quant(Q07)(_inputs((30, 64), "cpu"))
    b = Quant(Q07)(_inputs((64, 20), "cpu"))
    spec = spec_for_mac(NARROW)
    for precision, dtype in (("highest", torch.float32), ("medium", torch.float64)):
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(precision)
        try:
            calls.clear()
            out = exact.run_exact(_route(), spec, a, b, False, False)
        finally:
            torch.set_float32_matmul_precision(previous)
        assert [c[1] for c in calls] == [dtype]
        assert _same(out, exact._run_gemm(spec, a, b, False, False))
