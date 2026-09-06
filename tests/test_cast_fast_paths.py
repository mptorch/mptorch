"""
Tests that the host and the device agree bit for bit on every deterministic
cast, which since finding C5 in dev/gemm_roadmap.md is a claim about two
different implementations rather than one.

The float-arithmetic cast fast paths (G3, G7, G8) used to be ``#if
defined(__CUDA_ARCH__)``: the Veltkamp split they are built on is exact only
if the compiler does not contract ``t - (t - x)`` into an FMA, which the _rn
intrinsics guarantee and a host compiler only promises under
``-ffp-contract=off``. setup.py now asks for that promise and defines
``MPTORCH_FAST_CAST`` when it is given, so a host build takes the same paths
-- except ``cast_superfp_rne_fast``, which is exact on the host and slower
than the integer path it would replace, and so stays off there.

The upshot is that for one format and one rounding mode the two backends can
now be running different code: binaryK RNE is the float path on both,
superfp RNE is the float path on the device and the bitwise one on the host,
and the two SR paths differ from neither. All of them are meant to be the
same function of the input, and this file is what says so. The exhaustive
proof is elsewhere -- dev/benchmarks/gemm_cast_{float,superfp,sr}_arith.cu
compare every one of the 2^32 float inputs, and build for the host as well
since C5 -- but that runs in minutes, not in a test suite, and it compares a
path against its own build. This compares the two builds against each other.

Elementwise quantization is the right shape for that: it applies the cast and
nothing else, so a difference is the cast's. A GEMM is not -- its accumulation
order is the backend's tiling, so its results are not expected to match across
devices, and tests/test_qmatmul.py checks each backend against a reference
instead.
"""

import pytest
import torch

from mptorch.number import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import binaryK_quantize, superfp_quantize
from tests.markers import requires_cuda

# Every rounding mode but SR, which draws its own randomness and is compared
# against a reference per backend in tests/test_sr_rng.py.
DETERMINISTIC = [rm for rm in RoundMode if rm is not RoundMode.SR]

# 100_003 is prime, so no backend's vector width divides it and the scalar
# tail runs on both -- the same argument tests/test_quantize_dispatch.py makes.
SIZE = 100_003

# Widths that straddle the fast paths' gate: (8, 4) and (11, 5) are admitted,
# (6, 3) has a narrow exponent, and one of them is wide enough in the
# significand that make_binaryK_params' split-product bound refuses it -- so
# the parametrization covers both sides of the gate without naming which is
# which, which is make_binaryK_params' business and not this file's.
BINARYK_FORMATS = [(8, 4), (11, 5), (6, 3), (24, 8)]
SUPERFP_FORMATS = [(3, 4, 1, 7), (2, 3, 2, 3), (5, 5, 1, 15), (3, 4, 8, 7)]


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality -- so -0.0 and 0.0 are different -- except for NaN payloads.

    A NaN only has to still be a NaN. The two backends do not agree on more
    than that under directed rounding: RU, RD and RZ return the input's exact
    NaN on the host and a canonical ``0x7fffffff`` on the device, for every
    binaryK format (superfp agrees on both). That predates C5 and is untouched
    by it -- those modes have no fast path, and the pre-C5 build diverges the
    same way -- so it is stated here rather than quietly excluded. RNE, RNA and
    RO do preserve the payload on both, and this comparison would catch it if
    one of them stopped.
    """
    nan = a.isnan()
    if not torch.equal(nan, b.isnan()):
        return False
    return torch.equal(a.view(torch.int32)[~nan], b.view(torch.int32)[~nan])


@pytest.fixture(scope="module")
def x() -> torch.Tensor:
    """Inputs spanning the arms: subnormal, normal, overflowing, and 0/inf/NaN."""
    torch.manual_seed(1234)
    body = torch.randn(SIZE - 8) * torch.exp2(torch.randint(-24, 24, (SIZE - 8,)).float())
    edges = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 65504.0])
    return torch.cat([body, edges])


@requires_cuda
@pytest.mark.parametrize("rounding_mode", DETERMINISTIC)
@pytest.mark.parametrize("saturation_mode", list(SaturationMode))
@pytest.mark.parametrize("subnormals_mode", list(SubnormalsMode))
@pytest.mark.parametrize("K,P", BINARYK_FORMATS)
def test_binaryK_quantize_host_matches_device(
    x, K, P, rounding_mode, saturation_mode, subnormals_mode
):
    kw = dict(
        K=K,
        P=P,
        rounding_mode=rounding_mode,
        saturation_mode=saturation_mode,
        subnormals_mode=subnormals_mode,
    )
    assert _same_bits(binaryK_quantize(x, **kw), binaryK_quantize(x.cuda(), **kw).cpu())


@requires_cuda
@pytest.mark.parametrize("rounding_mode", DETERMINISTIC)
@pytest.mark.parametrize("saturation_mode", list(SaturationMode))
@pytest.mark.parametrize("man_bits,exp_bits,normal_binades,bias", SUPERFP_FORMATS)
def test_superfp_quantize_host_matches_device(
    x, man_bits, exp_bits, normal_binades, bias, rounding_mode, saturation_mode
):
    kw = dict(
        man_bits=man_bits,
        exp_bits=exp_bits,
        normal_binades=normal_binades,
        bias=bias,
        rounding_mode=rounding_mode,
        saturation_mode=saturation_mode,
    )
    assert _same_bits(superfp_quantize(x, **kw), superfp_quantize(x.cuda(), **kw).cpu())
