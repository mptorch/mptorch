"""Properties of the random stream behind ``RoundMode.SR``.

The elementwise SR path generates its draws inside the kernel: one seed is
drawn from torch's generator per launch, and each element's random word is
Philox of that seed and the element's own linear index, rather than a tensor
of draws filled by ``randint_like`` per call. Four consecutive binary32
elements share one Philox block and take one word each
(mptorch/csrc/common/philox.h), a mapping that can be wrong in ways the other
SR tests (grid identity, bounding and unbiasedness in
test_binaryk_quantize.py and test_superfp_quantize.py) would not notice: a
bias tied to position mod 4, a correlation between neighbours, or a
dependence on how at::parallel_for happens to cut the work. These cover that.

A float64 element is rounded in binary64 and takes two words, so two elements
share a block: every property is checked for it too, position mod 2
included, with 40 random bits, more than binary32 has below any mantissa.
"""

import math

import pytest
import torch

from mptorch import RoundMode
from mptorch.quant import binaryK_quantize, superfp_quantize
from tests.markers import available_devices

# e4m3 near 1.0: the neighbouring representable values are 1.0 and 1.125, so
# an input halfway between them must round up half the time.
LO, STEP = 1.0, 0.125
HALFWAY = LO + 0.5 * STEP


DTYPES = [torch.float32, torch.float64]
# The random bits each carrier's SR is exercised with: 20 of binary32's 23 and
# 40 of binary64's 52, below a 3-bit mantissa.
PRNG_BITS = {torch.float32: 20, torch.float64: 40}


def sr_up(n, device, dtype=torch.float32):
    """Round-up indicator for ``n`` copies of a value exactly halfway up a gap."""
    x = torch.full((n,), HALFWAY, dtype=dtype, device=device)
    q = binaryK_quantize(
        x, 8, 4, bias=7, prng_bits=PRNG_BITS[dtype], rounding_mode=RoundMode.SR, is_signed=True
    )
    return (q > LO).float()


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "dtype,modulus",
    [
        (torch.float32, 4),
        (torch.float32, 8),
        (torch.float64, 2),
        (torch.float64, 4),
        (torch.float64, 8),
    ],
)
def test_sr_has_no_positional_bias(device, dtype, modulus):
    """Every position within a Philox block must round up equally often."""
    n = 1 << 20
    up = sr_up(n, device, dtype)
    sigma = math.sqrt(0.25 / (n / modulus))
    for k in range(modulus):
        rate = up.view(-1, modulus)[:, k].mean().item()
        assert abs(rate - 0.5) < 5 * sigma, (
            f"position {k} mod {modulus} rounds up {rate:.6f} of the time, "
            f"expected 0.5 +/- {5 * sigma:.6f}"
        )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sr_neighbours_are_uncorrelated(device, dtype):
    """The round-up indicator's autocorrelation at lags 1 to 5 must be at noise
    level; elements sharing a block must not see related words."""
    n = 1 << 20
    up = sr_up(n, device, dtype)
    centered = up - up.mean()
    denom = (centered * centered).sum().item()
    assert denom > 0.0, "stochastic rounding produced a constant result"
    for lag in (1, 2, 3, 4, 5):
        r = (centered[:-lag] * centered[lag:]).sum().item() / denom
        assert abs(r) < 5 / math.sqrt(n), f"lag {lag} autocorrelation {r:.6f}"


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sr_successive_calls_are_independent(device, dtype):
    """Two calls on the same input must draw different, uncorrelated values."""
    n = 1 << 16
    first, second = sr_up(n, device, dtype), sr_up(n, device, dtype)
    assert not torch.equal(first, second)
    agreement = (first == second).float().mean().item()
    assert abs(agreement - 0.5) < 5 * math.sqrt(0.25 / n), (
        f"the two calls agree on {agreement:.5f} of elements"
    )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "quantize",
    [
        lambda x: binaryK_quantize(
            x, 8, 4, bias=7, prng_bits=20, rounding_mode=RoundMode.SR, is_signed=True
        ),
        lambda x: superfp_quantize(
            x,
            2,
            5,
            normal_binades=1,
            bias=15,
            prng_bits=21,
            rounding_mode=RoundMode.SR,
            is_signed=True,
        ),
    ],
    ids=["binaryK", "superfp"],
)
def test_sr_is_reproducible_under_manual_seed(device, dtype, quantize):
    """The same seed must reproduce the same draws, so the per-launch seed
    comes from torch's generator and nothing else."""
    x = torch.randn(50_001, device=device, dtype=dtype) * 0.5
    outputs = []
    for _ in range(2):
        torch.manual_seed(9871)
        torch.cuda.manual_seed_all(9871)
        outputs.append(quantize(x))
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize("threads", [2, 8])
@pytest.mark.parametrize("dtype", DTYPES)
def test_sr_is_independent_of_cpu_thread_count(threads, dtype):
    """A CPU result must not depend on where at::parallel_for cuts the work."""
    x = torch.randn(200_003, dtype=dtype) * 0.5
    restore = torch.get_num_threads()
    try:
        results = []
        for count in (1, threads):
            torch.set_num_threads(count)
            torch.manual_seed(4242)
            results.append(
                binaryK_quantize(
                    x,
                    8,
                    4,
                    bias=7,
                    prng_bits=PRNG_BITS[dtype],
                    rounding_mode=RoundMode.SR,
                    is_signed=True,
                )
            )
    finally:
        torch.set_num_threads(restore)
    assert torch.equal(results[0], results[1])


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 33, 1023])
def test_sr_tail_elements_stay_in_bounds(device, dtype, n):
    """The elements past the last full vector take a different code path."""
    x = torch.randn(n, device=device, dtype=dtype) * 0.5
    rd = binaryK_quantize(x, 8, 4, bias=7, is_signed=True, rounding_mode=RoundMode.RD)
    ru = binaryK_quantize(x, 8, 4, bias=7, is_signed=True, rounding_mode=RoundMode.RU)
    sr = binaryK_quantize(
        x, 8, 4, bias=7, is_signed=True, prng_bits=PRNG_BITS[dtype], rounding_mode=RoundMode.SR
    )
    assert bool(((sr == rd) | (sr == ru)).all())
