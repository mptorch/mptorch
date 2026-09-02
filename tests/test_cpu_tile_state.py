"""The CPU GEMM's per-output-element tile state -- finding C3's gate.

The reduction state for a 32x32 output tile used to be a
``std::vector<Accumulator>`` built per tile, each element carrying its own
copy of the Mac policy. It is now a pair of dense buffers owned by the worker
task and *reused across every tile that task takes*, with the format policy
held once for the whole call. Three things could break that the old shape
made impossible, and each has a test here:

* stale state leaking from one tile into the next, since the buffers are no
  longer freshly constructed per tile;
* a ragged edge tile (``ti < 32`` or ``tj < 32``) indexing the buffers as if
  it were full;
* the mixed-format path binding the wrong palette slot to an element, since
  the slot is now a pointer resolved per tile rather than a copy living in
  the element's own accumulator.

The checks are bit-exact -- raw words, not tolerances -- because the change
is supposed to move nothing at all.
"""

from typing import TypedDict

import pytest
import torch

from mptorch.number import RoundMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_mixed,
    superfp_matmul,
    superfp_matmul_fma,
)

# Shapes chosen to straddle the 32x32 tiling: a full tile, ragged in one
# dimension, ragged in both, and a single leftover row/column.
SHAPES = [(64, 64, 64), (70, 33, 51), (33, 33, 33), (65, 17, 96), (96, 5, 1)]
DETERMINISTIC = [m for m in RoundMode if m is not RoundMode.SR]


class _BKParams(TypedDict):
    mul_K: int
    mul_P: int
    acc_K: int
    acc_P: int


class _SFPParams(TypedDict):
    mul_man_bits: int
    mul_exp_bits: int
    mul_normal_binades: int
    mul_bias: int
    acc_man_bits: int
    acc_exp_bits: int
    acc_normal_binades: int
    acc_bias: int


BK: _BKParams = dict(mul_K=8, mul_P=4, acc_K=8, acc_P=4)
SFP: _SFPParams = dict(
    mul_man_bits=3,
    mul_exp_bits=4,
    mul_normal_binades=1,
    mul_bias=7,
    acc_man_bits=3,
    acc_exp_bits=4,
    acc_normal_binades=1,
    acc_bias=7,
)


def words(t):
    """Compare as raw words so signed zeros and NaN payloads count too."""
    return t.detach().cpu().contiguous().view(torch.int32)


def same(a, b, ctx):
    assert torch.equal(words(a), words(b)), f"outputs differ: {ctx}"


@pytest.fixture
def restore_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def ops(a, b, mode):
    """Every CPU GEMM shape, at one format per family.

    Each op re-seeds first: under SR the per-call seed is drawn from the
    global generator, so two calls that are meant to be compared have to
    start from the same generator state. Harmless for the other six modes,
    which never draw."""
    pb = 4 if mode is RoundMode.SR else 0

    def seeded(fn):
        def run():
            torch.manual_seed(4321)
            return fn()

        return run

    return {
        "binaryK split": seeded(
            lambda: binaryK_matmul(
                a, b, **BK, rounding_mode=mode, mul_prng_bits=pb, acc_prng_bits=pb
            )
        ),
        "binaryK fma": seeded(
            lambda: binaryK_matmul_fma(a, b, fma_K=8, fma_P=4, rounding_mode=mode, fma_prng_bits=pb)
        ),
        "superfp split": seeded(
            lambda: superfp_matmul(
                a, b, **SFP, rounding_mode=mode, mul_prng_bits=pb, acc_prng_bits=pb
            )
        ),
        "superfp fma": seeded(
            lambda: superfp_matmul_fma(
                a,
                b,
                fma_man_bits=3,
                fma_exp_bits=4,
                fma_normal_binades=1,
                fma_bias=7,
                rounding_mode=mode,
                fma_prng_bits=pb,
            )
        ),
    }


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", list(RoundMode))
def test_output_is_independent_of_thread_count(shape, mode, restore_threads):
    """The tile buffers are per worker task, so how tiles are shared out must
    not change a single bit -- SR included, since its stream is keyed by the
    output element's global index rather than by tiling."""
    m, k, n = shape
    torch.manual_seed(1234)
    a, b = torch.randn(m, k), torch.randn(k, n)

    reference = None
    for nthreads in (1, 2, 4, 8):
        torch.set_num_threads(nthreads)
        got = {name: fn() for name, fn in ops(a, b, mode).items()}
        if reference is None:
            reference = got
        else:
            for name in got:
                same(reference[name], got[name], f"{name} {shape} {mode.name} @{nthreads} threads")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_row_slice_matches_the_full_gemm(shape, mode):
    """One row computed alone lands in a 1-row tile grid instead of somewhere
    inside a 32-row one. The K-reduction for an element does not depend on
    tiling, so every word must match -- state left over from a previous tile
    would not.

    Deterministic modes only: SR seeds each element from its index in the
    *whole* output, so a 1-row GEMM is a different (and correctly different)
    stream."""
    m, k, n = shape
    torch.manual_seed(1234)
    a, b = torch.randn(m, k), torch.randn(k, n)

    full = {name: fn() for name, fn in ops(a, b, mode).items()}
    for i in (0, m // 2, m - 1):
        sliced = {name: fn() for name, fn in ops(a[i : i + 1], b, mode).items()}
        for name in full:
            same(full[name][i : i + 1], sliced[name], f"{name} {shape} {mode.name} row {i}")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_mixed_palette_binds_the_right_slot_per_row(shape, mode):
    """Each row of a per-row mixed-format call must equal that row of the
    single-format call in the slot's own format. The palette slot is now a
    pointer resolved per tile rather than a copy inside the element's
    accumulator, so a mis-resolved slot is exactly what this catches."""
    m, k, n = shape
    torch.manual_seed(1234)
    a, b = torch.randn(m, k), torch.randn(k, n)

    palette = [(8, 4), (6, 3), (5, 2)]
    idx = (torch.arange(m) % len(palette)).to(torch.int32).view(m, 1)
    mixed = binaryK_matmul_mixed(
        a,
        b,
        idx,
        mul_K=[p[0] for p in palette],
        mul_P=[p[1] for p in palette],
        rounding_mode=mode,
    )

    for slot, (kk, pp) in enumerate(palette):
        single = binaryK_matmul(a, b, mul_K=kk, mul_P=pp, acc_K=kk, acc_P=pp, rounding_mode=mode)
        rows = (idx.view(-1) == slot).nonzero().view(-1)
        same(mixed[rows], single[rows], f"slot {slot} K={kk} P={pp} {shape} {mode.name}")


@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_a_tall_output_reuses_its_buffers_cleanly(mode, restore_threads):
    """Many tiles per worker task is the case where buffer reuse actually
    happens: one thread, an output several tile-rows tall, so every tile after
    the first runs on state the previous tile left behind."""
    torch.set_num_threads(1)
    m, k, n = 160, 8, 160  # 5x5 tiles, all taken by the one worker
    torch.manual_seed(1234)
    a, b = torch.randn(m, k), torch.randn(k, n)

    full = binaryK_matmul(a, b, **BK, rounding_mode=mode)
    for i in (0, 33, 64, m - 1):  # first tile, second tile-row, third, last
        same(
            full[i : i + 1],
            binaryK_matmul(a[i : i + 1], b, **BK, rounding_mode=mode),
            f"row {i} {mode.name}",
        )
