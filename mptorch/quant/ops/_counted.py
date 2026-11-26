"""The counted twins of the single-format GEMM ops.

Each single-format GEMM op, NAIVE or ``*_accumulated``, has a counted twin
(``csrc/common/gemm_counted.h``) on the ``*_accumulated`` schema, which takes
every accumulate algorithm, NAIVE included, and returns the result with an
int64 ``[3, 6]`` tensor of its casts' numeric events: the roles ``mul``,
``acc`` and ``outer`` by :data:`mptorch.quant.events.EVENTS`. Its result is
its twin's word for word, under SR too: the kernels only read the casts.

A NAIVE spec's ``args`` are its op's, which the ``*_accumulated`` op extends
by a block size and an outer format, so its counted call appends the tail
that says "no outer format". An accumulated spec's ``args`` are the counted
op's already. Each single-format conv op has a counted twin too, on its own
schema, which a spec reaches as it reaches the conv op (`_conv._conv_ops`).
The fixed-point GEMM and conv ops have counted twins on their own schemas.
The palette, cross-family, odd and block GEMMs have no counted twin.

The counted kernels are built with ``MPTORCH_COUNTING=1``; without it the
counted ops raise, naming the flag.
"""

from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from ._spec import _GemmSpec

# The *_accumulated ops' tail for NAIVE: no block size, no outer format.
_NO_OUTER_BINARYK = (0, False, 0, 0, 0, True, 0, 0, 0)
_NO_OUTER_SUPERFP = (0, False, 0, 0, 0, 0, True, 0, 0)

_TWINS: dict[Any, tuple[Callable[..., Any], tuple[Any, ...]]] = {}
_CONV_TWINS: dict[Any, tuple[Callable[..., Any], tuple[Any, ...]]] = {}


def _twins() -> dict[Any, tuple[Callable[..., Any], tuple[Any, ...]]]:
    """Each covered op's counted twin and the tail its arguments take."""
    if not _TWINS:
        ops = torch.ops.mptorch
        for family, tail in (("binaryK", _NO_OUTER_BINARYK), ("superfp", _NO_OUTER_SUPERFP)):
            for mac in ("", "_fma"):
                counted = getattr(ops, f"custom_matmul_{family}{mac}_counted").default
                _TWINS[getattr(ops, f"custom_matmul_{family}{mac}").default] = (counted, tail)
                _TWINS[getattr(ops, f"custom_matmul_{family}{mac}_accumulated").default] = (
                    counted,
                    (),
                )
                conv = getattr(ops, f"custom_conv_{family}{mac}_counted").default
                _CONV_TWINS[getattr(ops, f"custom_matmul_{family}{mac}").default] = (conv, tail)
                _CONV_TWINS[getattr(ops, f"custom_matmul_{family}{mac}_accumulated").default] = (
                    conv,
                    (),
                )
        # fixed point, NAIVE only: the twins take the fixed-point ops' own arguments
        for mac in ("", "_fma"):
            gemm = getattr(ops, f"custom_matmul_fixedpoint{mac}").default
            _TWINS[gemm] = (getattr(ops, f"custom_matmul_fixedpoint{mac}_counted").default, ())
            _CONV_TWINS[gemm] = (getattr(ops, f"custom_conv_fixedpoint{mac}_counted").default, ())
    return _TWINS


def _counted_twin(spec: _GemmSpec) -> tuple[Callable[..., Any], tuple[Any, ...]] | None:
    """The counted op and its arguments after the operands and the transpose
    flags for a resolved GEMM, or ``None`` when its op has no counted twin."""
    twin = _twins().get(spec.op)
    if twin is None:
        return None
    op, tail = twin
    return op, spec.args + tail


def _counted_conv_twin(spec: _GemmSpec) -> tuple[Callable[..., Any], tuple[Any, ...]] | None:
    """The counted conv op of a resolved GEMM and its arguments after the
    pass's geometry, or ``None`` when its conv op has no counted twin."""
    _twins()
    twin = _CONV_TWINS.get(spec.op)
    if twin is None:
        return None
    op, tail = twin
    return op, spec.args + tail


def _counted_parts(
    spec: _GemmSpec,
) -> (
    tuple[tuple[Callable[..., Any], tuple[Any, ...]], tuple[Callable[..., Any], tuple[Any, ...]]]
    | None
):
    """A resolved GEMM's counted op and its conv op, each with the arguments
    it takes past the spec's, or ``None`` when its op has no counted twin."""
    gemm = _twins().get(spec.op)
    if gemm is None:
        return None
    return gemm, _CONV_TWINS[spec.op]


@cache
def _counting_built() -> bool:
    """Whether this build has the counted kernels (``MPTORCH_COUNTING=1``):
    without them a counted op raises, naming the flag."""
    x = torch.ones(1, 1)
    op, tail = _twins()[torch.ops.mptorch.custom_matmul_binaryK_fma.default]
    fused = (True, 8, 4, 7, True, 0, 0, 0, 0, 0)  # fma_quant ... fma_prng_bits, NAIVE, RNE
    try:
        op(x, x, False, False, *fused, *tail)
    except RuntimeError as e:
        if "MPTORCH_COUNTING" in str(e):
            return False
        raise
    return True
