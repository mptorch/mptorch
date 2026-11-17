"""What a tensor narrower than its carrier can store: the storage checks' rule.

A float16 or bfloat16 tensor is rounded in binary32, its default carrier,
and the result is converted back when it is written: `SIMDTraits`' store in
the elementwise quantizers, `store_elem` in the GEMM, a `static_cast` on the
host, round to nearest on both backends. Under `carrier=torch.float64`
float32 joins them: the operands are widened, rounded in binary64, and the
result narrowed by `mptorch.quant.ops._narrowed`, once, to nearest. Either
conversion is a second rounding. A format whose results are not all values
of the dtype is quantized partially, as one that outruns its carrier is,
with one difference worth saying out loud: a result above the dtype's range
is stored as an infinity *whatever the format's saturation mode says*, so a
`SAT_FINITE` format overflows too.

Which results a call can produce depends on the op, and the two answers
differ enough to need two rules:

  a GEMM       rounds sums and products, which land anywhere, so every value
               of its last format is some result: the format's value set
               must be the dtype's. The carrier's "representable" question
               asked of a narrower float:

                                  float16     bfloat16    float32
                 precision        P <= 11     P <= 8      P <= 24      raises
                 largest value    < 2**16     < 2**128    < 2**128     warns
                 finest step      >= 2**-24   >= 2**-133  >= 2**-149   warns

  elementwise  rounds the tensor's own values, which are already the
               dtype's. A format at least as fine as the dtype somewhere
               hands those back unchanged there, so precision and a fine
               bottom cost nothing, and a result off the dtype's grid comes
               from three places only, at the format's edges: its largest
               value inside the dtype's range but not a value of it (an
               input above saturates onto it, in a saturating mode); the
               dtype's largest value not on the format's grid while the
               format reaches past it (an input there rounds up, out of the
               dtype's range); and `EXTENDED_NORMALS`' hole, an input the
               dtype has and the format does not, whose way up is not the
               dtype's. Each warns.

Both raise for the storage twin of the carrier's "no finite value" error, a
format whose largest value is below the dtype's smallest, of which nothing
is stored. And neither has the carrier's "faithful" floor: that exists
because the casts read an exponent field, where storing a value reads
nothing and rounds it correctly. Under binary32, for bfloat16 only the
GEMM's precision bound is new, since its range is binary32's to within its
precision, so a format binary32 holds with at most eight bits of precision
is inside it at both ends. Under binary64 the carrier passes ranges no
narrower dtype holds, so every row can speak. The rule does not otherwise
depend on the carrier, which is wider than the dtype either way and rounds
the dtype's own values exactly.

The dtype belongs to the call, not the format (a `BinaryK` does not know
what it will be stored in), so the family modules' `check_*_storage` run
per call, from `mptorch.quant.ops`, on the format whose values the result
holds: the elementwise quantizer's, and a GEMM's accumulate or fused format.
A GEMM's multiply format never reaches storage (its products are
intermediates in the carrier), and with the running sum left unquantized no
format's values are stored at all.
"""

from typing import NamedTuple

import torch

from ._carrier import _Extent
from .modes import SaturationMode


class _Storage(NamedTuple):
    """A dtype a rounded result is stored in, as `_storage_findings` reads it."""

    name: str
    man_bits: int  # stored significand bits
    top_exp: int  # the exponent of the largest finite value
    min_exp: int  # the exponent of the smallest positive value, a subnormal


_STORAGE: dict[torch.dtype, _Storage] = {
    torch.float16: _Storage("float16", 10, 15, -24),
    torch.bfloat16: _Storage("bfloat16", 7, 127, -133),
    torch.float32: _Storage("float32", 23, 127, -149),
}


def _normalized(M: int, k: int) -> tuple[int, int, int]:
    """``M * 2**k``, for ``M > 0``, as (odd significand, its exponent, the
    exponent of its leading bit).

    Strips the trailing zeros of ``M`` into the exponent so that two spellings
    of one value compare equal.
    """
    tz = (M & -M).bit_length() - 1
    M >>= tz
    return M, k + tz, k + tz + M.bit_length() - 1


def _in_storage(M: int, k: int, st: _Storage) -> bool:
    """Whether ``M * 2**k`` is a finite value of the dtype.

    True when the value is on the dtype's grid at that magnitude (the normals'
    ``2**(E - man_bits)``, or the subnormals' fixed step below them) and not
    above its largest value, which with at most ``man_bits + 1`` significant
    bits is a test on the leading bit's binade.
    """
    _, k, lead = _normalized(M, k)
    min_normal_exp = st.min_exp + st.man_bits
    return lead <= st.top_exp and k >= max(lead, min_normal_exp) - st.man_bits


def _on_format_grid(M: int, k: int, ext: _Extent) -> bool:
    """Whether ``M * 2**k`` sits on the format's grid, ignoring its top.

    The top is ignored on purpose: the question asked here is which
    neighbours an input at that magnitude has, which the grid's continuation
    above the largest value answers. Below the full-mantissa binades the
    answer depends on what the format keeps down there.
    """
    M, k, lead = _normalized(M, k)
    m = ext.man_bits
    if lead >= ext.normal_exp:
        return k >= lead - m
    if ext.below == "SUBNORMALS":
        return k >= ext.normal_exp - m
    if ext.below == "EXTENDED_NORMALS" and m and lead == ext.normal_exp - 1:
        return k >= lead - m and M != 1  # that binade's power of two is the hole
    if ext.below == "SUPERNORMALS":
        return M == 1 and lead >= ext.bottom_exp
    return False


def _storage_value_set_findings(ext: _Extent, st: _Storage) -> tuple[str | None, str | None]:
    """A GEMM's rule: the format's value set against the dtype's.

    Returns ``(error, warning)``. Precision is checked first, because the two
    range bounds are only exact once it holds. With at most the dtype's
    significand bits, a format value in a binade the dtype has normals for is
    one of them, and one below that is a multiple of ``2**step_exp`` under the
    dtype's smallest normal, which is a dtype subnormal as long as
    ``step_exp`` is not below the dtype's own step; and every format value
    under ``2**(top_exp + 1)`` is at most the dtype's largest finite value as
    long as ``top_exp`` is not above its own.
    """
    label, name = ext.label, st.name
    if ext.man_bits > st.man_bits:
        return (
            f"{label} has {ext.man_bits + 1} bits of precision, and a {name} result holds "
            f"{st.man_bits + 1}: a GEMM's result can be any of the format's values, and "
            f"storing it rounds the ones {name} lacks a second time, onto {name}'s grid"
        ), None
    if ext.top_exp > st.top_exp:
        return None, (
            f"{label} reaches 2^{ext.top_exp}, above {name}'s largest finite value "
            f"2^{st.top_exp + 1} - 2^{st.top_exp - st.man_bits}: a {name} result stores "
            f"the format's values above that as infinity, whatever the saturation mode "
            f"(needs bias >= {ext.top_field - st.top_exp})"
        )
    if ext.step_exp < st.min_exp:
        return None, (
            f"{label} spaces its values 2^{ext.step_exp} apart at the bottom, below "
            f"{name}'s 2^{st.min_exp}: a {name} result rounds the values down there "
            f"a second time, onto {name}'s coarser grid, when it is stored"
        )
    return None, None


def _storage_edge_findings(
    ext: _Extent, st: _Storage, saturation: SaturationMode
) -> tuple[str | None, str | None]:
    """The elementwise rule: which of the dtype's own values round off its grid.

    Returns ``(error, warning)``; only warnings arise here. An input already
    on the format's grid comes back unchanged, and one that is not sits where
    the format's grid is coarser than the dtype's, whose neighbours there are
    therefore values of the dtype too. What is left is the three edges the
    module docstring names, and nothing else.

    The second edge is reached only by saturating. A largest value that is not
    the dtype's has more significant bits than the dtype, so the format is
    finer there, and every input of the dtype above it is already on the
    format's continued grid: rounding leaves it where it is and `SAT_NONE`
    makes it an infinity, which is a value of the dtype.
    """
    label, name, m = ext.label, st.name, ext.man_bits
    top = ((1 << m) + ext.top_code, ext.top_exp - m)  # the largest finite value
    dtype_top = ((1 << (st.man_bits + 1)) - 1, st.top_exp - st.man_bits)
    if ext.top_exp != st.top_exp:
        # different binades decide it, and an exponent field of up to 30 bits
        # would make lining the two up as integers a very large shift
        above, below = ext.top_exp > st.top_exp, ext.top_exp < st.top_exp
    else:
        finer = max(m, st.man_bits)
        f, d = top[0] << (finer - m), dtype_top[0] << (finer - st.man_bits)
        above, below = f > d, f < d
    if above and not _on_format_grid(*dtype_top, ext):
        return None, (
            f"{label} reaches past {name}'s largest finite value, which is not on the "
            f"format's grid: a {name} input near it rounds up out of {name}'s range and "
            f"is stored as infinity, whatever the saturation mode"
        )
    saturates = saturation is not SaturationMode.SAT_NONE
    if below and saturates and not _in_storage(*top, st):
        return None, (
            f"{label}'s largest finite value, ({top[0]}) * 2^{top[1]}, is not a {name} "
            f"value: a {name} input above it saturates onto it, and storing the result "
            f"rounds it a second time"
        )
    hole = ext.normal_exp - 1
    if (
        ext.below == "EXTENDED_NORMALS"
        and m
        and _in_storage(1, hole, st)
        and not _in_storage((1 << m) + 1, hole - m, st)
    ):
        return None, (
            f"{label} has no value at 2^{hole}, and {name} does: the first value above "
            f"that hole, (2^{m} + 1) * 2^{hole - m}, is where a {name} input there rounds, "
            f"and it is not a {name} value"
        )
    return None, None


def _storage_findings(
    ext: _Extent, storage: torch.dtype, elementwise: bool, saturation: SaturationMode
) -> tuple[str | None, str | None]:
    """Hold an `_Extent` against a storage dtype under the rule ``elementwise``
    selects, after the one error both rules share: a format with no nonzero
    finite value the dtype can store.
    """
    st = _STORAGE[storage]
    if ext.top_exp < st.min_exp:
        return (
            f"{ext.label}'s largest finite value is below 2^{ext.top_exp + 1}, under "
            f"{st.name}'s smallest positive value 2^{st.min_exp}: a {st.name} result "
            f"stores none of the format's nonzero finite values"
        ), None
    if elementwise:
        return _storage_edge_findings(ext, st, saturation)
    return _storage_value_set_findings(ext, st)
