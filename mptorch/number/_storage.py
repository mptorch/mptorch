"""Checks that a storage dtype narrower than the carrier can hold a simulated format's results.

Three number formats meet in every call, and this module keeps their names
apart:

* the *simulated format*: the format the cast rounds to (a `BinaryK`,
  `SuperFP` or `FixedPoint`);
* the *carrier*: the binary float the cast computes in, binary32 or binary64
  (`_carrier`);
* the *storage dtype*: the torch dtype of the result tensor, when it is
  narrower than the carrier.

A float16 or bfloat16 tensor is rounded in binary32, its default carrier, and
the result is converted back to the storage dtype, rounding to nearest, when
it is written (by `SIMDTraits`' store in the elementwise quantizers,
`store_elem` in the GEMMs, and a `static_cast` on the host). Under
`carrier=torch.float64` the same applies to float32 tensors: the operands are
widened, rounded in binary64, and the result is narrowed once, to nearest, by
`mptorch.quant.ops._narrowed`. Either way the conversion is a second
rounding. A simulated format whose values are not all values of the storage
dtype is therefore quantized only partially, like a simulated format that
exceeds its carrier, with one important difference: a result above the
storage dtype's range is stored as an infinity *whatever the simulated
format's saturation mode says*, so even a `SAT_FINITE` format can overflow.

Which results an operation can produce depends on the operation, and the two
cases differ enough to need two rules:

  a GEMM       rounds sums and products, which can take any value, so every
               value of the simulated format can occur in the result, and all
               of them must be values of the storage dtype. This is the
               carrier's "representable" test (`_carrier`) applied to the
               storage dtype. The simulated format must meet every
               requirement in the table, and the last column says what a
               call does when it does not:

                 requirement on the        storage dtype                        if not
                 simulated format          float16     bfloat16    float32      met
                 precision P               <= 11       <= 8        <= 24        raises
                 largest value             < 2**16     < 2**128    < 2**128     warns
                 smallest gap in values    >= 2**-24   >= 2**-133  >= 2**-149   warns

  elementwise  rounds the input tensor's own values, which are already values
               of the storage dtype. Where the simulated format is at least
               as fine as the storage dtype, it returns them unchanged, so
               extra precision or a finer bottom does no harm. Where it is
               coarser, an input rounds to a neighbour on the simulated
               format's coarser grid, which is a value of the storage dtype
               too. A result that the storage dtype cannot hold can therefore
               only come from one of three edge cases, each of which warns:

               - The simulated format's largest value is inside the storage
                 dtype's range but is not a value of the storage dtype, and a
                 larger input saturates onto it (in a saturating mode).
               - The simulated format reaches beyond the storage dtype's
                 largest value, which is not a value of the simulated format,
                 so an input near it rounds up, out of the storage dtype's
                 range.
               - Under `EXTENDED_NORMALS` the simulated format has no value at
                 the power of two at the foot of its lowest binade (the
                 "hole"). An input there, which the storage dtype holds, can
                 round up to the next value of the simulated format, which the
                 storage dtype may not hold.

Both rules raise an error for a simulated format whose largest value is below
the storage dtype's smallest positive value, since none of its nonzero values
can be stored (the storage counterpart of the carrier's "no finite value"
error). Neither rule has the carrier's "faithful" limit: that limit exists
because the casts read the exponent field, whereas converting a value to the
storage dtype reads no field and always rounds correctly.

Under binary32, only the GEMM's precision limit matters for bfloat16:
bfloat16 has binary32's exponent range and differs from it only in
precision, so a simulated format that binary32 holds, with at most eight bits
of precision, fits bfloat16 at both ends. Under binary64 the carrier accepts
ranges that no storage dtype holds, so every row of the table can apply.
Otherwise the rules do not depend on the carrier, which is wider than the
storage dtype in either case and represents its values exactly.

The storage dtype is chosen by the call, not by the simulated format (a
`BinaryK` does not know what it will be stored in), so the family modules'
`check_*_storage` functions run on each call, from `mptorch.quant.ops`, on the
simulated format whose values the result holds: the quantizer's format, or a
GEMM's accumulate or fused format. A GEMM's multiply format is never stored
(its products stay in the carrier), and when the running sum is not
quantized, no simulated format's values are stored at all.
"""

from typing import NamedTuple

import torch

from ._carrier import _Extent
from .modes import SaturationMode


class _Storage(NamedTuple):
    """A storage dtype, described by the numbers `_storage_findings` uses.
    Each ``*_exp`` field is an unbiased exponent."""

    name: str
    man_bits: int  # stored mantissa bits, without the implicit leading bit
    top_exp: int  # the exponent of the largest finite value
    min_exp: int  # the exponent of the smallest positive value, a subnormal


_STORAGE: dict[torch.dtype, _Storage] = {
    torch.float16: _Storage("float16", 10, 15, -24),
    torch.bfloat16: _Storage("bfloat16", 7, 127, -133),
    torch.float32: _Storage("float32", 23, 127, -149),
}


def _normalized(M: int, k: int) -> tuple[int, int, int]:
    """Rewrite ``M * 2**k`` (``M > 0``) as ``(M', k', lead)``: an odd ``M'``
    with ``M' * 2**k'`` the same value, and ``lead``, the exponent of its
    leading bit.

    Moving the trailing zero bits of ``M`` into the exponent gives each value
    a single representation, so two spellings of the same value compare
    equal.
    """
    tz = (M & -M).bit_length() - 1
    M >>= tz
    return M, k + tz, k + tz + M.bit_length() - 1


def _in_storage(M: int, k: int, st: _Storage) -> bool:
    """Return whether ``M * 2**k`` is a finite value of the storage dtype.

    It is one when it is a multiple of the storage dtype's spacing at that
    magnitude (``2**(E - man_bits)`` in the normal binade starting at
    ``2**E``, and the fixed subnormal spacing below the normals) and is not
    above the storage dtype's largest value. A value that passes the spacing
    test has at most ``man_bits + 1`` significant bits, so it is at most the
    largest value exactly when its leading bit is not above the storage
    dtype's top binade.
    """
    _, k, lead = _normalized(M, k)
    min_normal_exp = st.min_exp + st.man_bits
    return lead <= st.top_exp and k >= max(lead, min_normal_exp) - st.man_bits


def _on_format_grid(M: int, k: int, ext: _Extent) -> bool:
    """Return whether ``M * 2**k`` is a value of the simulated format, with
    its grid continued past its largest value.

    The top is ignored on purpose: the caller asks where an input at that
    magnitude would round to, and the grid continued above the largest value
    answers that. Below the binades with a full mantissa, the answer depends
    on what the simulated format has there (subnormals, the extra binade of
    ``EXTENDED_NORMALS``, supernormals, or nothing).
    """
    M, k, lead = _normalized(M, k)
    m = ext.man_bits
    if lead >= ext.normal_exp:
        return k >= lead - m
    if ext.below == "SUBNORMALS":
        return k >= ext.normal_exp - m
    if ext.below == "EXTENDED_NORMALS" and m and lead == ext.normal_exp - 1:
        return k >= lead - m and M != 1  # the power of two at that binade's foot is the hole
    if ext.below == "SUPERNORMALS":
        return M == 1 and lead >= ext.bottom_exp
    return False


def _storage_value_set_findings(ext: _Extent, st: _Storage) -> tuple[str | None, str | None]:
    """The GEMM rule: whether every value of the simulated format is a value
    of the storage dtype.

    Return ``(error, warning)``. Precision is checked first, because the two
    range checks are exact only once it holds. If the simulated format has no
    more significand bits than the storage dtype, then:

    * a value of the simulated format in a binade where the storage dtype has
      normal numbers is one of them;
    * a smaller value of the simulated format is a multiple of
      ``2**step_exp`` below the storage dtype's smallest normal, which is a
      subnormal of the storage dtype as long as ``step_exp`` is not below the
      exponent of its smallest subnormal;
    * every value of the simulated format is below ``2**(top_exp + 1)``, and
      so at most the storage dtype's largest finite value, as long as
      ``top_exp`` is not above the storage dtype's.
    """
    label, name = ext.label, st.name
    if ext.man_bits > st.man_bits:
        return (
            f"{label} has {ext.man_bits + 1} bits of precision, and a {name} result holds "
            f"{st.man_bits + 1}: a GEMM can produce any of the format's values, and storing "
            f"one that {name} cannot represent rounds it a second time"
        ), None
    if ext.top_exp > st.top_exp:
        return None, (
            f"{label} reaches 2^{ext.top_exp}, above {name}'s largest finite value, "
            f"2^{st.top_exp + 1} - 2^{st.top_exp - st.man_bits}: storing a result in {name} "
            f"turns the format's values above that into infinity, whatever the saturation "
            f"mode (a bias of at least {ext.top_field - st.top_exp} avoids this)"
        )
    if ext.step_exp < st.min_exp:
        return None, (
            f"{label} has values 2^{ext.step_exp} apart at the bottom of its range, finer "
            f"than {name}'s smallest subnormal, 2^{st.min_exp}: storing a result in {name} "
            f"rounds those values a second time, onto {name}'s coarser spacing"
        )
    return None, None


def _storage_edge_findings(
    ext: _Extent, st: _Storage, saturation: SaturationMode
) -> tuple[str | None, str | None]:
    """The elementwise rule: whether rounding values of the storage dtype to
    the simulated format can produce a value the storage dtype does not hold.

    Return ``(error, warning)``; this rule only ever warns. An input that is
    already a value of the simulated format comes back unchanged. Any other
    input lies where the simulated format's grid is coarser than the storage
    dtype's, so its two neighbours in the simulated format are values of the
    storage dtype too. That leaves the three edge cases listed in the module
    docstring, and nothing else.

    The case where the simulated format's largest value is not a value of the
    storage dtype matters only in a saturating mode. Such a largest value has
    more significant bits than the storage dtype, so the simulated format's
    grid is finer there, and every storage dtype value above it lies on the
    simulated format's grid continued upward. Rounding leaves such an input
    unchanged; `SAT_NONE` then turns it into an infinity, which the storage
    dtype holds, whereas a saturating mode clamps it to the simulated format's
    largest value, which the storage dtype does not hold.
    """
    label, name, m = ext.label, st.name, ext.man_bits
    top = ((1 << m) + ext.top_code, ext.top_exp - m)  # the simulated format's largest value
    dtype_top = ((1 << (st.man_bits + 1)) - 1, st.top_exp - st.man_bits)  # the storage dtype's
    if ext.top_exp != st.top_exp:
        # in different binades the exponents decide; lining the two
        # significands up as integers would need a shift as large as the
        # exponent difference, which a 30-bit exponent field makes enormous
        above, below = ext.top_exp > st.top_exp, ext.top_exp < st.top_exp
    else:
        finer = max(m, st.man_bits)
        f, d = top[0] << (finer - m), dtype_top[0] << (finer - st.man_bits)
        above, below = f > d, f < d
    if above and not _on_format_grid(*dtype_top, ext):
        return None, (
            f"{label} reaches beyond {name}'s largest finite value, which is not a value of "
            f"the format: a {name} input near it can round up out of {name}'s range, and is "
            f"then stored as infinity, whatever the saturation mode"
        )
    saturates = saturation is not SaturationMode.SAT_NONE
    if below and saturates and not _in_storage(*top, st):
        return None, (
            f"{label}'s largest finite value, {top[0]} x 2^{top[1]}, is not a {name} value: "
            f"a {name} input above it saturates to it, and storing that result in {name} "
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
            f"{label} has no value at 2^{hole}, which {name} has: a {name} input there can "
            f"round up to the format's next value, (2^{m} + 1) x 2^{hole - m}, which is not "
            f"a {name} value"
        )
    return None, None


def _storage_findings(
    ext: _Extent, storage: torch.dtype, elementwise: bool, saturation: SaturationMode
) -> tuple[str | None, str | None]:
    """Compare a simulated format's range (an `_Extent`) with a storage dtype,
    by the elementwise rule or the GEMM rule as ``elementwise`` selects.

    The error both rules share is checked first: a simulated format whose
    largest finite value is below the storage dtype's smallest positive value,
    so that none of its nonzero values can be stored.
    """
    st = _STORAGE[storage]
    if ext.top_exp < st.min_exp:
        return (
            f"{ext.label}'s largest finite value is below 2^{ext.top_exp + 1}, under "
            f"{st.name}'s smallest positive value, 2^{st.min_exp}: a {st.name} result "
            f"stores none of the format's nonzero values"
        ), None
    if elementwise:
        return _storage_edge_findings(ext, st, saturation)
    return _storage_value_set_findings(ext, st)
