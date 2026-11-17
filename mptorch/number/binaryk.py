"""IEEE P3109's binaryK formats, and the checks that hold one to a carrier and a dtype.

:class:`BinaryK` is the format. Its checks hold it against the carrier a call
rounds in and the dtype the call's result is stored in, with machinery the
other families share: `_carrier` for the carrier and `_storage` for the
dtype. What is binaryK's own is how its range is read off the parameters,
`_binaryK_extent`, whose floor depends on the subnormals mode.
"""

import warnings
from dataclasses import KW_ONLY, dataclass
from functools import lru_cache

import torch

from ._carrier import (
    _BINARY64,
    _bottom_by_field,
    _Carrier,
    _Extent,
    _largest_finite,
    _pointing_wider,
    _range_findings,
    _top_of_range,
    _width_error,
)
from ._storage import _storage_findings
from .base import FloatFormat, FormatRangeWarning, _report_per_call, _resolve_domain
from .modes import Domain, SaturationMode, SubnormalsMode


@dataclass(frozen=True)
class BinaryK(FloatFormat):
    """An IEEE P3109 binaryK float: ``K`` total bits of which ``P`` are precision.

    P3109 is the draft Standard for Arithmetic Formats for Machine Learning
    (Fitzgibbon, Wintersteiger and Sarnoff give an overview in
    arXiv:2606.04028). It parameterizes its binary formats by bitwidth,
    precision (the implicit leading bit included), signedness and domain, and
    names them ``Binary<K>p<P><s|u><e|f>``. ``K``, ``P``, ``is_signed`` and
    ``domain`` are those four. ``BinaryK(8, 4)`` is ``Binary8p4se``, and
    ``BinaryK(8, 4, domain=Domain.FINITE, saturation=SaturationMode.SAT_FINITE)``
    is ``Binary8p4sf``. The finite domain admits only ``SAT_FINITE``, so that
    saturation mode alone also selects it.

    Three of P3109's conventions are worth knowing:

    * The format has one zero, unsigned, at the all-zero code. No cast returns
      ``-0.0``.
    * In a signed format the NaN is the code with only the sign bit set
      (which would hold ``-0.0``). In an unsigned format it is the top code.
    * The largest finite value is counted in code points. With one or two
      mantissa bits, the codes reserved for infinity and NaN can push it a
      binade or two down.

    :doc:`/concepts` maps the rest of the standard onto this class and says
    where MPTorch goes beyond it.

    The rounding mode is a property of the operation, not of the format (see
    :class:`mptorch.quant.Quant` and :class:`mptorch.quant.SplitMac`), because
    the kernels take it as a template parameter shared by a multiply and its
    accumulate. The instance is frozen, and therefore hashable, because the
    code that resolves a format memoizes on its value. The derived ``bias`` is
    filled in at construction, not at each call site.

    Args:
        K (int): total bits, sign included.
        P (int): bits of precision, the implicit leading bit included, so
            ``P - 1`` mantissa bits are stored; ``1 <= P <= K``.
        bias (int, optional): exponent bias. Default: P3109's, ``2**(K-P-1)``
            signed and ``2**(K-P)`` unsigned, which encodes 1.0 at the middle
            code point. Any other bias gives a format outside the standard.
            The layout of OCP's E4M3 is spelled this way:
            ``BinaryK(8, 4, bias=7)``.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        prng_bits (int): width of the random mantissa drawn by
            :attr:`RoundMode.SR`, ignored under every other rounding mode.
            Default: 0.
        saturation (SaturationMode): overflow behaviour. Default:
            ``SaturationMode.SAT_NONE``.
        domain (Domain, optional): whether the format has infinities.
            ``Domain.FINITE`` admits only ``SAT_FINITE``. ``Domain.EXTENDED``
            with ``SAT_FINITE`` is stored as
            ``SaturationMode.SAT_FINITE_EXTENDED``. Default: ``None``, which
            follows from ``saturation``: finite for ``SAT_FINITE``, extended
            otherwise. The instance stores the resolved domain.
        subnormals (SubnormalsMode): what lies below the smallest normal. Any
            value other than ``SUBNORMALS`` gives a format outside the
            standard. Default: ``SubnormalsMode.SUBNORMALS``.

    Raises:
        ValueError: for the finite domain with a saturation mode other than
            ``SAT_FINITE``. Otherwise only for a format that neither carrier
            can simulate: more than 53 bits of precision,
            ``P - 1 + prng_bits`` above 52, an exponent field past 30 bits, a
            negative ``prng_bits``, or no finite value that binary64 holds.
            Nothing warns here. Whether float32 or float64 arithmetic carries
            the format depends on the call, so each call checks the format
            against its own carrier (see :class:`FormatRangeWarning`).
        TypeError: if ``saturation`` is not a ``SaturationMode`` or
            ``domain`` is not a ``Domain``.

    Example::

        >>> from mptorch import BinaryK
        >>> BinaryK(8, 4).bias  # P3109 Binary8p4se
        8
        >>> BinaryK(8, 4, is_signed=False).bias  # Binary8p4ue
        16
        >>> BinaryK(8, 4, bias=7).man_bits  # OCP E4M3's layout
        3
    """

    K: int
    P: int
    _: KW_ONLY
    bias: int | None = None
    is_signed: bool = True
    prng_bits: int = 0
    saturation: SaturationMode = SaturationMode.SAT_NONE
    domain: Domain | None = None
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS

    def __post_init__(self) -> None:
        if self.P < 1 or self.K < self.P:
            raise ValueError(f"BinaryK needs 1 <= P <= K, got K={self.K}, P={self.P}")
        saturation, domain = _resolve_domain(self.saturation, self.domain, "BinaryK")
        object.__setattr__(self, "saturation", saturation)
        object.__setattr__(self, "domain", domain)
        if self.prng_bits < 0:
            raise ValueError(f"BinaryK prng_bits must be non-negative, got {self.prng_bits}")
        if self.bias is None:
            middle = 2 ** (self.K - self.P - 1) if self.is_signed else 2 ** (self.K - self.P)
            object.__setattr__(self, "bias", middle)
        assert self.bias is not None
        # Raise for what no carrier can do, and nothing more: the carrier
        # belongs to the call, which holds the format to it (`check_binaryK`)
        check_binaryK(
            self.K,
            self.P,
            self.bias,
            self.is_signed,
            self.saturation,
            self.subnormals,
            self.prng_bits,
            carrier=torch.float64,
            warn=False,
        )

    @property
    def man_bits(self) -> int:
        """Mantissa bits, as the cast counts them (the implicit bit is not one)."""
        return self.P - 1

    @property
    def max_finite(self) -> float:
        """The largest finite value, as P3109 counts it, in code points.

        In the extended domain the top code is infinity, and in an unsigned
        format the code below it, or the top one in the finite domain, is the
        NaN; with one or two mantissa bits those can move the value a binade
        or two down. It is the value ``SaturationMode.SAT_FINITE`` saturates
        to, and the one a scaled quantizer maps a tensor's largest magnitude
        onto.

        Example::

            >>> from mptorch import BinaryK, SaturationMode
            >>> BinaryK(8, 4).max_finite  # Binary8p4se: the top code is infinity
            224.0
            >>> BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE).max_finite
            240.0
        """
        assert self.bias is not None  # set by __post_init__
        # P3109's reserved codes, as `_binaryK_extent` counts them
        reserved = (self.saturation is not SaturationMode.SAT_FINITE) + (not self.is_signed)
        exp_bits = self.K - self.P + (not self.is_signed)
        return _largest_finite(exp_bits, self.man_bits, self.bias, reserved)


def check_binaryK(
    K: int,
    P: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.SAT_NONE,
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype = torch.float32,
    warn: bool = True,
    stacklevel: int = 3,
) -> None:
    """Hold a binaryK format against what a carrier can hold.

    A format that cannot function in the carrier is an error; one whose range
    merely outruns it quantizes partially rather than wrongly, and is a
    warning. :class:`BinaryK` calls this with ``carrier=torch.float64`` and
    ``warn=False``, which raises for what no carrier can do and says nothing
    else, because which carrier a format meets is the call's to know; the
    call holds it to that one with :func:`check_binaryK_carrier`.

    Args:
        K (int): total bits of the format.
        P (int): bits of precision, the implicit leading bit included.
        bias (int): the resolved exponent bias, never ``None``.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.SAT_NONE``.
        subnormals (SubnormalsMode): what lies below the smallest normal.
            Default: ``SubnormalsMode.SUBNORMALS``.
        prng_bits (int): random bits drawn for stochastic rounding. Default: 0.
        carrier (torch.dtype): ``torch.float32`` for binary32 or
            ``torch.float64`` for binary64. Default: ``torch.float32``.
        warn (bool): whether to issue the warning. Default: ``True``.
        stacklevel (int): how far above this function the caller's own code
            is, so the warning points there. Default: 3.

    Raises:
        ValueError: for a format that cannot function in the carrier: more
            precision than the carrier has, stochastic bits with no
            significand left to draw from, an exponent field past 30 bits, a
            negative ``prng_bits``, or no finite value the carrier holds.
    """
    error, warning = _binaryK_findings(
        K, P, bias, is_signed, saturation, subnormals, prng_bits, carrier
    )
    if error is not None:
        raise ValueError(error)
    if warn and warning is not None:
        warnings.warn(warning, FormatRangeWarning, stacklevel=stacklevel)


def check_binaryK_carrier(
    K: int,
    P: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.SAT_NONE,
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype,
) -> None:
    """:func:`check_binaryK`, reported the way a call reports it.

    The same findings, with the warning naming the first frame outside mptorch
    and torch, the line that made the call, however deep inside them the call
    was reached from (a layer's backward, for instance). The arguments are
    those of :func:`check_binaryK` without ``warn`` and ``stacklevel``, and
    ``carrier`` is required.

    Raises:
        ValueError: as :func:`check_binaryK`.
    """
    _report_per_call(
        *_binaryK_findings(K, P, bias, is_signed, saturation, subnormals, prng_bits, carrier)
    )


def check_binaryK_storage(
    K: int,
    P: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.SAT_NONE,
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    *,
    storage: torch.dtype,
    elementwise: bool = False,
) -> None:
    """Hold a binaryK format against the dtype a result is written in.

    ``storage`` is a dtype narrower than the carrier the cast rounds in:
    ``torch.float16`` or ``torch.bfloat16`` under binary32, and those or
    ``torch.float32`` under binary64. Writing the result rounds it a second
    time, onto the dtype's grid. :func:`check_binaryK_carrier` holds the
    format against the carrier, and this does not repeat it. Which results
    can land off the dtype's grid depends on what was rounded, so there are
    two rules (``mptorch/number/_storage.py`` derives both): a quantizer's
    inputs are already the dtype's values, and only three edges of the
    format's range can move one off the grid, whereas a GEMM's sums reach
    every value the format has, so its whole value set must be the dtype's. The dtype belongs
    to the call, not the format, so :mod:`mptorch.quant.ops` runs this per
    call on the format whose values the result holds, and the warning names
    the first frame outside mptorch and torch.

    Args:
        K (int): total bits of the format.
        P (int): bits of precision, the implicit leading bit included.
        bias (int): the resolved exponent bias.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.SAT_NONE``.
        subnormals (SubnormalsMode): what lies below the smallest normal.
            Default: ``SubnormalsMode.SUBNORMALS``.
        storage (torch.dtype): the dtype the result is stored in.
        elementwise (bool): ``True`` for a quantizer, ``False`` for a GEMM.
            Default: ``False``.

    Raises:
        ValueError: where nothing of the format survives being stored (its
            largest value is below the dtype's smallest), or, for a GEMM,
            where the format has more precision than the dtype.
    """
    _report_per_call(
        *_binaryK_storage_findings(
            K, P, bias, is_signed, saturation, subnormals, storage, elementwise
        )
    )


# --- what the checks find ----------------------------------------------------


def _binaryK_label(K: int, P: int, bias: int, is_signed: bool, subnormals: SubnormalsMode) -> str:
    """The format as its constructor call, for messages; defaults are omitted."""
    return (
        f"BinaryK(K={K}, P={P}, bias={bias}"
        + ("" if is_signed else ", is_signed=False")
        + ("" if subnormals is SubnormalsMode.SUBNORMALS else f", {subnormals.name}")
        + ")"
    )


# The other two reasons `_Extent.bottom_why` can give, both at half a
# `NORMALS` or `EXTENDED_NORMALS` floor; `_binaryK_extent` says where each
# applies.


def _bottom_tie_by_field(c: _Carrier) -> str:
    """Why a one-code-per-binade format's floor cannot sit at the smallest normal."""
    return (
        "with one code per binade the cast rounds by exponent field alone, and half that "
        f"value, 2^{c.min_normal_exp - 1}, is a {c.name} subnormal, in the field every "
        "one of them shares -- so round-to-nearest-even reads it as a tie between two "
        "binades and returns the smallest value, where the format's answer is zero"
    )


def _bottom_half_unheld(c: _Carrier) -> str:
    """Why a full-precision extended-normals floor needs a step the carrier lacks."""
    return (
        f"half that value, 2^{c.min_normal_exp - 1} x (1 + 2^-{c.man_bits}), is the "
        f"round-to-nearest boundary below it and needs a step {c.name} does not have, so "
        f"the cast holds it as 2^{c.min_normal_exp - 1} -- and round-to-nearest-away "
        "returns the smallest value for that input, where the format's answer is zero"
    )


def _binaryK_extent(
    label: str,
    man_bits: int,
    exp_bits: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    subnormals: SubnormalsMode,
    carrier: _Carrier,
) -> _Extent:
    """A binaryK format's range, as an `_Extent`.

    Call it only once `_width_error` has passed, since this shifts by
    ``exp_bits``. The top comes from `_top_of_range`; the bottom and the finest
    step depend on the subnormals mode, and so does the floor the carrier's
    cast is exact down to.
    """
    # P3109's reserved codes: +infinity outside the finite domain, and NaN
    # above it in an unsigned format (`make_binaryK_params`)
    reserved = (saturation is not SaturationMode.SAT_FINITE) + (0 if is_signed else 1)
    top_field, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved, carrier)
    min_exp = 1 - bias  # the smallest normal's exponent
    if subnormals is SubnormalsMode.SUBNORMALS:
        # the subnormals are one grid of fixed spacing, and its step is both
        # the smallest value and the finest spacing in the format
        bottom_exp = step_exp = min_exp - man_bits
    elif subnormals is SubnormalsMode.NORMALS:
        bottom_exp, step_exp = min_exp, min_exp - man_bits
    else:
        # EXTENDED_NORMALS: one more binade of normals below the rest, minus
        # its mantissa-zero code, which is still the zero, so the smallest
        # value is one step into that binade; with man_bits == 0 the step is
        # the whole binade and the floor is the one NORMALS has
        bottom_exp = min_exp - 1 if man_bits else min_exp
        step_exp = bottom_exp - man_bits
    # How low the floor may go is not the same for the three modes, and the
    # difference is which of them reads an exponent field down there. In
    # binary32's numbers, binary64's following in parentheses:
    # `SUBNORMALS` derives its grid from the input's exponent field, which
    # every subnormal of the carrier shares, so it is blind below 2**-126
    # (2**-1022) and needs its floor a binade above that. `NORMALS` and
    # `EXTENDED_NORMALS` decide by comparing magnitudes as words, which reads
    # no field at all, so they are exact to 2**-126, and no further, because
    # below it the floor leaves the carrier's normals and nothing flushes. Two
    # of their formats stop a binade short of that, both at half the floor,
    # the one point the compare needs besides the floor itself:
    #
    #   man_bits == 0   the round before the compare reads a field after all.
    #                   With no mantissa it rounds between binades, and a floor
    #                   at 2**-126 puts half of it, 2**-127, in field 0, where
    #                   the round sees a tie between binades and carries it up
    #                   to the floor before the compare can send it to zero.
    #   EXTENDED, full  at the carrier's full precision (P = 24, 53) half the
    #   precision       floor, 2**-127 * (1 + 2**-23), needs a 2**-150 step
    #                   (2**-1075); the carrier holds it as 2**-127, and
    #                   round-to-nearest-away takes that input up to the floor.
    #
    # `dev/benchmarks/sweeps/cast_all_modes_sweep.cu` found both in binary32 by
    # checking every rounding mode over all 2**32 inputs; binary64 is exact
    # at both points, and the same bounds apply one carrier wider.
    if subnormals is SubnormalsMode.SUBNORMALS:
        min_bottom, bottom_why = carrier.min_simulable_exp, _bottom_by_field(carrier)
    elif man_bits == 0:
        min_bottom, bottom_why = carrier.min_simulable_exp, _bottom_tie_by_field(carrier)
    elif subnormals is SubnormalsMode.EXTENDED_NORMALS:
        # half the floor's last bit, 2**(bottom_exp - 1 - man_bits), must exist
        min_bottom = max(carrier.min_normal_exp, man_bits + 1 + carrier.min_subnormal_exp)
        bottom_why = _bottom_half_unheld(carrier)
    else:
        min_bottom, bottom_why = carrier.min_normal_exp, _bottom_by_field(carrier)
    if bottom_exp < carrier.min_normal_exp:
        # the floor itself is out of reach, which says more
        bottom_why = _bottom_by_field(carrier)
    return _Extent(
        label,
        exp_bits,
        man_bits,
        top_field,
        top_exp,
        top_code,
        bottom_exp,
        step_exp,
        min_bottom,
        bottom_why,
        min_exp,
        subnormals.name,
    )


@lru_cache(maxsize=512)
def _binaryK_findings(
    K: int,
    P: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    subnormals: SubnormalsMode,
    prng_bits: int,
    carrier: torch.dtype,
) -> tuple[str | None, str | None]:
    """What ``carrier`` says about a binaryK format, as ``(error, warning)``.

    Memoized on the whole parameter tuple, which is why it takes plain values
    rather than a :class:`BinaryK`. The widths are checked first, since the
    range derivation shifts by the exponent width.
    """
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    label = _binaryK_label(K, P, bias, is_signed, subnormals)

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        error = _width_error(label, exp_bits, man_bits, prng_bits, c)
        if error is not None:
            return error, None
        ext = _binaryK_extent(label, man_bits, exp_bits, bias, is_signed, saturation, subnormals, c)
        return _range_findings(ext, c)

    return _pointing_wider(find, carrier)


@lru_cache(maxsize=256)
def _binaryK_storage_findings(
    K: int,
    P: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    subnormals: SubnormalsMode,
    storage: torch.dtype,
    elementwise: bool,
) -> tuple[str | None, str | None]:
    """What storing a binaryK result in ``storage`` says, as ``(error, warning)``.

    Memoized on the parameter tuple, like `_binaryK_findings`. A format the
    carrier check has already rejected yields ``(None, None)``, since no range
    can be read from it and the carrier's error is the one to report.
    """
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    label = _binaryK_label(K, P, bias, is_signed, subnormals)
    # Read against binary64, the wider carrier. The extent a storage rule reads
    # (the top, the step, the floor) is the format's own and the same in
    # either carrier for a format binary32 holds, and a format only binary64
    # holds can reach storage too, under `carrier=torch.float64`.
    if _width_error(label, exp_bits, man_bits, 0, _BINARY64) is not None:
        # the carrier's to report, and the carrier check runs before any call
        # reaches this: no range can be read from such a format
        return None, None
    ext = _binaryK_extent(
        label, man_bits, exp_bits, bias, is_signed, saturation, subnormals, _BINARY64
    )
    if ext.top_field < 1 or ext.top_exp < _BINARY64.min_normal_exp:
        return None, None  # likewise: no finite value at all, the carrier's error
    return _storage_findings(ext, storage, elementwise, saturation)
