"""IEEE P3109's binaryK formats, and checks that a carrier and a storage dtype can hold one.

:class:`BinaryK` describes the format. Its checks compare it with the carrier
a call rounds in and with the storage dtype of the call's result, using
code shared with the other families (`_carrier` for the carrier, `_storage`
for the storage dtype). The part specific to binaryK is `_binaryK_extent`, which
computes the format's range from its parameters; the lowest value the carrier
allows depends on the subnormals mode.
"""

import warnings
from dataclasses import KW_ONLY, dataclass
from functools import lru_cache

import torch

from ._carrier import (
    _BINARY64,
    _bottom_by_field,
    _Carrier,
    _exponent_error,
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
    """An IEEE P3109 binaryK floating-point format: ``K`` bits, ``P`` of them precision.

    P3109 is the draft Standard for Arithmetic Formats for Machine Learning
    (Fitzgibbon, Wintersteiger and Sarnoff give an overview in
    arXiv:2606.04028). It describes each binary format by four parameters:
    its bit width, its precision (including the implicit leading bit),
    whether it is signed, and its domain. It names the format
    ``Binary<K>p<P><s|u><e|f>`` after them, and they are this class's ``K``,
    ``P``, ``is_signed`` and ``domain``. ``BinaryK(8, 4)`` is
    ``Binary8p4se``, and
    ``BinaryK(8, 4, domain=Domain.FINITE, saturation=SaturationMode.SAT_FINITE)``
    is ``Binary8p4sf``. Since the finite domain allows only ``SAT_FINITE``,
    ``saturation=SaturationMode.SAT_FINITE`` alone also selects it.

    Three of P3109's conventions are worth knowing:

    * The format has a single zero, unsigned, at the all-zero code. No cast
      returns ``-0.0``.
    * In a signed format the NaN is the code with only the sign bit set
      (which would otherwise encode ``-0.0``). In an unsigned format it is the
      top code.
    * The largest finite value is the highest code not reserved for infinity
      or NaN. With one or two bits of precision (``P`` of 1 or 2), the
      reserved codes can fill the top binade or more, which moves the largest
      finite value one or two binades down.

    :doc:`/concepts` maps the rest of the standard onto this class and says
    where MPTorch goes beyond it.

    The rounding mode is not part of the format: it belongs to the operation
    (see :class:`mptorch.quant.Quant` and :class:`mptorch.quant.SplitMac`),
    because the kernels take it as a compile-time parameter shared by a
    GEMM's multiply and accumulate. The object is frozen, and therefore
    hashable, because the code that turns a format into kernel arguments
    caches its result keyed on the format. The default ``bias`` is computed
    once, when the object is built.

    Args:
        K (int): total bits, including the sign bit.
        P (int): bits of precision, including the implicit leading bit, so
            ``P - 1`` mantissa bits are stored; ``1 <= P <= K``.
        bias (int, optional): exponent bias. Default: P3109's, ``2**(K-P-1)``
            if signed and ``2**(K-P)`` if unsigned, which puts 1.0 at the
            middle code. Any other bias gives a format outside the standard;
            OCP's E4M3 bit layout, for example, is ``BinaryK(8, 4, bias=7)``.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        prng_bits (int): number of random bits drawn by the stochastic
            rounding modes (:attr:`RoundMode.SR`, ``SR_B``, ``SR_C``);
            ignored by the other modes. Default: 0.
        saturation (SaturationMode): what happens on overflow. Default:
            ``SaturationMode.SAT_NONE``.
        domain (Domain, optional): whether the format has infinities.
            ``Domain.FINITE`` allows only ``SAT_FINITE``, and
            ``Domain.EXTENDED`` with ``SAT_FINITE`` is stored as
            ``SaturationMode.SAT_FINITE_EXTENDED``. Default: ``None``, meaning
            that the domain follows from ``saturation``: finite for
            ``SAT_FINITE``, extended otherwise. The object stores the
            resolved domain.
        subnormals (SubnormalsMode): what the format has below its smallest
            normal number. Modes other than ``SUBNORMALS`` give formats
            outside the standard. Default: ``SubnormalsMode.SUBNORMALS``.

    Raises:
        ValueError: if ``P`` is not in ``1 .. K``, or for the finite domain
            with a saturation mode other than ``SAT_FINITE``. Otherwise only
            for a format that neither carrier can simulate: more than 53 bits
            of precision, ``P - 1 + prng_bits`` above 52, an exponent field
            wider than 30 bits, a bias that puts ``1 - bias`` or
            ``2**exp_bits - 1 - bias`` outside -2^30 to 2^30 (the kernels
            compute them in a 32-bit int), a negative ``prng_bits``, or no
            finite normal value that binary64 can hold. Building the format never warns:
            whether binary32 or binary64 arithmetic carries it is decided by
            each call, which checks the format against its own carrier (see
            :class:`FormatRangeWarning`).
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
        # Raise only for what no carrier can do; each call checks the format
        # against its own carrier (`check_binaryK_carrier`)
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
        """Stored mantissa bits, ``P - 1`` (the implicit leading bit is not stored)."""
        return self.P - 1

    @property
    def max_finite(self) -> float:
        """The largest finite value, counted in code points as P3109 counts it.

        The highest codes of the top binade can be reserved: in an unsigned
        format the top code is the NaN, and in the extended domain the next
        code down (the top code, if the format is signed) is the infinity.
        With one or two bits of precision, the reserved codes can move the
        largest finite value one or two binades down. It is the value that a
        saturating mode clamps to.

        Example::

            >>> from mptorch import BinaryK, SaturationMode
            >>> BinaryK(8, 4).max_finite  # Binary8p4se: the top code is infinity
            224.0
            >>> BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE).max_finite
            240.0
        """
        assert self.bias is not None  # set by __post_init__
        # the codes P3109 reserves for infinity and NaN, counted as in `_binaryK_extent`
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
    """Check whether a carrier can hold a binaryK format.

    A format that cannot work in the carrier raises an error. A format whose
    range merely exceeds the carrier's is quantized partially but not wrongly,
    and gets a :class:`FormatRangeWarning`. :class:`BinaryK` calls this with
    ``carrier=torch.float64`` and ``warn=False``, which raises only for what
    no carrier can do and never warns, because the carrier is chosen by each
    call; the call checks the format against its own carrier with
    :func:`check_binaryK_carrier`.

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
        stacklevel (int): passed to ``warnings.warn``, so that the warning
            points at the caller's code. Default: 3, the caller of this
            function's caller.

    Raises:
        ValueError: for a format that cannot work in the carrier: more
            precision than the carrier has, more stochastic-rounding bits
            than fit in the carrier's significand next to the mantissa, an
            exponent field wider than 30 bits, a bias that puts ``1 - bias``
            or ``2**exp_bits - 1 - bias`` outside -2^30 to 2^30 (the kernels
            compute them in a 32-bit int), a negative ``prng_bits``, or no
            finite normal value that the carrier can hold.
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
    """:func:`check_binaryK` as run on each call of a quantizer or GEMM.

    It finds the same problems, but attributes the warning to the first
    stack frame outside mptorch and torch, which is the user's line that made
    the call, however deeply the check runs inside them (in a layer's
    backward pass, for example). The arguments are those of
    :func:`check_binaryK` without ``warn`` and ``stacklevel``, and
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
    """Check whether the storage dtype of a result can hold a binaryK format's results.

    ``storage``, the storage dtype, is the dtype of the result tensor, which
    is narrower than the carrier the cast rounds in: ``torch.float16`` or
    ``torch.bfloat16`` under binary32, and those or ``torch.float32`` under
    binary64. Writing the result to it rounds the result a second time, to a
    value of the storage dtype. This check does not repeat the carrier check
    of :func:`check_binaryK_carrier`.

    There are two rules, because which results can fall outside the storage
    dtype depends on what was rounded (``mptorch/number/_storage.py`` derives
    both). A quantizer's inputs are already values of the storage dtype, and
    only three edge cases of the binaryK format's range can turn one into a
    result the storage dtype does not hold. A GEMM's sums can take every value
    of the binaryK format, so all of them must be values of the storage dtype.
    The storage dtype is chosen by the call, not by the binaryK format, so
    :mod:`mptorch.quant.ops` runs this check on each call, on the format whose
    values the result holds, and attributes the warning to the first stack
    frame outside mptorch and torch.

    Args:
        K (int): total bits of the format.
        P (int): bits of precision, the implicit leading bit included.
        bias (int): the resolved exponent bias.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.SAT_NONE``.
        subnormals (SubnormalsMode): what lies below the smallest normal.
            Default: ``SubnormalsMode.SUBNORMALS``.
        storage (torch.dtype): the storage dtype, the dtype of the result tensor.
        elementwise (bool): ``True`` for a quantizer, ``False`` for a GEMM.
            Default: ``False``.

    Raises:
        ValueError: if the storage dtype can store none of the binaryK
            format's nonzero values (the format's largest value is below the
            storage dtype's smallest positive value), or, for a GEMM, if the
            format has more precision than the storage dtype.
    """
    _report_per_call(
        *_binaryK_storage_findings(
            K, P, bias, is_signed, saturation, subnormals, storage, elementwise
        )
    )


# --- how the checks reach their findings -------------------------------------


def _binaryK_label(K: int, P: int, bias: int, is_signed: bool, subnormals: SubnormalsMode) -> str:
    """The format written as a constructor call, for messages, leaving out
    ``is_signed`` and ``subnormals`` when they have their default values."""
    return (
        f"BinaryK(K={K}, P={P}, bias={bias}"
        + ("" if is_signed else ", is_signed=False")
        + ("" if subnormals is SubnormalsMode.SUBNORMALS else f", subnormals={subnormals}")
        + ")"
    )


# The two other explanations that `_Extent.bottom_why` can carry. Both concern
# the input at half the smallest value of a `NORMALS` or `EXTENDED_NORMALS`
# format; `_binaryK_extent` says when each applies.


def _bottom_tie_by_field(c: _Carrier) -> str:
    """Explain why a format with one value per binade (``P = 1``) cannot have
    its smallest value at the carrier's smallest normal."""
    return (
        "with one value per binade the cast rounds by the exponent field alone, and half "
        f"the smallest value, 2^{c.min_normal_exp - 1}, is a {c.name} subnormal, whose "
        "exponent field all subnormals share, so round-to-nearest-even sees a tie between "
        "two binades and returns the smallest value instead of zero"
    )


def _bottom_half_unheld(c: _Carrier) -> str:
    """Explain why an ``EXTENDED_NORMALS`` format at the carrier's full
    precision needs a spacing that the carrier does not have."""
    half = f"2^{c.min_normal_exp - 1}"
    return (
        f"half the smallest value, {half} x (1 + 2^-{c.man_bits}), is where round-to-nearest "
        f"switches from zero to the smallest value, and {c.name} cannot represent it, so the "
        f"cast uses {half} instead, and round-to-nearest-away then returns the smallest value "
        f"for an input of {half}, where the right answer is zero"
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
    """Compute a binaryK format's range as an `_Extent`.

    Call it only after `_width_error` has passed, since it shifts by
    ``exp_bits``. The top comes from `_top_of_range`. The smallest value, the
    smallest gap between values, and the lowest smallest value that the
    carrier's cast handles correctly all depend on the subnormals mode.
    """
    # the codes P3109 reserves: +infinity outside the finite domain, and, in
    # an unsigned format, the NaN above it (`make_binaryK_params`)
    reserved = (saturation is not SaturationMode.SAT_FINITE) + (0 if is_signed else 1)
    top_field, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved, carrier)
    min_exp = 1 - bias  # the smallest normal's exponent
    if subnormals is SubnormalsMode.SUBNORMALS:
        # the subnormals are evenly spaced, so their spacing is both the
        # smallest value and the smallest gap in the format
        bottom_exp = step_exp = min_exp - man_bits
    elif subnormals is SubnormalsMode.NORMALS:
        bottom_exp, step_exp = min_exp, min_exp - man_bits
    else:
        # EXTENDED_NORMALS: one more binade of normals below the others,
        # without its zero-mantissa code, which still encodes zero, so the
        # smallest value is one step above that binade's foot. With
        # man_bits == 0 that step is the whole binade, and the smallest value
        # is the same as under NORMALS.
        bottom_exp = min_exp - 1 if man_bits else min_exp
        step_exp = bottom_exp - man_bits
    # How low the smallest value may be differs between the three modes,
    # depending on whether the cast reads the exponent field at the bottom of
    # the range. In binary32's numbers, with binary64's in parentheses:
    #
    # `SUBNORMALS` works out the spacing from the input's exponent field,
    # which all of the carrier's subnormals share, so it cannot tell inputs
    # below 2**-126 (2**-1022) apart, and needs its smallest value at least a
    # binade above that. `NORMALS` and `EXTENDED_NORMALS` find the bottom by
    # comparing magnitudes, which reads no field, so they are exact down to
    # 2**-126, but no lower, since below that the smallest value would leave
    # the carrier's normal range. Two kinds of format stop a binade short of
    # 2**-126. Both fail at half the smallest value, the one point other than
    # the smallest value itself that the comparison needs:
    #
    #   man_bits == 0   the rounding step before the comparison does read a
    #                   field. With no mantissa it rounds between binades, and
    #                   with the smallest value at 2**-126, half of it,
    #                   2**-127, has exponent field 0. The rounding step sees
    #                   a tie between two binades and carries it up to the
    #                   smallest value before the comparison can send it to
    #                   zero.
    #   EXTENDED, full  at the carrier's full precision (P = 24, 53), half the
    #   precision       smallest value, 2**-127 * (1 + 2**-23), needs a step
    #                   of 2**-150 (2**-1075). The carrier holds it as
    #                   2**-127, and round-to-nearest-away then takes an input
    #                   of 2**-127 up to the smallest value.
    #
    # `dev/benchmarks/sweeps/cast_all_modes_sweep.cu` found both in binary32
    # by comparing every rounding mode with binary64 over all 2**32 inputs
    # (2**-127 is a binary64 normal, so binary64 gets these inputs right).
    # The same limits apply to binary64 at its own exponents.
    if subnormals is SubnormalsMode.SUBNORMALS:
        min_bottom, bottom_why = carrier.min_simulable_exp, _bottom_by_field(carrier)
    elif man_bits == 0:
        min_bottom, bottom_why = carrier.min_simulable_exp, _bottom_tie_by_field(carrier)
    elif subnormals is SubnormalsMode.EXTENDED_NORMALS:
        # the carrier must hold the last bit of half the smallest value,
        # 2**(bottom_exp - 1 - man_bits)
        min_bottom = max(carrier.min_normal_exp, man_bits + 1 + carrier.min_subnormal_exp)
        bottom_why = _bottom_half_unheld(carrier)
    else:
        min_bottom, bottom_why = carrier.min_normal_exp, _bottom_by_field(carrier)
    if bottom_exp < carrier.min_normal_exp:
        # the smallest value itself is below the carrier's normals, which is
        # the more useful explanation
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
    """Return the findings of ``carrier`` on a binaryK format, as ``(error, warning)``.

    It takes the format's parameters as plain values, not a :class:`BinaryK`,
    and caches its result on all of them. The widths are checked first,
    because computing the range shifts by the exponent width.
    """
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    label = _binaryK_label(K, P, bias, is_signed, subnormals)

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        error = _width_error(label, exp_bits, man_bits, prng_bits, c) or _exponent_error(
            label, exp_bits, bias
        )
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
    """Return the findings on storing a binaryK format's results in
    ``storage``, as ``(error, warning)``.

    Cached on the parameters, like `_binaryK_findings`. A format that the
    carrier check rejects returns ``(None, None)``: its range cannot be
    computed, and the carrier check's error is the one to report.
    """
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    label = _binaryK_label(K, P, bias, is_signed, subnormals)
    # Compute the range against binary64, the wider carrier. What a storage
    # rule reads (the top, the smallest gap, the bottom) belongs to the format
    # and comes out the same in either carrier for a format that binary32
    # holds, and a format that only binary64 holds can be stored too, under
    # `carrier=torch.float64`.
    if _width_error(label, exp_bits, man_bits, 0, _BINARY64) is not None:
        # the carrier check reports this, and it runs before any call gets
        # here; the range of such a format cannot be computed
        return None, None
    ext = _binaryK_extent(
        label, man_bits, exp_bits, bias, is_signed, saturation, subnormals, _BINARY64
    )
    if ext.top_field < 1 or ext.top_exp < _BINARY64.min_normal_exp:
        return None, None  # likewise: no finite normal value, which the carrier check reports
    return _storage_findings(ext, storage, elementwise, saturation)
