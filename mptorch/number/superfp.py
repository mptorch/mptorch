"""Supernormal float formats, and checks that a carrier and a storage dtype can hold one.

:class:`SuperFP` describes the format. Its checks work like binaryK's and use
the same shared code (`_carrier`, `_storage`). The part specific to superfp
is its layout of regions: binades with a full mantissa at the top of the
range, and below them powers of two, the supernormals
(`_superfp_regions_error`, `_superfp_extent`).
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
from .modes import Domain, SaturationMode


@dataclass(frozen=True)
class SuperFP(FloatFormat):
    """A supernormal floating-point format, as implemented by ``common/cast_superfp.h``.

    The top ``normal_binades`` binades of the exponent range have a full
    ``man_bits``-bit mantissa, as in an ordinary float. The other binades
    would have used ``n = (2**exp_bits - normal_binades) * 2**man_bits``
    codes for their mantissas; a superfp uses those codes instead for zero
    and ``n - 1`` successive powers of two below the normal binades (the
    *supernormals*). A value below the smallest supernormal rounds to zero or
    to the smallest supernormal, as the rounding mode decides. The format
    thus gives up precision at the bottom of its range in exchange for a much
    wider range at the same bit width.

    The class differs from :class:`BinaryK` on purpose:

    * ``bias`` is required, because there is no standard default to derive.
    * Supernormals exist only at the bottom of the range, so a single
      ``normal_binades`` describes the layout.
    * There is no ``subnormals`` mode, because the format has no subnormals.

    Its zero, infinities and NaN are encoded as IEEE P3109 encodes them in a
    :class:`BinaryK`:

    * The format has a single zero, unsigned, at the all-zero code. No cast
      returns ``-0.0``.
    * In a signed format the NaN is the code with only the sign bit set
      (which would otherwise encode ``-0.0``). In an unsigned format it is the
      top code.
    * The largest finite value is the highest code not reserved for infinity
      or NaN. An unsigned format's largest value is therefore one code below
      that of the signed format with the same fields.

    Like :class:`BinaryK`, the object is frozen, and therefore hashable, so
    that the code turning it into kernel arguments can cache its result keyed
    on the format.

    Args:
        man_bits (int): stored mantissa bits in the normal binades.
        exp_bits (int): width of the exponent field, at least 1.
        normal_binades (int): number of binades at the top of the range that
            keep a full mantissa: from 1 to ``2**exp_bits - 1``, so that some
            codes remain for supernormals.
        bias (int): exponent bias.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        prng_bits (int): number of random bits drawn by the stochastic
            rounding modes; ignored by the other modes. Default: 0.
        saturation (SaturationMode): what happens on overflow. Default:
            ``SaturationMode.SAT_NONE``.
        domain (Domain, optional): whether the format has infinities, as for
            :class:`BinaryK`. Default: ``None``, which follows from
            ``saturation``.

    Raises:
        ValueError: for ``man_bits < 0``, ``exp_bits < 1``, a
            ``normal_binades`` outside ``1 .. 2**exp_bits - 1``, normal
            binades that hold no finite value (their codes all go to the
            infinity and the NaN), or the finite domain with a saturation mode
            other than ``SAT_FINITE``; and, as for :class:`BinaryK`, for a
            format that neither carrier can simulate. Each call checks the
            format against its own carrier (see :class:`FormatRangeWarning`).
        TypeError: as for :class:`BinaryK`.

    Example::

        >>> from mptorch import SuperFP
        >>> from mptorch.quant import Quant
        >>> import torch
        >>> f = SuperFP(man_bits=3, exp_bits=4, normal_binades=2, bias=7)
        >>> x = torch.tensor([300.0, 3.0, 0.001])  # normal, supernormal, supernormal
        >>> Quant(f)(x).tolist()
        [288.0, 4.0, 0.0009765625]
    """

    man_bits: int
    exp_bits: int
    normal_binades: int
    bias: int
    _: KW_ONLY
    is_signed: bool = True
    prng_bits: int = 0
    saturation: SaturationMode = SaturationMode.SAT_NONE
    domain: Domain | None = None

    def __post_init__(self) -> None:
        saturation, domain = _resolve_domain(self.saturation, self.domain, "SuperFP")
        object.__setattr__(self, "saturation", saturation)
        object.__setattr__(self, "domain", domain)
        if self.man_bits < 0 or self.exp_bits < 1:
            raise ValueError(
                f"SuperFP needs man_bits >= 0 and exp_bits >= 1, "
                f"got man_bits={self.man_bits}, exp_bits={self.exp_bits}"
            )
        if self.normal_binades < 1:
            raise ValueError(f"SuperFP needs normal_binades >= 1, got {self.normal_binades}")
        if self.prng_bits < 0:
            raise ValueError(f"SuperFP prng_bits must be non-negative, got {self.prng_bits}")

        # Raise only for what no carrier can do; each call checks the format
        # against its own carrier (`check_superfp_carrier`)
        check_superfp(
            self.man_bits,
            self.exp_bits,
            self.normal_binades,
            self.bias,
            self.is_signed,
            self.saturation,
            self.prng_bits,
            carrier=torch.float64,
            warn=False,
        )

    @property
    def max_finite(self) -> float:
        """The largest finite value, counted in code points as P3109 counts it.

        The highest codes of the top binade can be reserved: in an unsigned
        format the top code is the NaN, and in the extended domain the next
        code down (the top code, if the format is signed) is the infinity.
        With no mantissa bits, the reserved codes move the largest finite
        value one or two binades down. It is the value that a saturating mode
        clamps to.

        Example::

            >>> from mptorch import SaturationMode, SuperFP
            >>> SuperFP(3, 4, 2, 7).max_finite
            448.0
            >>> SuperFP(3, 4, 2, 7, saturation=SaturationMode.SAT_FINITE).max_finite
            480.0
            >>> SuperFP(3, 4, 2, 7, is_signed=False).max_finite
            416.0
        """
        reserved = _superfp_reserved(self.is_signed, self.saturation)
        return _largest_finite(self.exp_bits, self.man_bits, self.bias, reserved)


def check_superfp(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.SAT_NONE,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype = torch.float32,
    warn: bool = True,
    stacklevel: int = 3,
) -> None:
    """Check whether a carrier can hold a superfp format.

    Behaves as :func:`check_binaryK`, and takes its arguments in the same
    order.

    Args:
        man_bits (int): stored mantissa bits.
        exp_bits (int): width of the exponent field.
        normal_binades (int): number of binades at the top of the range that
            keep a full mantissa; the codes of the others encode the
            supernormal powers of two.
        bias (int): the exponent bias.
        is_signed (bool): whether the format has a sign bit. An unsigned
            format spends its top code on NaN. Default: ``True``.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.SAT_NONE``.
        prng_bits (int): random bits drawn for stochastic rounding. Default: 0.
        carrier (torch.dtype): ``torch.float32`` or ``torch.float64``.
            Default: ``torch.float32``.
        warn (bool): whether to issue the warning. Default: ``True``.
        stacklevel (int): passed to ``warnings.warn``, as for
            :func:`check_binaryK`. Default: 3.

    Raises:
        ValueError: as :func:`check_binaryK`, with a negative ``man_bits`` or
            ``exp_bits`` in place of its bounds on ``P``, and also for a
            ``normal_binades`` outside ``1 .. 2**exp_bits - 1``, which leaves
            the format no normal binade or no supernormal codes, and for
            normal binades whose codes all go to the infinity and the NaN.
        TypeError: if ``is_signed`` is not a ``bool``.
    """
    _check_is_signed(is_signed, "check_superfp")
    error, warning = _superfp_findings(
        man_bits, exp_bits, normal_binades, bias, is_signed, saturation, prng_bits, carrier
    )
    if error is not None:
        raise ValueError(error)
    if warn and warning is not None:
        warnings.warn(warning, FormatRangeWarning, stacklevel=stacklevel)


def check_superfp_carrier(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.SAT_NONE,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype,
) -> None:
    """The superfp counterpart of :func:`check_binaryK_carrier`.

    The arguments are those of :func:`check_superfp` without ``warn`` and
    ``stacklevel``, and ``carrier`` is required.

    Raises:
        ValueError: as :func:`check_superfp`.
        TypeError: as :func:`check_superfp`.
    """
    _check_is_signed(is_signed, "check_superfp_carrier")
    _report_per_call(
        *_superfp_findings(
            man_bits, exp_bits, normal_binades, bias, is_signed, saturation, prng_bits, carrier
        )
    )


def check_superfp_storage(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.SAT_NONE,
    *,
    storage: torch.dtype,
    elementwise: bool = False,
) -> None:
    """The superfp counterpart of :func:`check_binaryK_storage`.

    It takes the format's parameters as :func:`check_superfp` does, and the
    same ``storage`` and ``elementwise`` keywords as
    :func:`check_binaryK_storage`.

    Raises:
        ValueError: as :func:`check_binaryK_storage`.
        TypeError: as :func:`check_superfp`.
    """
    _check_is_signed(is_signed, "check_superfp_storage")
    _report_per_call(
        *_superfp_storage_findings(
            man_bits, exp_bits, normal_binades, bias, is_signed, saturation, storage, elementwise
        )
    )


# --- how the checks reach their findings -------------------------------------


def _check_is_signed(is_signed: object, name: str) -> None:
    """Raise a ``TypeError`` for an ``is_signed`` that is not a ``bool``.

    The public checks take ``is_signed`` before ``saturation``, as
    :func:`check_binaryK` does, so a call that passes ``saturation`` in its
    place would otherwise check a signed format in the default mode.
    """
    if not isinstance(is_signed, bool):
        raise TypeError(
            f"{name}: is_signed must be a bool, got {is_signed!r}. The arguments after bias "
            "are is_signed, saturation and prng_bits, in check_binaryK's order"
        )


def _superfp_reserved(is_signed: bool, saturation: SaturationMode) -> int:
    """Return the number of codes at the top of the range that are not finite values.

    They are P3109's, as for a binaryK (`make_superfp_params`): the infinity
    outside the finite domain, and in an unsigned format the NaN above it. A
    signed format's NaN is the code that would hold -0.0, which takes no
    magnitude code.
    """
    return (saturation is not SaturationMode.SAT_FINITE) + (0 if is_signed else 1)


def _superfp_label(
    man_bits: int, exp_bits: int, normal_binades: int, bias: int, is_signed: bool = True
) -> str:
    """The format written as a constructor call, for messages, leaving out
    ``is_signed`` when it is ``True``."""
    return (
        f"SuperFP(man_bits={man_bits}, exp_bits={exp_bits}, "
        f"normal_binades={normal_binades}, bias={bias}"
        + ("" if is_signed else ", is_signed=False")
        + ")"
    )


def _superfp_regions_error(label: str, exp_bits: int, normal_binades: int) -> str | None:
    """Return an error for a ``normal_binades`` outside ``1 .. 2**exp_bits - 1``,
    the superfp layouts that the cast cannot work with, or ``None``."""
    binades = 1 << exp_bits
    if normal_binades < 1:
        # `SuperFP` rejects this in `__post_init__`, but the wrappers in
        # `mptorch.quant.ops` that take plain integers come here unchecked.
        # With no normal binade the top binade holds supernormals, powers of
        # two, but the cast still saturates onto that binade's largest code
        # with a full mantissa, which is not a value of the format.
        return (
            f"{label} has no normal binade: normal_binades must be at least 1, because the "
            "format's largest finite value, which the cast saturates onto, lies in a normal binade"
        )
    if normal_binades >= binades:
        # No codes are left for supernormals, which puts the bottom of the
        # supernormal region one binade *above* the bottom of the normal
        # region. The cast tests for the supernormal region and then for
        # underflow, so the format's lowest normal binade would round to zero.
        return (
            f"{label} leaves no supernormal codes: normal_binades must be at most "
            f"2**exp_bits - 1 = {binades - 1}, or the cast rounds the format's lowest "
            f"normal binade to zero"
        )
    return None


def _superfp_top_error(
    label: str,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    carrier: _Carrier,
) -> str | None:
    """Return an error for a format whose normal binades hold no finite value, or ``None``.

    Call it only after `_width_error` and `_superfp_regions_error` have
    passed. A normal binade has ``2**man_bits`` codes, so with no mantissa
    bits, or one, the codes reserved at the top of the range
    (`_superfp_reserved`) can be every code the normal binades have. The
    largest finite value would then be a supernormal, but the cast saturates
    only in the normal region, and its supernormal region rounds up to the
    bottom of the normal region, which would be a reserved code.
    """
    reserved = _superfp_reserved(is_signed, saturation)
    top_field, _, _ = _top_of_range(exp_bits, man_bits, bias, reserved, carrier)
    if top_field >= (1 << exp_bits) - normal_binades:
        return None
    special = {
        (True, False): "the infinity",
        (False, True): "the NaN",
        (True, True): "the infinity and the NaN",
    }[(saturation is not SaturationMode.SAT_FINITE, not is_signed)]
    codes = normal_binades << man_bits
    owner = "its only normal code goes" if codes == 1 else f"its {codes} normal codes all go"
    return (
        f"{label} has no finite normal value: {owner} to {special}, and the cast needs its "
        "largest finite value in a normal binade. Give it more normal binades or mantissa bits"
    )


def _superfp_extent(
    label: str,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    carrier: _Carrier,
) -> _Extent:
    """Compute a superfp format's range as an `_Extent`.

    Call it only after `_width_error`, `_superfp_regions_error` and
    `_superfp_top_error` have passed. The normal region is the top
    ``normal_binades`` binades, which have a full mantissa. Below it, the
    ``(binades - normal_binades) * 2**man_bits`` codes that the other binades
    would have used for mantissas encode zero (one code) and successive
    powers of two (all the others): the supernormals.
    """
    binades = 1 << exp_bits
    reserved = _superfp_reserved(is_signed, saturation)
    top_field, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved, carrier)
    normal_cutoff = binades - bias - normal_binades
    # the exponent of the smallest supernormal (`superfp_region_cutoffs`)
    bottom_exp = normal_cutoff - (binades - normal_binades) * (1 << man_bits) + 1
    step_exp = min(bottom_exp, normal_cutoff - man_bits)
    return _Extent(
        label,
        exp_bits,
        man_bits,
        top_field,
        top_exp,
        top_code,
        bottom_exp,
        step_exp,
        carrier.min_simulable_exp,  # the cast reads the exponent field for supernormals
        _bottom_by_field(carrier),
        normal_cutoff,
        "SUPERNORMALS",
    )


@lru_cache(maxsize=512)
def _superfp_findings(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    prng_bits: int,
    carrier: torch.dtype,
) -> tuple[str | None, str | None]:
    """Return the findings of ``carrier`` on a superfp format, as ``(error, warning)``.

    The superfp counterpart of `_binaryK_findings`, with the region-layout
    checks added after the width checks.
    """
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias, is_signed)

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        error = (
            _width_error(label, exp_bits, man_bits, prng_bits, c)
            or _exponent_error(label, exp_bits, bias)
            or _superfp_regions_error(label, exp_bits, normal_binades)
            or _superfp_top_error(
                label, man_bits, exp_bits, normal_binades, bias, is_signed, saturation, c
            )
        )
        if error is not None:
            return error, None
        ext = _superfp_extent(
            label, man_bits, exp_bits, normal_binades, bias, is_signed, saturation, c
        )
        return _range_findings(ext, c)

    return _pointing_wider(find, carrier)


@lru_cache(maxsize=256)
def _superfp_storage_findings(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    storage: torch.dtype,
    elementwise: bool,
) -> tuple[str | None, str | None]:
    """Return the findings on storing a superfp format's results in
    ``storage``, as ``(error, warning)``.

    The superfp counterpart of `_binaryK_storage_findings`.
    """
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias, is_signed)
    if (
        _width_error(label, exp_bits, man_bits, 0, _BINARY64) is not None
        or _superfp_regions_error(label, exp_bits, normal_binades) is not None
        or _superfp_top_error(
            label, man_bits, exp_bits, normal_binades, bias, is_signed, saturation, _BINARY64
        )
        is not None
    ):
        return None, None  # as for binaryK: the carrier check reports this
    ext = _superfp_extent(
        label, man_bits, exp_bits, normal_binades, bias, is_signed, saturation, _BINARY64
    )
    if ext.top_field < 1 or ext.top_exp < _BINARY64.min_normal_exp:
        return None, None
    return _storage_findings(ext, storage, elementwise, saturation)
