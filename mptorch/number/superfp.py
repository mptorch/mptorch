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

    The format reserves no code for NaN, so being signed or not does not
    change its largest value. Its single zero is unsigned, as in a binaryK.
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
            ``normal_binades`` outside ``1 .. 2**exp_bits - 1``, or the finite
            domain with a saturation mode other than ``SAT_FINITE``; and, as
            for :class:`BinaryK`, for a format that neither carrier can
            simulate. Each call checks the format against its own carrier
            (see :class:`FormatRangeWarning`).
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
            self.saturation,
            self.prng_bits,
            carrier=torch.float64,
            warn=False,
        )

    @property
    def max_finite(self) -> float:
        """The largest finite value. It is the top code of the top binade,
        or the code below it when the top code is an infinity (outside the
        finite domain). The format has no NaN code.

        Example::

            >>> from mptorch import SaturationMode, SuperFP
            >>> SuperFP(3, 4, 2, 7).max_finite
            448.0
            >>> SuperFP(3, 4, 2, 7, saturation=SaturationMode.SAT_FINITE).max_finite
            480.0
        """
        reserved = int(self.saturation is not SaturationMode.SAT_FINITE)  # as in `_superfp_extent`
        return _largest_finite(self.exp_bits, self.man_bits, self.bias, reserved)


def check_superfp(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode = SaturationMode.SAT_NONE,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype = torch.float32,
    warn: bool = True,
    stacklevel: int = 3,
) -> None:
    """Check whether a carrier can hold a superfp format.

    Behaves as :func:`check_binaryK`. It takes no ``is_signed``: a superfp
    reserves no code for NaN, signed or not, so signedness does not change
    its range.

    Args:
        man_bits (int): stored mantissa bits.
        exp_bits (int): width of the exponent field.
        normal_binades (int): number of binades at the top of the range that
            keep a full mantissa; the codes of the others encode the
            supernormal powers of two.
        bias (int): the exponent bias.
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
            the format no normal binade or no supernormal codes.
    """
    error, warning = _superfp_findings(
        man_bits, exp_bits, normal_binades, bias, saturation, prng_bits, carrier
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
    """
    _report_per_call(
        *_superfp_findings(man_bits, exp_bits, normal_binades, bias, saturation, prng_bits, carrier)
    )


def check_superfp_storage(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
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
    """
    _report_per_call(
        *_superfp_storage_findings(
            man_bits, exp_bits, normal_binades, bias, saturation, storage, elementwise
        )
    )


# --- how the checks reach their findings -------------------------------------


def _superfp_label(man_bits: int, exp_bits: int, normal_binades: int, bias: int) -> str:
    """The format written as a constructor call, for messages."""
    return (
        f"SuperFP(man_bits={man_bits}, exp_bits={exp_bits}, "
        f"normal_binades={normal_binades}, bias={bias})"
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


def _superfp_extent(
    label: str,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode,
    carrier: _Carrier,
) -> _Extent:
    """Compute a superfp format's range as an `_Extent`.

    Call it only after `_width_error` and `_superfp_regions_error` have
    passed. The normal region is the top ``normal_binades`` binades, which
    have a full mantissa. Below it, the
    ``(binades - normal_binades) * 2**man_bits`` codes that the other binades
    would have used for mantissas encode zero (one code) and successive
    powers of two (all the others): the supernormals.
    """
    binades = 1 << exp_bits
    # superfp reserves no code for NaN, signed or not (`make_superfp_params`)
    reserved = saturation is not SaturationMode.SAT_FINITE
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
    saturation: SaturationMode,
    prng_bits: int,
    carrier: torch.dtype,
) -> tuple[str | None, str | None]:
    """Return the findings of ``carrier`` on a superfp format, as ``(error, warning)``.

    The superfp counterpart of `_binaryK_findings`, with the region-layout
    check added after the width checks.
    """
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias)

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        error = (
            _width_error(label, exp_bits, man_bits, prng_bits, c)
            or _exponent_error(label, exp_bits, bias)
            or _superfp_regions_error(label, exp_bits, normal_binades)
        )
        if error is not None:
            return error, None
        ext = _superfp_extent(label, man_bits, exp_bits, normal_binades, bias, saturation, c)
        return _range_findings(ext, c)

    return _pointing_wider(find, carrier)


@lru_cache(maxsize=256)
def _superfp_storage_findings(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode,
    storage: torch.dtype,
    elementwise: bool,
) -> tuple[str | None, str | None]:
    """Return the findings on storing a superfp format's results in
    ``storage``, as ``(error, warning)``.

    The superfp counterpart of `_binaryK_storage_findings`.
    """
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias)
    if (
        _width_error(label, exp_bits, man_bits, 0, _BINARY64) is not None
        or _superfp_regions_error(label, exp_bits, normal_binades) is not None
    ):
        return None, None  # as for binaryK: the carrier check reports this
    ext = _superfp_extent(label, man_bits, exp_bits, normal_binades, bias, saturation, _BINARY64)
    if ext.top_field < 1 or ext.top_exp < _BINARY64.min_normal_exp:
        return None, None
    return _storage_findings(ext, storage, elementwise, saturation)
