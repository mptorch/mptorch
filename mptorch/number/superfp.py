"""Supernormal floats, and the checks that hold one to a carrier and a dtype.

:class:`SuperFP` is the format, and its checks are binaryK's, on the same
machinery (`binaryk`). What is superfp's own is the region layout:
full-mantissa binades at the top and supernormal powers of two below them
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
    """A supernormal float, as ``common/cast_superfp.h`` implements it.

    The top ``normal_binades`` binades of the exponent range keep a full
    ``man_bits``-bit mantissa. The remaining binades would have spent
    ``n = (2**exp_bits - normal_binades) * 2**man_bits`` codes on mantissas.
    Those codes instead encode the zero and ``n - 1`` further powers of two
    below the normal binades (the supernormals). Everything below those
    flushes to zero. The format therefore trades precision at the bottom of
    the range for dynamic range at the same width.

    The class is deliberately not symmetric with :class:`BinaryK`:

    * ``bias`` is required and has no default, because the cast has no rule
      to derive one.
    * There is a single ``normal_binades``, not one per end of the range.
    * There is no ``subnormals`` mode, because the format has no subnormals.

    The format spends no code point on a NaN, so signedness does not move the
    top of the range. Its one zero is unsigned, like binaryK's. The instance
    is frozen, and therefore hashable, because the code that resolves a
    format memoizes on its value, as for :class:`BinaryK`.

    Args:
        man_bits (int): stored mantissa bits in the normal region.
        exp_bits (int): exponent field width, at least 1.
        normal_binades (int): binades at the top of the range that keep their
            mantissa. At least 1 and at most ``2**exp_bits - 1``, so that at
            least one supernormal code remains.
        bias (int): exponent bias.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        prng_bits (int): width of the random mantissa drawn by
            :attr:`RoundMode.SR`. Default: 0.
        saturation (SaturationMode): overflow behaviour. Default:
            ``SaturationMode.SAT_NONE``.
        domain (Domain, optional): whether the top code is an infinity, as
            for :class:`BinaryK`. Default: ``None``, which follows from
            ``saturation``.

    Raises:
        ValueError: for the finite domain with a saturation mode other than
            ``SAT_FINITE``; for a format that neither carrier can simulate, as
            for :class:`BinaryK`; and for a ``normal_binades`` that leaves no
            supernormal codes. Each call checks the format against its own
            carrier (see :class:`FormatRangeWarning`).
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

        # Raise for what no carrier can do, and nothing more (`check_superfp`)
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
        """The largest finite value: the top binade's top code, less the
        infinity outside the finite domain (the format has no NaN).

        Example::

            >>> from mptorch import SaturationMode, SuperFP
            >>> SuperFP(3, 4, 2, 7).max_finite
            448.0
            >>> SuperFP(3, 4, 2, 7, saturation=SaturationMode.SAT_FINITE).max_finite
            480.0
        """
        reserved = int(self.saturation is not SaturationMode.SAT_FINITE)  # `_superfp_extent`
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
    """Hold a superfp format against what a carrier can hold.

    The same contract as :func:`check_binaryK`. There is no ``is_signed``:
    superfp spends no code point on a NaN either way, so signedness does not
    move the top of the range.

    Args:
        man_bits (int): stored mantissa bits.
        exp_bits (int): exponent field width.
        normal_binades (int): binades at the top of the range that keep their
            mantissa; the rest encode supernormal powers of two.
        bias (int): the exponent bias.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.SAT_NONE``.
        prng_bits (int): random bits drawn for stochastic rounding. Default: 0.
        carrier (torch.dtype): ``torch.float32`` or ``torch.float64``.
            Default: ``torch.float32``.
        warn (bool): whether to issue the warning. Default: ``True``.
        stacklevel (int): how far above this function the caller's own code
            is. Default: 3.

    Raises:
        ValueError: as :func:`check_binaryK`, and also for a
            ``normal_binades`` that leaves no supernormal codes.
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
    """:func:`check_binaryK_carrier` for a superfp format.

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
    """:func:`check_binaryK_storage` for a superfp format.

    The same contract, with the format spelled as :func:`check_superfp`
    spells it and the same ``storage`` and ``elementwise`` keywords.

    Raises:
        ValueError: as :func:`check_binaryK_storage`.
    """
    _report_per_call(
        *_superfp_storage_findings(
            man_bits, exp_bits, normal_binades, bias, saturation, storage, elementwise
        )
    )


# --- what the checks find ----------------------------------------------------


def _superfp_label(man_bits: int, exp_bits: int, normal_binades: int, bias: int) -> str:
    """The format as its constructor call, for messages."""
    return (
        f"SuperFP(man_bits={man_bits}, exp_bits={exp_bits}, "
        f"normal_binades={normal_binades}, bias={bias})"
    )


def _superfp_regions_error(label: str, exp_bits: int, normal_binades: int) -> str | None:
    """The one superfp layout no range can be read from, or ``None``."""
    binades = 1 << exp_bits
    if normal_binades >= binades:
        # no supernormal codes left, which puts the supernormal floor one
        # *above* the normal one; the cast tests supernormal-then-underflow,
        # so the format's lowest normal binade would flush to zero
        return (
            f"{label} leaves no supernormal codes: normal_binades must be at most "
            f"2**exp_bits - 1 = {binades - 1}, or the cast flushes the format's "
            f"lowest normal binade to zero"
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
    """A superfp format's range, as an `_Extent`.

    Call it only once `_width_error` and `_superfp_regions_error` have passed.
    The normal region holds the top ``normal_binades`` binades with a full
    mantissa; below it the ``(binades - normal_binades) * 2**man_bits`` codes
    the other binades would have spent on mantissas encode the zero and, all
    but that one, further powers of two, the supernormals, and everything
    below those flushes.
    """
    binades = 1 << exp_bits
    # superfp spends no code on a NaN, signed or not (`make_superfp_params`)
    reserved = saturation is not SaturationMode.SAT_FINITE
    top_field, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved, carrier)
    normal_cutoff = binades - bias - normal_binades
    # the smallest supernormal, `superfp_region_cutoffs`
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
        carrier.min_simulable_exp,  # the supernormal arm reads the exponent field
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
    """What ``carrier`` says about a superfp format, as ``(error, warning)``.

    The superfp twin of `_binaryK_findings`, with the region-layout check
    added to the width checks.
    """
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias)

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        error = _width_error(label, exp_bits, man_bits, prng_bits, c) or _superfp_regions_error(
            label, exp_bits, normal_binades
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
    """What storing a superfp result in ``storage`` says, as ``(error, warning)``.

    The superfp twin of `_binaryK_storage_findings`.
    """
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias)
    if (
        _width_error(label, exp_bits, man_bits, 0, _BINARY64) is not None
        or _superfp_regions_error(label, exp_bits, normal_binades) is not None
    ):
        return None, None  # as for binaryK: the carrier check has said so already
    ext = _superfp_extent(label, man_bits, exp_bits, normal_binades, bias, saturation, _BINARY64)
    if ext.top_field < 1 or ext.top_exp < _BINARY64.min_normal_exp:
        return None, None
    return _storage_findings(ext, storage, elementwise, saturation)
