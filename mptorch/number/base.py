"""The base classes and the warning helpers that every format module uses.

`Number` and `FloatFormat` are the base classes of the format types.
`_resolve_domain` records a `Domain` in the saturation mode of a `BinaryK` or
`SuperFP`. `FormatRangeWarning` is the warning issued when a format's range
exceeds the carrier it is rounded in or the dtype its result is stored in, and
`_report_per_call` raises an error or issues that warning on behalf of a call,
pointing at the user's line that made the call.

Each family module defines one format type, a `Number` subclass that holds
exactly the parameters its cast reads. A format object lets a caller name a
format once, as `BinaryK(8, 4)`, instead of repeating its five or six kernel
arguments wherever a GEMM, a quantizer or a layer takes a format. It also lets
the formats of one operation (a GEMM's multiply and accumulate formats, say)
differ in every parameter the kernel lets them differ in, such as the
saturation mode.

The format types are frozen dataclasses that compute their derived values
(such as a default bias) in `__post_init__`. Being frozen makes them hashable,
so the code that turns a format into kernel arguments can cache its result
keyed on the format. Doing that work on every call would cost 2.3-6.4 us, and
avoiding it is why `mptorch.quant.gemm`'s factories do it once per layer.
Because a frozen object cannot change after it is built, its hash always
agrees with the values derived from it.
"""

import os
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache

from .modes import Domain, SaturationMode


@dataclass(frozen=True)
class Number:
    """Base class of every simulated number format.

    Its direct subclasses are :class:`FloatFormat` (floating-point formats
    whose elements are rounded one by one), :class:`FixedPoint` (integers
    times a fixed step; supported by the elementwise quantizers and as a block
    format's element, but not by the GEMMs) and :class:`BlockFormat` (blocks
    of elements that share a scale, such as OCP MX, NVFP4 and block floating
    point). Other families, such as logarithmic or posit formats, would also
    derive from it, but each needs a kernel first, so none is declared ahead
    of time.

    Example::

        >>> from mptorch import BinaryK, Number
        >>> isinstance(BinaryK(8, 4), Number)
        True
    """


@dataclass(frozen=True)
class FloatFormat(Number):
    """Base class for the two floating-point families the kernels implement.

    :class:`BinaryK` and :class:`SuperFP` are its subclasses. A quantizer or
    GEMM factory that takes a ``FloatFormat`` accepts either.

    Example::

        >>> from mptorch import FloatFormat, SuperFP
        >>> isinstance(SuperFP(3, 4, 2, 7), FloatFormat)
        True
    """


def _resolve_domain(
    saturation: SaturationMode, domain: Domain | None, label: str
) -> tuple[SaturationMode, Domain]:
    """Return the saturation mode and the domain that a format object stores.

    Each combination is stored one way only, so that two spellings of the
    same format compare equal (and share a cache entry): ``SAT_FINITE`` with
    the extended domain is stored as ``SAT_FINITE_EXTENDED``, and a ``None``
    domain is derived from the mode. Raises ``ValueError`` for the finite
    domain with any mode but ``SAT_FINITE``, and ``TypeError`` for an argument
    of the wrong type.
    """
    if not isinstance(saturation, SaturationMode):
        raise TypeError(f"{label}: saturation must be a SaturationMode, got {saturation!r}")
    finite = saturation is SaturationMode.SAT_FINITE
    if domain is None:
        return saturation, Domain.FINITE if finite else Domain.EXTENDED
    if not isinstance(domain, Domain):
        raise TypeError(f"{label}: domain must be a Domain or None, got {domain!r}")
    if domain is Domain.FINITE:
        if not finite:
            raise ValueError(
                f"{label}: the finite domain allows only SaturationMode.SAT_FINITE "
                f"(P3109's SatFinite), got {saturation}"
            )
        return saturation, domain
    if finite:
        return SaturationMode.SAT_FINITE_EXTENDED, domain
    return saturation, domain


# --- errors and warnings raised from inside a call ---------------------------


class FormatRangeWarning(UserWarning):
    """Warns that part of a format's range does not fit the float it is rounded or stored in.

    The cast to such a format still rounds correctly over the part of the
    format's range that the *carrier* (the binary float the cast computes in)
    can represent, but some of the format's values are never returned. The
    carrier is binary64 for a float64 tensor and binary32 for float32, float16
    and bfloat16 tensors, unless the call asks for binary64 with
    ``carrier=torch.float64``.

    The quantizers and GEMMs of :mod:`mptorch.quant` issue this warning, on
    each call, when a format does one of three things:

    * it has values above the carrier's largest finite value;
    * it has two neighbouring values closer together than the carrier's
      smallest subnormal (:math:`2^{-149}` in binary32, :math:`2^{-1074}` in
      binary64);
    * its smallest positive value is below the lowest value the carrier's
      cast handles correctly.

    That lower limit depends on how the cast locates the bottom of the
    format's range:

    * :math:`2^{-125}` in binary32 (:math:`2^{-1021}` in binary64) for a
      :class:`BinaryK` with ``SubnormalsMode.SUBNORMALS`` and for a
      :class:`SuperFP`. Their casts work out where an input falls among the
      format's subnormal or supernormal values from the input's exponent
      field. All of the carrier's subnormals share one exponent field, so a
      format that reached lower would round its lowest binade with the wrong
      spacing.
    * :math:`2^{-126}` (:math:`2^{-1022}`), the carrier's smallest normal, for
      a :class:`BinaryK` with ``SubnormalsMode.NORMALS`` or
      ``SubnormalsMode.EXTENDED_NORMALS``. Their casts find the bottom of the
      range by comparing magnitudes, without reading the exponent field. Two
      cases need the higher limit of the first item: ``P = 1`` in either mode,
      and ``EXTENDED_NORMALS`` at the carrier's full precision (``P = 24`` in
      binary32, ``P = 53`` in binary64).

    :doc:`/concepts` ("What the carrier can hold") derives both limits.
    :class:`BinaryK` and :class:`SuperFP` do not warn when they are built,
    because the carrier is chosen by each call, not by the format; they raise
    only for formats that neither carrier can handle.

    The same warning is issued when the result is stored in a dtype narrower
    than the carrier (float16 or bfloat16 under binary32; those and float32
    under binary64) and some of the format's results are not values of that
    dtype, for example because they exceed its largest finite value or are
    spaced more finely than its smallest subnormal. The cast rounds in the
    carrier, and storing the result in the dtype rounds it a second time. A
    result above the dtype's largest finite value is stored as an infinity,
    whatever the format's saturation mode.

    This is a warning, not an error, because such a format is still useful: a
    binaryK with a very wide range, for example, can be used to study
    precision alone, and then its unreachable extremes do not matter. Each
    warning is attributed to the line of user code that made the call, so
    Python's default warning filter shows it once per line, not once per
    training step.

    Example::

        >>> import warnings
        >>> import torch
        >>> from mptorch import BinaryK, FormatRangeWarning
        >>> from mptorch.quant import Quant
        >>> q = Quant(BinaryK(16, 8, bias=255))  # smallest value 2**-261
        >>> warnings.simplefilter("ignore", FormatRangeWarning)
        >>> y = q(torch.ones(2))  # rounded in binary32, with the warning silenced
    """


@lru_cache(maxsize=1)
def _library_frames() -> tuple[str, ...]:
    """Return the directories of the mptorch and torch packages, for
    ``warnings.warn``'s ``skip_file_prefixes``.

    Each path ends in a separator. A per-call check runs inside an op, and
    the op can be reached from a wrapper function, a layer's math hook or an
    autograd backward pass, at varying stack depths and sometimes through
    torch's own frames (``Module.__call__``, ``Function.apply``). No fixed
    ``stacklevel`` can point at the user's code from there. Skipping every
    frame inside these two directories does: the warning is attributed to the
    first frame outside them, which is the line that made the call. Python's
    default filter then shows the warning once for that line, so a training
    loop reports it only once.
    """
    # mptorch's package directory, two levels up from this file (mptorch/number/base.py)
    roots = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.__file__:
        roots.append(os.path.dirname(os.path.abspath(torch_module.__file__)))
    return tuple(os.path.join(root, "") for root in roots)


def _warn_per_call(message: str) -> None:
    """Issue ``message`` as a ``FormatRangeWarning`` attributed to the caller's line.

    Dynamo (``torch.compile``) cannot trace this function: it does not trace
    ``warnings.warn``, a builtin, nor the ``os.path`` calls in
    ``_library_frames``, so tracing it would break the graph wherever a
    format warns (and make a ``fullgraph`` compile fail). The function is
    therefore marked so that Dynamo calls it once, at trace time, and treats
    its result as a constant. The attribute set below is the mark that
    ``torch.compiler.assume_constant_result`` sets. It is set directly
    because importing that function imports ``torch._dynamo``, which would
    add 0.76 s to ``import mptorch``. The mark is valid here because the
    function returns ``None`` and its one argument is a string that Dynamo
    knows as a constant. Marking the ``check_*`` functions instead would
    fail, because Dynamo may treat their integer arguments as symbolic.
    """
    warnings.warn(message, FormatRangeWarning, skip_file_prefixes=_library_frames())


_warn_per_call._dynamo_marked_constant = True  # ty: ignore[missing-attribute]


def _report_per_call(error: str | None, warning: str | None) -> None:
    """Raise ``ValueError(error)`` if there is an error; otherwise issue the warning, if any.

    The warning is attributed to the caller's line, as `_warn_per_call` does.
    """
    if error is not None:
        raise ValueError(error)
    if warning is not None:
        _warn_per_call(warning)
