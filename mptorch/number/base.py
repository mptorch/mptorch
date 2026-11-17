"""What every format module builds on.

`Number` and `FloatFormat` are the base classes of the format value types, and
`_resolve_domain` is how the two float families fold a `Domain` into their
saturation mode. `FormatRangeWarning` is what a format warns with when its
range outruns the carrier it is rounded in or the dtype its result is stored
in, and `_report_per_call` is how a call's checks raise or warn, naming the
caller's line.

The family modules define one value type per simulated format, each a
`Number`, carrying exactly what that format's cast reads and nothing else.
They exist so a caller can name a format once, `BinaryK(8, 4)`, instead of
spelling its five or six schema arguments at each of the places a GEMM, a
quantizer or a layer takes one, and so the two halves of a mixed-precision
configuration can differ where the kernel already lets them differ.

They are frozen and they resolve their derived constants in `__post_init__`,
because the thing that reads them memoizes on the value: re-deriving a
format per call costs 2.3-6.4 us, which is the whole reason
`mptorch.quant.gemm`'s factories resolve once per layer. Frozen also makes
them hashable, which is what the memo keys on, and means a format's hash
cannot drift from the constants derived at construction.
"""

import os
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache

from .modes import Domain, SaturationMode


@dataclass(frozen=True)
class Number:
    """Base class for every simulated number format.

    Its subclasses are :class:`FloatFormat` (the element-by-element float
    families), :class:`FixedPoint` (the multiples of a step that a word of
    bits holds; an elementwise quantizer so far, and no GEMM) and
    :class:`BlockFormat` (block minifloats: OCP MX and NVFP4, whose elements
    share a scale per block). Block floating point with integer mantissas,
    and logarithmic and tapered (posit) formats would also belong here. Each
    needs a kernel first, so none is declared as an empty class in the
    meantime.

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
    """The saturation mode and the domain a format object stores.

    One spelling per combination, so that two spellings of one format compare
    (and memoize) equal: ``SAT_FINITE`` in the extended domain is stored as
    ``SAT_FINITE_EXTENDED``, and the domain is filled in from the mode.
    """
    if not isinstance(saturation, SaturationMode):
        raise TypeError(f"{label} saturation must be a SaturationMode, got {saturation!r}")
    finite = saturation is SaturationMode.SAT_FINITE
    if domain is None:
        return saturation, Domain.FINITE if finite else Domain.EXTENDED
    if not isinstance(domain, Domain):
        raise TypeError(f"{label} domain must be a Domain or None, got {domain!r}")
    if domain is Domain.FINITE:
        if not finite:
            raise ValueError(
                f"{label}: the finite domain admits only SatFinite "
                f"(SaturationMode.SAT_FINITE), got {saturation}"
            )
        return saturation, domain
    if finite:
        return SaturationMode.SAT_FINITE_EXTENDED, domain
    return saturation, domain


# --- how a call reports a finding --------------------------------------------


class FormatRangeWarning(UserWarning):
    """Warns that a format's range outruns the float it is rounded or stored in.

    Quantizing to such a format is partial: the cast is correct over the part
    of the format that the carrier holds, and some of the format's values are
    never returned. The carrier is the operand's: binary64 for a float64
    tensor, and binary32 for float32, float16 and bfloat16, unless the call
    names binary64 with ``carrier=torch.float64``.

    The quantizers and GEMMs of :mod:`mptorch.quant` warn per call when a
    format does one of three things: reaches above the carrier's largest
    finite value; spaces its values finer than the carrier's smallest
    subnormal (:math:`2^{-149}` in binary32, :math:`2^{-1074}` in binary64);
    or has a smallest positive value below the floor the carrier gives it.
    That floor depends on how the cast finds the bottom of the format:

    * :math:`2^{-125}` (:math:`2^{-1021}`) for a :class:`BinaryK` with
      ``SubnormalsMode.SUBNORMALS`` and for a :class:`SuperFP`. Their casts
      place an input on the subnormal or supernormal grid using its exponent
      field, which all of the carrier's subnormals share. A format reaching
      lower would have its bottom binade rounded onto the wrong grid.
    * :math:`2^{-126}` (:math:`2^{-1022}`), the carrier's smallest normal, for
      a :class:`BinaryK` with ``SubnormalsMode.NORMALS`` or
      ``SubnormalsMode.EXTENDED_NORMALS``. Their casts find the bottom by
      comparing magnitudes and read no field. Two cases stop a binade
      earlier, at the first floor: ``P = 1`` under ``NORMALS``, and
      ``EXTENDED_NORMALS`` at the carrier's full precision (``P = 24``,
      ``P = 53``).

    :doc:`/concepts` ("What the carrier can hold") derives both.
    :class:`BinaryK` and :class:`SuperFP` do not warn when they are built,
    because a format does not know which carrier it will meet. They raise only
    for what neither carrier can do.

    The same warning is issued when the result is stored in a dtype narrower
    than the carrier (float16 or bfloat16 under binary32, and float32 as well
    under binary64) and the format's range exceeds *that* dtype: above its
    largest finite value, or spaced finer than its smallest subnormal. The
    cast rounds in the carrier, and storing the result rounds a second time,
    onto the dtype's grid. A value above the dtype's range is stored as an
    infinity, whatever the format's saturation mode.

    This is a warning and not an error because such a format is still useful.
    A wide binaryK is a reasonable precision-only target, and its unreachable
    top is then intended. Each warning names the line that made the call, so
    the default filter reports it once per line, not once per training step.

    Example::

        >>> import warnings
        >>> import torch
        >>> from mptorch import BinaryK, FormatRangeWarning
        >>> from mptorch.quant import Quant
        >>> q = Quant(BinaryK(16, 8, bias=255))  # values down to 2**-261
        >>> warnings.simplefilter("ignore", FormatRangeWarning)
        >>> y = q(torch.ones(2))  # rounded in binary32, without the warning
    """


@lru_cache(maxsize=1)
def _library_frames() -> tuple[str, ...]:
    """The source trees a per-call warning looks past to name its caller.

    Returns the mptorch and torch package directories, each with a trailing
    separator, for ``warnings.warn``'s ``skip_file_prefixes``. A storage
    check runs where the dtype is known, inside the op, and the op is reached
    from a wrapper, a layer's math hook or an autograd Function's backward, at
    different depths and some of them under torch's own frames
    (``Module.__call__``, ``Function.apply``). So rather than count frames
    with ``stacklevel`` the warning names the first frame outside both trees,
    which is the line that made the call: the location worth reading, and the
    one ``warnings``' default once-per-location rule then keys on, so a
    training loop hears it once.
    """
    # mptorch's own tree: this file is mptorch/number/base.py
    roots = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.__file__:
        roots.append(os.path.dirname(os.path.abspath(torch_module.__file__)))
    return tuple(os.path.join(root, "") for root in roots)


def _warn_per_call(message: str) -> None:
    """Warn ``message`` as a ``FormatRangeWarning`` naming the caller's line.

    Under ``torch.compile`` Dynamo runs this once, while it traces, rather
    than tracing it: ``warnings.warn`` is a builtin it does not trace, and
    ``_library_frames`` calls ``os.path``, which it does not either, so
    traced, the call would break the graph (and fail a ``fullgraph`` compile)
    wherever a format warns. The attribute set below is what
    ``torch.compiler.assume_constant_result`` sets, written directly because
    that function imports ``torch._dynamo``, which ``import mptorch`` does
    not otherwise pay for (0.76 s). It holds here because the result is
    ``None`` and the one argument is a string Dynamo has as a constant;
    marking the ``check_*`` functions instead fails on their integer
    arguments, which Dynamo may hold as symbols.
    """
    warnings.warn(message, FormatRangeWarning, skip_file_prefixes=_library_frames())


_warn_per_call._dynamo_marked_constant = True  # ty: ignore[missing-attribute]


def _report_per_call(error: str | None, warning: str | None) -> None:
    """Raise the error if there is one, else warn, naming the caller's line."""
    if error is not None:
        raise ValueError(error)
    if warning is not None:
        _warn_per_call(warning)
