__all__ = [
    "Number",
    "FloatType",
    "FixedPoint",
    "FloatingPoint",
    "BlockFloatingPoint",
    "SuperNormalFloat",
    "SaturationMode",
    "SubnormalsMode",
]

from enum import Enum

SaturationMode = Enum(
    "SaturationMode", [("SAT_FINITE", 0), ("SAT_PROPAGATE", 1), ("OVF_INF", 2)]
)

SubnormalsMode = Enum(
    "SubnormalsMode", [("SUBNORMALS", 0), ("NORMALS", 1), ("EXTENDED_NORMALS", 2)]
)


class RoundMode(Enum):
    """
    Enum for floating-point rounding modes.

    Result :math:`y` is obtained from an input :math:`x` depending on the rounding mode.
    """

    RNE = 0  #: Round to nearest, ties to even
    RNA = 1  #: Round to nearest, ties to away from zero
    RU = 2  #: Return the smallest :math:`y` such that :math:`y \ge x`
    RD = 3  #: Return the largest :math:`y` such that :math:`y \le x`
    RZ = 4  #: Return the largest :math:`y` such that :math:`|y| \le |x|`
    SR = 5  #: Stochastic Rounding


class Number:
    """Base class for all supported number formats.

    Users should always instantiate one of the derived classes.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        """Placeholder __str__ method."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Placeholder __repr__ method."""
        raise NotImplementedError


class FloatType(Number):
    """Base class for all float-like number formats.

    Similar to the :class:`Number` class, users should not instantiate
    this class directly. It is useful as a means to determine if
    a number format is of float type.
    """

    pass


class FixedPoint(Number):
    r"""
    Low-Precision Fixed Point Number Format. Defined similarly to
    *Deep Learning with Limited Numerical Precision* (https://arxiv.org/abs/1502.02551).

    The representable range is :math:`\left[-2^{\text{wl}-\text{fl}-1},
    2^{\text{wl}-\text{fl}-1}-2^{-\text{fl}}\right]`
    and the precision unit (smallest nonzero absolute value) is :math:`2^{-\text{fl}}`.
    Numbers outside of the representable range can be clamped (if `clamp` is true).
    We can also give up the smallest representable number to make the range symmetric,
    :math:`\left[-2^{\text{wl}-\text{fl}-1}+2^{-\text{fl}},
    2^{\textnormal{wl}-\text{fl}-1}-2^{-\text{fl}}\right]` (if `symmetric` is true).

    Define :math:`\lfloor x \rfloor` to be the largest representable number (multiples of :math:`2^-\text{fl}`)
    smaller than :math:`x`. For numbers within the representable range, we support two kinds of fixed point
    quantization: *round to nearest* (RN) and *stochastic rounding* (SR). They correspond to

    .. math::

       \text{RN}(x)
       =
       \Biggl \lbrace
       {
       \lfloor x \rfloor, \text{ if } \lfloor x \rfloor \leq x \leq \lfloor x \rfloor + 2^{-\text{fl}-1}
       \atop
        \lfloor x \rfloor + 2^{-\text{fl}}, \text{ if } \lfloor x \rfloor + 2^{-\text{fl}-1} < x \leq \lfloor x \rfloor + 2^{-\text{fl}}
       }

    or

    .. math::
       \textnormal{SR}(x)
       =
       \Biggl \lbrace
       {
       \lfloor x \rfloor, \text{ with probabilty } 1 - \frac{x - \lfloor x \rfloor}{2^{-\text{fl}}}
       \atop
        \lfloor x \rfloor + 2^{-\text{fl}}, \text{ with probabilty } \frac{x - \lfloor x \rfloor}{2^{-\text{fl}}}
       }

    Args:
        wl: word length of each fixed point number
        fl: fractional length of each fixed point number
        clamp: whether to clamp unrepresentable numbers
        symmetric: whether to make the representable range symmetric
    """

    def __init__(self, wl: int, fl: int, clamp: bool = True, symmetric: bool = False):
        assert wl > 0, "invalid bits for word length: {}".format(wl)
        assert fl > 0, "invalid bits for fractional length: {}".format(fl)
        assert type(symmetric) == bool, "invalid type for clamping choice: {}".format(
            type(clamp)
        )
        assert type(symmetric) == bool, "invalid type for symmetric: {}".format(
            type(symmetric)
        )
        self.wl = wl
        self.fl = fl
        self.clamp = clamp
        self.symmetric = symmetric

    def __str__(self):
        """Print format information in a string format."""
        return "FixedPoint (wl={:d}, fl={:d})".format(self.wl, self.fl)

    def __repr__(self):
        """Custom __repr__ method yielding information about the format."""
        return "FixedPoint (wl={:d}, fl={:d})".format(self.wl, self.fl)


class FloatingPoint(FloatType):
    r"""
    Low-Precision Floating Point Format. Follows rules set out in the IEEE-754 standard, applying
    them in the context of custom precision formats.

    By default, we set the exponent bias to be :math:`2^{\text{exp}-1} - 1`. In terms of rounding mode (see
    available quantization functions), we offer support for: *round to nearest even* (RNE),
    *round to nearest away* (RNA), *round towards positive infinity* (RU), *round towards negative infinity* (RD),
    *round towards zero* (RZ), and *stochastic rounding* (SR).

    Args:
        exp: number of bits allocated for exponent
        man: number of bits allocated for mantissa, referring to number of bits that are
            supposed to be stored on hardware (not counting the virtual bits)
        bias: the bias for the number format (if none specified use the default value)
        subnormals: allow the use of subnormal values
        saturate: clamp values instead of using infinities in case of overflow
    """

    def __init__(
        self,
        exp: int,
        man: int,
        bias: int | None = None,
        subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS,
        saturate: bool = False,
    ):
        assert 8 >= exp > 0, "invalid bits for exponent:{}".format(exp)
        assert 23 >= man >= 0, "invalid bits for mantissa:{}".format(man)
        self.exp = exp
        self.man = man
        self.subnormals = subnormals
        self.saturate = saturate
        if bias:
            self.bias = bias
        else:
            self.bias = 2 ** (exp - 1) - 1

        self.subnormal_min = (
            2.0 ** (1 - self.bias - self.man)
            if subnormals == SubnormalsMode.SUBNORMALS
            else None
        )
        self.subnormal_max = (
            2.0 ** (1 - self.bias) * (1.0 - 2.0 ** (-self.man))
            if subnormals == SubnormalsMode.SUBNORMALS
            else None
        )
        self.normal_max = 2.0 ** (2**self.exp - 2 - self.bias) * (
            2.0 - 2.0 ** (-self.man)
        )
        self.normal_min = (
            2.0 ** (-self.bias)
            if subnormals == SubnormalsMode.EXTENDED_NORMALS
            else 2.0 ** (1 - self.bias)
        )

    def __str__(self):
        return f"FloatingPoint (exponent={self.exp}, mantissa={self.man}, bias={self.bias})"

    def __repr__(self):
        return f"FloatingPoint (exponent={self.exp}, mantissa={self.man}, bias={self.bias})"

    @property
    def is_fp32(self) -> bool:
        """Returns if the format is equivalent to the IEEE-754 ``binary32`` format."""
        return self.man == 23 and self.exp == 8 and self.bias == 127

    @property
    def is_fp16(self) -> bool:
        """Returns if the format is equivalent to the IEEE-754 ``binary16`` format."""
        return self.man == 10 and self.exp == 5 and self.bias == 15

    @property
    def is_bfloat16(self) -> bool:
        """Returns if the format is equivalent to the ``bfloat16`` format."""
        return self.man == 7 and self.exp == 8 and self.bias == 127


class SuperNormalFloat(FloatType):
    r"""
    Low-Precision SuperNormal Floating Point Format. Described in
    *Range Extension with Supernormals for Mixed-Precision 8-bit DNN Training*
    (https://www.arith2025.org/proceedings/215900a017.pdf)

    We set the exponent bias to be :math:`2^{\text{exp}-1}`. For rounding
    mode, we apply *round to nearest even*.

    Args:
        exp: number of bits allocated for exponent
        man: number of bits allocated for mantissa, referring to number of bits that are
            supposed to be stored on hardware (not counting the virtual bits)
        binades: number of binades transformed into log range
        saturate: clamp values instead of using infinities in case of overflow
    """

    def __init__(
        self,
        exp: int,
        man: int,
        binades: int | tuple[int] | tuple[int, int],
        saturate=False,
    ):
        assert 8 >= exp > 0, "invalid bits for exponent:{}".format(exp)
        assert 23 >= man > 0, "invalid bits for mantissa:{}".format(man)
        if isinstance(binades, int):
            assert 8 >= binades > 0, "invalid binade size:{}".format(binades)
        elif len(binades) == 1:
            assert 8 >= binades[0] > 0, "invalid binade size:{}".format(binades[0])
        else:
            assert 8 >= binades[0] > 0, "invalid binade size:{}".format(binades[0])
            assert 8 >= binades[1] > 0, "invalid binade size:{}".format(binades[1])
        self.exp = exp
        self.man = man
        if isinstance(binades, int):
            self.binades_l, self.binades_u = binades, binades
        elif len(binades) == 1:
            self.binades_l, self.binades_u = binades[0], binades[0]
        else:
            self.binades_l, self.binades_u = binades[0], binades[1]
        # TODO: to remove
        self.binades = (self.binades_l, self.binades_u)
        self.saturate = saturate

        min_exp = 1 - 2 ** (exp - 1) + (self.binades_l - 1)
        max_exp = 2 ** (exp - 1) - 2 - (self.binades_u - 1)
        self.subnormal_min = 2 ** (min_exp - self.binades_l * (2**man) + 1)
        self.subnormal_max = 2 ** (min_exp - 1)
        self.normal_min = 2**min_exp
        self.normal_max = 2**max_exp
        self.supernormal_min = 2 ** (max_exp + 1)
        self.supernormal_max = 2 ** (
            max_exp + self.binades_u * (2**man) - 1 + int(saturate)
        )

    def __str__(self):
        return "SuperNormalFloat (exponent={:d}, mantissa={:d}, binades=({:d}, {:d}))".format(
            self.exp, self.man, self.binades_l, self.binades_u
        )

    def __repr__(self):
        return "SuperNormalFloat (exponent={:d}, mantissa={:d}, binades=({:d}, {:d}))".format(
            self.exp, self.man, self.binades_l, self.binades_u
        )


class BlockFloatingPoint(Number):
    r"""
    Low-Precision Block Floating Point Format.

    BlockFloatingPoint shares an exponent across a block of numbers. The shared
    exponent is chosen from the largest magnitude in the block.

    Args:
        wl: word length of the tensor
        dim: block dimension to share exponent. (\*, D, \*) Tensor where
            D is at position `dim` will have D different exponents; use -1 if the
            entire tensor is treated as a single block (there is only 1 shared
            exponent).
    """

    def __init__(self, wl: int, dim: int = -1):
        assert wl > 0 and isinstance(wl, int), "invalid bits for word length:{}".format(
            wl
        )
        assert dim >= -1 and isinstance(dim, int), "invalid dimension"
        self.wl = wl
        self.dim = dim

    def __str__(self):
        return "BlockFloatingPoint (wl={:d}, dim={:d})".format(self.wl, self.dim)

    def __repr__(self):
        return "BlockFloatingPoint (wl={:d}, dim={:d})".format(self.wl, self.dim)
