__all__ = [
    "Number",
    "FloatType",
    "FixedPoint",
    "FloatingPoint",
    "BlockFloatingPoint",
    "SuperNormalFloat",
    "Binary8",
]


class Number:
    """Base class of all number formats."""

    def __init__(self):
        pass

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class FloatType(Number):
    """Base class of float-like number formats."""

    pass


class FixedPoint(Number):
    r"""
    Low-Precision Fixed Point Format. Defined similarly in
    *Deep Learning with Limited Numerical Precision* (https://arxiv.org/abs/1502.02551)

    The representable range is :math:`[-2^{wl-fl-1}, 2^{wl-fl-1}-2^{-fl}]`
    and a precision unit (smallest nonzero absolute value) is
    :math:`2^{-fl}`.
    Numbers outside of the representable range can be clamped
    (if `clamp` is true).
    We can also give up the smallest representable number to make the range
    symmetric, :math:`[-2^{wl-fl-1}^{-fl}, 2^{wl-fl-1}-2^{-fl}]`. (if `symmetric` is true).

    Define :math:`\lfloor x \rfloor` to be the largest representable number (multiples of :math:`2^{-fl}`) smaller than :math:`x`.
    For numbers within the representable range, fixed point quantizatio corresponds to

    .. math::

       NearestRound(x)
       =
       \Biggl \lbrace
       {
       \lfloor x \rfloor, \text{ if } \lfloor x \rfloor \leq x \leq \lfloor x \rfloor + 2^{-fl-1}
       \atop
        \lfloor x \rfloor + 2^{-fl}, \text{ if } \lfloor x \rfloor + 2^{-fl-1} < x \leq \lfloor x \rfloor + 2^{-fl}
       }

    or

    .. math::
       StochasticRound(x)
       =
       \Biggl \lbrace
       {
       \lfloor x \rfloor, \text{ with probabilty } 1 - \frac{x - \lfloor x \rfloor}{2^{-fl}}
       \atop
        \lfloor x \rfloor + 2^{-fl}, \text{ with probabilty } \frac{x - \lfloor x \rfloor}{2^{-fl}}
       }

    Args:
        - :attr: wl (int) : word length of each fixed point number
        - :attr: fl (int) : fractional length of each fixed point number
        - :attr: clamp (bool) : whether to clamp unrepresentable numbers
        - :attr: symmetric (bool) : whether to make the representable range symmetric
    """

    def __init__(self, wl, fl, clamp=True, symmetric=False):
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
        return "FixedPoint (wl={:d}, fl={:d})".format(self.wl, self.fl)

    def __repr__(self):
        return "FixedPoint (wl={:d}, fl={:d})".format(self.wl, self.fl)


class FloatingPoint(FloatType):
    """
    Low-Precision Floating Point Format.

    We set the exponent bias to be :math:`2^{exp-1} - 1`. For rounding
    mode, we apply *round to nearest even*.

    Args:
        - :attr: `exp`: number of bits allocated for exponent
        - :attr: `man`: number of bits allocated for mantissa, referring to number of bits that are
                        supposed to be stored on hardware (not counting the virtual bits)
        - :attr: `subnormals`: allow the use of subnormal values
        - :attr: `saturate`: clamp values instead of using infinities in case of overflow
    """

    def __init__(self, exp, man, subnormals=False, saturate=True):
        assert 8 >= exp > 0, "invalid bits for exponent:{}".format(exp)
        assert 23 >= man > 0, "invalid bits for mantissa:{}".format(man)
        self.exp = exp
        self.man = man
        self.subnormals = subnormals
        self.saturate = saturate
        self.subnormal_min = (
            2.0 ** (2 - 2 ** (self.exp - 1) - self.man) if subnormals else None
        )
        self.subnormal_max = (
            2.0 ** (2 - 2 ** (self.exp - 1)) * (1.0 - 2.0 ** (-self.man))
            if subnormals
            else None
        )
        self.normal_max = 2.0 ** (2 ** (self.exp - 1) - 1) * (2.0 - 2.0 ** (-self.man))
        self.normal_min = 2.0 ** (2 - 2 ** (self.exp - 1))

    def __str__(self):
        return "FloatingPoint (exponent={:d}, mantissa={:d})".format(self.exp, self.man)

    def __repr__(self):
        return "FloatingPoint (exponent={:d}, mantissa={:d})".format(self.exp, self.man)

    @property
    def is_fp32(self) -> bool:
        return self.man == 23 and self.exp == 8

    @property
    def is_fp16(self) -> bool:
        return self.man == 10 and self.exp == 5

    @property
    def is_bfloat16(self) -> bool:
        return self.man == 7 and self.exp == 8


class SuperNormalFloat(FloatType):
    """
    Low-Precision SuperNormal Floating Point Format.

    We set the exponent bias to be :math:`2^{exp-1}`. For rounding
    mode, we apply *round to nearest even*.

    Args:
        - :attr: `exp`: number of bits allocated for exponent
        - :attr: `man`: number of bits allocated for mantissa, referring to number of bits that are
                        supposed to be stored on hardware (not counting the virtual bits)
        - :attr: `binades`: number of binades tranformed into log range
        - :attr: `saturate`: clamp values instead of using infinities in case of overflow
    """

    def __init__(self, exp: int, man: int, binades: int | tuple[int], saturate=False):
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
    """
    Low-Precision Block Floating Point Format.

    BlockFloatingPoint shares an exponent across a block of numbers. The shared exponent is chosen from
    the largest magnitude in the block.

    Args:
        - :attr: `wl` word length of the tensor
        - :attr: `dim` block dimension to share exponent. (*, D, *) Tensor where
          D is at position `dim` will have D different exponents; use -1 if the
          entire tensor is treated as a single block (there is only 1 shared
          exponent).
    """

    def __init__(self, wl, dim=-1):
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


class Binary8(FloatType):
    """
    Low-Precision Binary8 Format following the P3109 standard.

    Binary8 is a format that takes a value P as an input to determines the number
    of mantissa and exponent bits.

    Args:
        - :attr: `P`: integer precision of the binary8 format
        - :attr: `signed`: boolean indicating whether the format is signed or unsigned
        - :attr: `subnormals`: allow the use of subnormal values
        - :attr: `overflow_policy`: string indicating the overflow policy (dictates the max float)
                                    - saturate_maxfloat2 : no infinity and +1 normalized value
                                    - saturate_maxfloat : use infinity
                                    - saturate_infty : use infinity
    """

    def __init__(
        self, P: int, signed=True, subnormals=True, overflow_policy="saturate_maxfloat2"
    ):
        assert 8 > P > 0, "Invalid P: {}".format(P)  # is P = 8 valid?
        assert overflow_policy in (
            "saturate_infty",
            "saturate_maxfloat",
            "saturate_maxfloat2",
        ), "Invalid overflow policy: {}".format(overflow_policy)

        self.P = P
        spec_exp = P == 1

        # Determine mantissa and exponent bits based on P and signed
        self.man = P - 1
        self.exp = (8 - P) if signed else (9 - P)
        max_exp = 2 ** (self.exp - 1) - 1
        min_exp = -max_exp + spec_exp

        # Define the subnormal and normal ranges
        if self.man != 1 and subnormals == True:
            self.subnormal_min = (2**-self.man) * (2**min_exp)  # this is good
            self.subnormal_max = (1 - 2**-self.man) * (2**min_exp)  # this is also good
        else:
            self.subnormal_min = None
            self.subnormal_max = None

        # no subnormal case
        if subnormals == True or self.man == 1:
            self.normal_min = 2**min_exp  # this is good
        else:  # values of P with subnormal values will have different normal_min
            self.normal_min = (1 + 2**-self.man) * (2 ** (min_exp - 1))

        if signed:
            if (
                overflow_policy == "saturate_maxfloat2"
            ):  # no inf case, so max is FF not FE
                if self.man > 0:
                    self.normal_max = (2 - 2**-self.man) * (
                        2**max_exp
                    )  # good for more than 0 mantissa
                else:
                    self.normal_max = 2 ** (max_exp + 1)  # 0 mantissa case
            else:  # normal case where FE is max
                if self.man > 0:
                    self.normal_max = (2 - 2 ** -(self.man - 1)) * (
                        2**max_exp
                    )  # good for more than 0 mantissa
                else:
                    self.normal_max = 2**max_exp  # 0 mantissa case
        else:  # unsigned case
            if (
                overflow_policy == "saturate_maxfloat2"
            ):  # no inf case, so max is FE not FD
                if self.man > 0:
                    self.normal_max = (2 - 2 ** -(self.man - 1)) * (
                        2**max_exp
                    )  # good for more than 0 mantissa
                else:
                    self.normal_max = 2**max_exp  # 0 mantissa case
            else:  # normal case where FD is max
                if self.man > 0:
                    self.normal_max = (2 - 2**-self.man - 2 ** -(self.man - 1)) * (
                        2**max_exp
                    )  # good for more than 1 mantissa
                elif self.man == 1:
                    self.normal_max = 1.5 * (
                        2 ** (max_exp - 1)
                    )  # (2 - 2**-self.man) * (2**(max_exp-1))
                else:
                    self.normal_max = 2 ** (max_exp - 1)

        self.signed = signed
        self.subnormals = subnormals
        self.overflow_policy = overflow_policy

    def __repr__(self):
        return (
            f"Binary8 (P={self.P}, exp={self.exp}, man={self.man}, signed={self.signed}, "
            f"subnormals={self.subnormals}, overflow_policy={self.overflow_policy})"
        )

    def __str__(self):
        return self.__repr__()
