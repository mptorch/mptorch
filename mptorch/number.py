__all__ = ["Number", "FloatType", "FixedPoint", "FloatingPoint", "BlockFloatingPoint", "SuperNormalFloat", "Binary8"]


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
        return "FixedPoint (wl={:d}, fl={:d})".format(self.wl, self.fl)

    def __repr__(self):
        return "FixedPoint (wl={:d}, fl={:d})".format(self.wl, self.fl)


class FloatingPoint(FloatType):
    """
    Low-Precision Floating Point Format.

    We set the exponent bias to be :math:`2^{exp-1} - 1`. For rounding
    mode, we apply *round to nearest even*.

    Args:
        exp: number of bits allocated for exponent
        man: number of bits allocated for mantissa, referring to number of bits that are
             supposed to be stored on hardware (not counting the virtual bits)
        subnormals: allow the use of subnormal values
        saturate: clamp values instead of using infinities in case of overflow
    """

    def __init__(self, exp: int, man: int, subnormals: bool = False, saturate: bool = True):
        assert 8 >= exp > 0, "invalid bits for exponent:{}".format(exp)
        assert 23 >= man > 0, "invalid bits for mantissa:{}".format(man)
        self.exp = exp
        self.man = man
        self.subnormals = subnormals
        self.saturate = saturate
        self.subnormal_min = 2.0**(2 - 2**(self.exp-1) - self.man) if subnormals else None
        self.subnormal_max = 2.0**(2 - 2**(self.exp-1)) * (1.0 - 2.0**(-self.man)) if subnormals else None
        self.normal_max    = 2.0**(2**(self.exp-1)-1) * (2.0 - 2.0**(-self.man))
        self.normal_min    = 2.0**(2 - 2**(self.exp-1))

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
        exp: number of bits allocated for exponent
        man: number of bits allocated for mantissa, referring to number of bits that are
             supposed to be stored on hardware (not counting the virtual bits)
        binades: number of binades tranformed into log range
        saturate: clamp values instead of using infinities in case of overflow
    """

    def __init__(self, exp: int, man: int, binades: int, saturate: bool = False):
        assert 8 >= exp > 0, "invalid bits for exponent:{}".format(exp)
        assert 23 >= man > 0, "invalid bits for mantissa:{}".format(man)
        assert 8 >= binades > 0, "invalid binade size:{}".format(binades)
        self.exp = exp
        self.man = man
        self.binades = binades
        self.saturate = saturate

        min_exp = 1 - 2**exp + (binades - 1)
        max_exp = 2**(exp-1) - 2 - (binades - 1)
        self.subnormal_min = 2**(min_exp - binades * (2**man) + 2)
        self.subnormal_max = 2**(min_exp - 1)
        self.normal_min = 2**min_exp
        self.normal_max = 2**max_exp
        self.supernormal_min = 2**(max_exp + 1)
        self.supernormal_max = 2**(max_exp + binades * (2**man) - 1 + int(saturate))

    def __str__(self):
        return "SuperNormalFloat (exponent={:d}, mantissa={:d}, binades={:d})".format(self.exp, self.man, self.binades)

    def __repr__(self):
        return "SuperNormalFloat (exponent={:d}, mantissa={:d}, binades={:d})".format(self.exp, self.man, self.binades)


class BlockFloatingPoint(Number):
    """
    Low-Precision Block Floating Point Format.

    BlockFloatingPoint shares an exponent across a block of numbers. The shared exponent is chosen from
    the largest magnitude in the block.

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


class Binary8(FloatType):
    """
    Low-Precision Binary8 Format following the P3109 standard.

    Binary8 is a format that takes a value P as an input to determines the number
    of mantissa and exponent bits.

    Args:
        P: integer precision of the binary8 format
        signed: boolean indicating whether the format is signed or unsigned
        subnormals: allow the use of subnormal values
        overflow_policy: string indicating the overflow policy, one of:
            `saturate_maxfloat2` (no infinity and +1 normalized value),
            `saturate_maxfloat` (clamp to maxfloat),
            `saturate_infty` (use infinity)
    """

    def __init__(self,
                 P: int,
                 signed: bool = True,
                 subnormals: int = True,
                 overflow_policy: str = "saturate_maxfloat2"
                ):
        assert 8 > P > 0, "Invalid P: {}".format(P)  # is P = 8 valid?
        assert overflow_policy in ("saturate_infty", "saturate_maxfloat", "saturate_maxfloat2"), \
            "Invalid overflow policy: {}".format(overflow_policy)

        self.P = P
        spec_exp = P == 1
        
        # Determine mantissa and exponent bits based on P and signed
        self.man = P - 1
        self.exp = (8 - P) if signed else (9 - P)
        max_exp = 2**(self.exp - 1) - 1
        min_exp = -max_exp + spec_exp

        # Define the subnormal and normal ranges
        if self.man != 1 and subnormals == True:
            self.subnormal_min = (2**-self.man) * (2**min_exp)   # this is good
            self.subnormal_max = (1 - 2**-self.man) * (2**min_exp)   # this is also good
        else:
            self.subnormal_min = None
            self.subnormal_max = None

        # no subnormal case
        if subnormals == True or self.man == 1:
            self.normal_min = 2**min_exp   # this is good
        else:   # values of P with subnormal values will have different normal_min
            self.normal_min = (1 + 2**-self.man) * (2**(min_exp - 1))
    
        if signed:
            if overflow_policy == "saturate_maxfloat2":    # no inf case, so max is FF not FE
                if self.man > 0:
                    self.normal_max = (2 - 2 **-self.man) * (2**max_exp)    # good for more than 0 mantissa 
                else:
                    self.normal_max = 2**(max_exp + 1)  # 0 mantissa case
            else:   # normal case where FE is max   
                if self.man > 0:
                    self.normal_max = (2 - 2**-(self.man-1)) * (2**max_exp)    # good for more than 0 mantissa 
                else:
                    self.normal_max = 2**max_exp    # 0 mantissa case
        else:   # unsigned case
            if overflow_policy == "saturate_maxfloat2":    # no inf case, so max is FE not FD
                if self.man > 0:
                    self.normal_max = (2 - 2**-(self.man-1)) * (2**max_exp)    # good for more than 0 mantissa 
                else:
                    self.normal_max = 2**max_exp    # 0 mantissa case
            else:   # normal case where FD is max
                if self.man > 0:
                    self.normal_max = (2 - 2**-self.man - 2**-(self.man-1)) * (2**max_exp)  # good for more than 1 mantissa
                elif self.man == 1:
                    self.normal_max = 1.5 * (2**(max_exp-1))      # (2 - 2**-self.man) * (2**(max_exp-1))
                else:
                    self.normal_max = 2**(max_exp-1)
        
        self.signed = signed
        self.subnormals = subnormals
        self.overflow_policy = overflow_policy

    def __repr__(self):
        return f"Binary8 (P={self.P}, exp={self.exp}, man={self.man}, signed={self.signed}, " \
               f"subnormals={self.subnormals}, overflow_policy={self.overflow_policy})"
    
    def __str__(self):
        return self.__repr__()
