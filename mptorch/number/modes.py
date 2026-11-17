"""The modes a format is rounded under, and its domain.

:class:`RoundMode`, :class:`SaturationMode`, :class:`SubnormalsMode` and
:class:`AccumulateAlgorithm` mirror the C++ enums of ``csrc/common/modes.h``
by name and by integer value, because the kernels receive ``.value``: a change
to one is a change to both. :class:`Domain` exists in Python only: a format
object folds it into its saturation mode when it is built
(``base._resolve_domain``).
"""

from enum import Enum


class SaturationMode(Enum):
    """What a format does with a value beyond its largest finite one.

    Mirrors ``SaturationMode`` in ``csrc/common/modes.h``, by name and value.
    Each member combines one of IEEE P3109's three saturation modes with the
    format's *domain* (see :class:`Domain`), which P3109 chooses separately.
    The finite domain spends on finite values the code points that the
    extended domain spends on the infinities, and it admits only
    ``SatFinite``. That leaves four valid combinations, one member each:
    :attr:`SAT_FINITE` is the finite domain, and the other three are the
    extended one.

    On a :class:`BinaryK` or :class:`SuperFP` the domain can be given with
    ``domain=`` instead: ``saturation=SAT_FINITE, domain=Domain.EXTENDED`` is
    stored as :attr:`SAT_FINITE_EXTENDED`. The flat ``*_quantize`` and
    ``*_matmul`` functions take no format object, so they name the
    combination with the member alone.

    The mode is a property of each format slot. A GEMM whose multiply
    saturates and whose accumulate overflows to infinity can therefore be
    expressed. A NaN input passes through every mode unchanged.

    Example::

        >>> from mptorch import BinaryK, SaturationMode
        >>> from mptorch.quant import Quant
        >>> import torch
        >>> x = torch.tensor([1.1, 3.3, 500.0])
        >>> Quant(BinaryK(8, 4, bias=7))(x).tolist()
        [1.125, 3.25, inf]
        >>> sat = SaturationMode.SAT_FINITE
        >>> Quant(BinaryK(8, 4, bias=7, saturation=sat))(x).tolist()
        [1.125, 3.25, 480.0]
    """

    #: P3109 ``SatFinite``, finite domain: every result beyond the largest
    #: finite value, an infinite input included, clamps to that value.
    SAT_FINITE = 0
    #: P3109 ``SatPropagate``, extended domain: a finite result beyond the
    #: largest finite value clamps to it, and an infinite one stays infinite.
    SAT_PROPAGATE = 1
    #: P3109 ``SatNone``, extended domain, the default: a result beyond the
    #: largest finite value becomes :math:`\pm\infty` under every rounding mode,
    #: because the overflow test reads the rounded value. This differs from IEEE
    #: 754, whose ``roundTowardZero`` never rounds a finite value to infinity.
    SAT_NONE = 2
    #: P3109 ``SatFinite``, extended domain: the format keeps its infinity
    #: codes, so its largest finite value is that of :attr:`SAT_NONE` and
    #: :attr:`SAT_PROPAGATE`. Every result beyond it, an infinite input
    #: included, clamps to it, as under :attr:`SAT_FINITE`.
    SAT_FINITE_EXTENDED = 3


class Domain(Enum):
    r"""Whether a format spends code points on the infinities: P3109's domain.

    The *extended* domain has :math:`\pm\infty` (the top code of each sign).
    The *finite* domain spends those codes on finite values, so its largest
    finite value is one code higher, and it admits only the ``SatFinite``
    saturation mode. A :class:`BinaryK` or :class:`SuperFP` takes the domain
    as ``domain=`` and folds it into its :class:`SaturationMode` when it is
    built. If it is left ``None``, it follows from the saturation mode: finite
    for ``SAT_FINITE`` and extended otherwise.

    Example::

        >>> from mptorch import BinaryK, Domain, SaturationMode
        >>> f = BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE, domain=Domain.EXTENDED)
        >>> f.saturation, f == BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE_EXTENDED)
        (<SaturationMode.SAT_FINITE_EXTENDED: 3>, True)
        >>> BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE).domain
        <Domain.FINITE: 1>
    """

    #: Has :math:`\pm\infty`: P3109's ``e`` formats (``Binary8p4se``).
    EXTENDED = 0
    #: No infinities, and every code a value or the NaN: the ``f`` formats.
    FINITE = 1


class SubnormalsMode(Enum):
    """How a format handles the range below its smallest normal number.

    Mirrors ``SubnormalsMode`` in ``csrc/common/modes.h``, by name and value.
    Only :attr:`SUBNORMALS` is IEEE P3109's: every P3109 format with more than
    one bit of precision has subnormals. The other two give formats outside
    the standard. The three differ only in where the bottom of the range is.

    Below the bottom they all behave the same way. An input ``x`` has two
    candidates for its magnitude, zero and the smallest value ``x_min``, and
    the rounding mode picks between them:

    * :attr:`RoundMode.RNE` and :attr:`RoundMode.RNA` take the nearer one. On
      a tie, ``RNE`` takes the zero and ``RNA`` takes ``x_min``.
    * The directed modes take the candidate in their own direction.
    * :attr:`RoundMode.RO` takes ``x_min``, the nonzero candidate.
    * :attr:`RoundMode.SR` takes ``x_min`` with probability ``|x| / x_min``.

    All three keep the same two special codes, which the casts' NaN detection
    and sign handling rely on. The all-zero code is the one zero. It is
    unsigned, as in P3109, and no cast ever returns ``-0.0``. In a signed
    format, the code with only the sign bit set (the one that would encode
    ``-0.0``) is the NaN.

    Example::

        >>> from mptorch import BinaryK, SubnormalsMode
        >>> from mptorch.quant import Quant
        >>> import torch
        >>> x = torch.tensor([2**-7, 2**-6])  # below and at the smallest normal
        >>> Quant(BinaryK(8, 4, bias=7))(x).tolist()
        [0.0078125, 0.015625]
        >>> flush = SubnormalsMode.NORMALS
        >>> Quant(BinaryK(8, 4, bias=7, subnormals=flush))(x).tolist()
        [0.0, 0.015625]
    """

    SUBNORMALS = 0  #: Gradual underflow, the default: exponent code 0 holds subnormals.
    NORMALS = 1  #: No subnormals: below the smallest normal, zero or that normal.
    #: No subnormals. Exponent code 0 is one more binade of normal numbers,
    #: except its mantissa-zero code, which is still the zero (and with the
    #: sign bit the NaN). That binade's values therefore start one step above
    #: the power of two at its foot.
    EXTENDED_NORMALS = 2


class RoundMode(Enum):
    """How a value is rounded onto a format's grid.

    Mirrors ``RoundMode`` in ``csrc/common/modes.h``, by name and value. Each
    member is one of IEEE P3109's rounding modes, whose name is given in
    parentheses, and defines the result :math:`y` of rounding an input
    :math:`x`.

    The rounding mode is a property of the operation, not of the format. The
    kernels take it as a template parameter, because a runtime ``switch``
    inside a GEMM's accumulate step cost 1.2 to 2.3 times the kernel time.
    :class:`mptorch.quant.Quant` and :class:`mptorch.quant.SplitMac` therefore
    carry it next to the format.

    Stochastic rounding draws each output element's random bits from a stream
    keyed on that element's index in the output. The result does not depend
    on how the work is split across threads or batches. P3109 defines three
    stochastic modes, ``SR`` (``StochasticA``, also spelled ``SR_A``),
    ``SR_B`` and ``SR_C``, which differ in how they read the ``prng_bits``
    random bits (:doc:`/concepts`, "Stochastic rounding").

    Example::

        >>> from mptorch import BinaryK, RoundMode
        >>> from mptorch.quant import Quant
        >>> import torch
        >>> x = torch.tensor([1.1, 3.3])
        >>> Quant(BinaryK(8, 4, bias=7), RoundMode.RNE)(x).tolist()
        [1.125, 3.25]
        >>> Quant(BinaryK(8, 4, bias=7), RoundMode.RZ)(x).tolist()
        [1.0, 3.25]
        >>> Quant(BinaryK(8, 4, bias=7), RoundMode.RU)(x).tolist()
        [1.125, 3.5]
    """

    RNE = 0  #: Round to nearest, ties to even (``NearestTiesToEven``)
    RNA = 1  #: Round to nearest, ties to away from zero (``NearestTiesToAway``)
    RU = 2  #: The smallest :math:`y` such that :math:`y \ge x` (``TowardPositive``)
    RD = 3  #: The largest :math:`y` such that :math:`y \le x` (``TowardNegative``)
    RZ = 4  #: The largest :math:`y` such that :math:`|y| \le |x|` (``TowardZero``)
    RO = 5  #: Round to odd (``ToOdd``)
    #: Stochastic rounding with ``prng_bits`` = :math:`N` random bits :math:`R`
    #: (``StochasticA``): with :math:`\eta` the fraction of :math:`|x|` between
    #: its two neighbours, rounds away from zero iff
    #: :math:`\lfloor \eta 2^N \rfloor + R \ge 2^N`
    SR = 6
    #: ``StochasticB``: away iff :math:`\lfloor \eta 2^{N+1} \rfloor + 2R + 1 \ge 2^{N+1}`
    SR_B = 7
    #: ``StochasticC``: away iff :math:`\mathrm{RNITE}(\eta 2^N) + R \ge 2^N`,
    #: RNITE rounding to nearest with ties to even
    SR_C = 8
    SR_A = 6  #: ``StochasticA``: ``SR``'s other name (an alias)

    @property
    def is_stochastic(self) -> bool:
        """Whether this mode draws random bits: ``SR`` (``SR_A``), ``SR_B`` or ``SR_C``."""
        return self.value >= 6


class AccumulateAlgorithm(Enum):
    """How the partial products of a dot product are folded into its sum.

    Mirrors ``AccumulateAlgorithm`` in ``csrc/common/modes.h``, by name and
    value. It selects the reduction inside a custom-arithmetic GEMM
    (:class:`mptorch.quant.SplitMac`, :class:`mptorch.quant.FusedMac`, the
    flat ``*_matmul`` wrappers).

    Notation: ``a`` and ``b`` are the two operands of one step, and ``s`` is
    the running sum. ``round_mul`` and ``round_acc`` are the mac's two
    roundings. ``round_acc`` is the identity when there is no accumulate
    format. In a fused mac it is the fused format's rounding, and the step is
    ``round_acc(fma(a, b, s))`` in place of
    ``round_acc(s + round_mul(a * b))``. ``round_out`` is the rounding of the
    mac's ``outer`` format (the identity without one). The products are taken
    in order, ``k = 1 .. K``:

    * :attr:`NAIVE`: ``s = round_acc(s + round_mul(a * b))``, as a hardware
      accumulator of the accumulate format does. The result is ``s``.
    * :attr:`KAHAN`: a compensation ``c`` is carried next to the sum, also in
      the accumulate format: ``y = round_acc(round_mul(a * b) - c)``,
      ``t = round_acc(s + y)``, ``c = round_acc(round_acc(t - s) - y)``,
      ``s = t``. The result is ``s``. With no accumulate format this is
      Kahan's summation in the carrier.
    * :attr:`BLOCK`: the ``NAIVE`` step runs on a block's sum ``s``. After
      every ``block_size`` products, and after the last one, the block is
      added to a total, ``tot = round_out(tot + s)``, and ``s`` restarts from
      zero. This gives two levels of precision: a narrow accumulate format
      inside a block, and a wider outer format across blocks. The result is
      ``tot``.
    * :attr:`TREE`: the rounded products ``p1, p2, ...`` of a block of
      ``block_size = 2**L`` are summed pairwise, in the order they arrive:
      ``round_acc(round_acc(p1 + p2) + round_acc(p3 + p4))`` and so on up.
      The block's root is added to the total as in ``BLOCK``. A last, partial
      block merges the subtrees it has, smallest first. The algorithm needs a
      product term, so a fused mac refuses it.

    Under :attr:`RoundMode.SR` every rounding that is not an identity draws
    once, in the order written. Nothing is padded, so a result does not
    depend on how a backend tiles ``K``, and the CPU, CUDA and MPS kernels
    agree to the bit in every deterministic mode (MPS up to the Apple GPU's
    flush of binary32 subnormals in the arithmetic between roundings). An MPS
    result is the CPU's under SR as well, since both draw from the CPU
    generator. ``KAHAN``, ``BLOCK`` and ``TREE`` are implemented for the
    single-format ops, the cross-family ones, the conv ops and the block
    GEMM, in both carriers (binary32 only on MPS, which has no float64). A
    palette raises.

    Example::

        >>> from mptorch import AccumulateAlgorithm
        >>> AccumulateAlgorithm.NAIVE.value
        0
    """

    NAIVE = 0  #: Quantize the running sum after every accumulation step.
    KAHAN = 1  #: Kahan-compensated summation, the compensation in the accumulate format.
    BLOCK = 2  #: Two-level (FABSum-style) block summation with an outer format.
    TREE = 3  #: Pairwise summation within a block, folded with an outer format.
