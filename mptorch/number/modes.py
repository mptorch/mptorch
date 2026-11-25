"""The enums for rounding, saturation, subnormals and accumulation, the format domain,
and a fixed-point sum's overflow.

:class:`RoundMode`, :class:`SaturationMode`, :class:`SubnormalsMode` and
:class:`AccumulateAlgorithm` have the same names and integer values as the C++
enums in ``csrc/common/modes.h``, because the kernels receive ``.value``: a
change to one must be made to the other. :class:`Domain` exists only in
Python: when a format object is built, it records the domain in its saturation
mode (``base._resolve_domain``). :class:`Overflow` exists only in Python too: a
fixed-point mac passes it to its op as a flag.
"""

from enum import Enum


class SaturationMode(Enum):
    """What a format does with a result whose magnitude exceeds its largest finite value.

    Mirrors ``SaturationMode`` in ``csrc/common/modes.h``, by name and value.
    IEEE P3109 specifies two separate properties: a saturation mode
    (``SatFinite``, ``SatPropagate`` or ``SatNone``) and a *domain* (see
    :class:`Domain`), which says whether the format has infinities. The finite
    domain has no infinities and allows only ``SatFinite``, so four
    combinations are valid, and each member of this enum is one of them:
    :attr:`SAT_FINITE` is the finite domain, and the other three members are
    in the extended domain.

    A :class:`BinaryK` or :class:`SuperFP` also accepts the two properties
    separately, as P3109 states them: ``saturation=SAT_FINITE,
    domain=Domain.EXTENDED`` is stored as :attr:`SAT_FINITE_EXTENDED`. The flat
    ``*_quantize`` and ``*_matmul`` functions take no format object and no
    ``domain``, so they need the combined member.

    Each format in an operation has its own saturation mode, so a GEMM can,
    for example, saturate its products and let its running sum overflow to
    infinity. A NaN input stays a NaN in every mode.

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

    #: P3109 ``SatFinite`` in the finite domain: any result beyond the largest
    #: finite value, including an infinite input, is clamped to the largest
    #: finite value of the same sign.
    SAT_FINITE = 0
    #: P3109 ``SatPropagate`` in the extended domain: a finite input that
    #: rounds beyond the largest finite value is clamped to it, and an
    #: infinite input stays infinite.
    SAT_PROPAGATE = 1
    #: P3109 ``SatNone`` in the extended domain, the default: a result that
    #: rounds beyond the largest finite value becomes :math:`\pm\infty`, in
    #: every rounding mode, because overflow is tested after rounding. This
    #: differs from IEEE 754, where, for example, round-toward-zero never turns
    #: a finite value into an infinity.
    SAT_NONE = 2
    #: P3109 ``SatFinite`` in the extended domain: the format keeps its
    #: infinity codes, so its largest finite value is the same as under
    #: :attr:`SAT_NONE` and :attr:`SAT_PROPAGATE`. As under :attr:`SAT_FINITE`,
    #: any result beyond it, including an infinite input, is clamped to it.
    SAT_FINITE_EXTENDED = 3


class Domain(Enum):
    r"""Whether a format has infinities: P3109's *domain*.

    In the *extended* domain the top code of each sign encodes
    :math:`\pm\infty`. The *finite* domain uses those codes for finite values
    instead, so its largest finite value is one code higher, and it allows
    only the ``SatFinite`` saturation mode. :class:`BinaryK` and
    :class:`SuperFP` take the domain as ``domain=`` and record it in their
    :class:`SaturationMode` when they are built. When ``domain`` is ``None``,
    it follows from the saturation mode: finite for ``SAT_FINITE``, extended
    otherwise.

    Example::

        >>> from mptorch import BinaryK, Domain, SaturationMode
        >>> f = BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE, domain=Domain.EXTENDED)
        >>> f.saturation, f == BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE_EXTENDED)
        (<SaturationMode.SAT_FINITE_EXTENDED: 3>, True)
        >>> BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE).domain
        <Domain.FINITE: 1>
    """

    #: With :math:`\pm\infty`: P3109's ``e`` formats, such as ``Binary8p4se``.
    EXTENDED = 0
    #: Without infinities, so every code is a finite value or the NaN. These
    #: are P3109's ``f`` formats, such as ``Binary8p4sf``.
    FINITE = 1


class Overflow(Enum):
    """What a fixed-point mac's running sum does past its format's range.

    A fixed-point word has no code for infinity, so a sum past the range
    either stops at the end on its side, as every fixed-point rounding in
    mptorch does, or wraps around modulo ``2**wl``, as two's complement and
    unsigned integer hardware does. Only the accumulate format of a
    fixed-point :class:`~mptorch.quant.SplitMac` or
    :class:`~mptorch.quant.FusedMac` takes ``WRAP``; the product's rounding
    always saturates, and so does an infinite product.

    With ``WRAP``, a sum keeps the exact sum's residue: the result is right
    whenever the final sum fits the format, however the partial sums
    overflowed on the way. A symmetric format has no modulus and refuses it.

    Example::

        >>> import torch
        >>> from mptorch import FixedPoint, Overflow
        >>> from mptorch.quant import SplitMac, qmm
        >>> a = torch.tensor([[100.0, 100.0, -100.0, -60.0]])
        >>> b = torch.ones(4, 1)
        >>> int8, int16 = FixedPoint(8, 0), FixedPoint(16, 0)
        >>> qmm(a, b, SplitMac(int16, int8)).item()   # 127 after the second step
        -33.0
        >>> qmm(a, b, SplitMac(int16, int8, overflow=Overflow.WRAP)).item()
        40.0
    """

    #: Stop at the end of the range on the sum's side.
    SATURATE = 0
    #: Wrap around modulo ``2**wl``, keeping the exact sum's residue.
    WRAP = 1


class SubnormalsMode(Enum):
    """How a format handles the range below its smallest normal number.

    Mirrors ``SubnormalsMode`` in ``csrc/common/modes.h``, by name and value.
    Only :attr:`SUBNORMALS` is in IEEE P3109: every P3109 format with more than
    one bit of precision has subnormals. The other two modes give formats
    outside the standard. The three modes differ in which values lie below
    the smallest normal number, and so in where the format's smallest
    positive value ``x_min`` is.

    Below ``x_min`` all three round the same way: the result's magnitude is
    either zero or ``x_min``, and the rounding mode chooses:

    * :attr:`RoundMode.RNE` and :attr:`RoundMode.RNA` take the nearer one. On
      a tie, ``RNE`` takes zero and ``RNA`` takes ``x_min``.
    * The directed modes take the candidate in their direction.
    * :attr:`RoundMode.RO` takes ``x_min``, the nonzero candidate.
    * :attr:`RoundMode.SR` takes ``x_min`` with probability ``|x| / x_min``.

    All three modes give two special codes the same meaning, which the casts'
    NaN detection and sign handling rely on. The all-zero code is the only
    zero; it is unsigned, as in P3109, and no cast returns ``-0.0``. In a
    signed format, the code with only the sign bit set, which would otherwise
    encode ``-0.0``, is the NaN.

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

    SUBNORMALS = 0  #: Gradual underflow, the default. Exponent field 0 holds the subnormals.
    NORMALS = 1  #: No subnormals. A value below the smallest normal rounds to zero or to it.
    #: No subnormals. Exponent field 0 holds one more binade (power-of-two
    #: interval) of normal numbers. Its code with a zero mantissa still
    #: encodes zero (and, with the sign bit set, the NaN), so this binade has
    #: no value at its lowest power of two: its smallest value is one step
    #: above it.
    EXTENDED_NORMALS = 2


class RoundMode(Enum):
    """How a value is rounded to a value of the format.

    Mirrors ``RoundMode`` in ``csrc/common/modes.h``, by name and value. Each
    member is one of IEEE P3109's rounding modes (P3109's name is given in
    parentheses); the descriptions give the result :math:`y` of rounding an
    input :math:`x`.

    The rounding mode belongs to the operation, not to the format. The
    kernels take it as a compile-time template parameter, so one mode applies
    to a whole kernel (a GEMM's multiply and accumulate alike); choosing it at
    run time inside a GEMM's accumulation loop made the kernels 1.2 to 2.3
    times slower. :class:`mptorch.quant.Quant` and
    :class:`mptorch.quant.SplitMac` therefore take the mode alongside the
    format, not inside it.

    Stochastic rounding draws the random bits for each output element from a
    random stream selected by that element's index in the output, so the
    result does not depend on how the work is split across threads or
    batches. P3109 defines three stochastic modes, ``SR`` (``StochasticA``,
    also named ``SR_A``), ``SR_B`` and ``SR_C``, which differ in how they
    combine the ``prng_bits`` random bits with the input (:doc:`/concepts`,
    "Stochastic rounding").

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
    #: Round to odd (``ToOdd``): an inexact :math:`x` goes to whichever of its
    #: two neighbours has an odd last mantissa bit
    RO = 5
    #: Stochastic rounding (``StochasticA``), with :math:`R` a uniformly random
    #: integer of :math:`N` = ``prng_bits`` bits. Let :math:`\eta` be the
    #: position of :math:`|x|` between its two neighbours, as a fraction of the
    #: gap between them. Then :math:`x` rounds away from zero if and only if
    #: :math:`\lfloor \eta 2^N \rfloor + R \ge 2^N`
    SR = 6
    #: ``StochasticB``: rounds away from zero if and only if
    #: :math:`\lfloor \eta 2^{N+1} \rfloor + 2R + 1 \ge 2^{N+1}`
    SR_B = 7
    #: ``StochasticC``: rounds away from zero if and only if
    #: :math:`\mathrm{RNITE}(\eta 2^N) + R \ge 2^N`, where RNITE rounds to the
    #: nearest integer, ties to even
    SR_C = 8
    SR_A = 6  #: ``StochasticA``: another name for ``SR`` (an alias)

    @property
    def is_stochastic(self) -> bool:
        """Whether this mode draws random bits, as ``SR`` (``SR_A``), ``SR_B`` and ``SR_C`` do."""
        return self.value >= 6


class AccumulateAlgorithm(Enum):
    """How a dot product adds up its products.

    Mirrors ``AccumulateAlgorithm`` in ``csrc/common/modes.h``, by name and
    value. It selects the summation algorithm inside a quantized GEMM, as set
    up by :class:`mptorch.quant.SplitMac`, :class:`mptorch.quant.FusedMac` or
    the flat ``*_matmul`` wrappers.

    Notation: each step of the dot product multiplies ``a`` by ``b`` and adds
    the product to the running sum ``s``. ``round_mul`` rounds to the
    multiply format and ``round_acc`` to the accumulate format; ``round_acc``
    does nothing when there is no accumulate format. A fused multiply-add
    (:class:`mptorch.quant.FusedMac`) rounds once, to its fused format, so its
    step ``round_acc(fma(a, b, s))`` replaces
    ``round_acc(s + round_mul(a * b))``. ``round_out`` rounds to the *outer*
    format that ``BLOCK`` and ``TREE`` use between blocks, and does nothing
    when there is none. The steps run in order, ``k = 1 .. K``:

    * :attr:`NAIVE`: ``s = round_acc(s + round_mul(a * b))``, which is what a
      hardware accumulator in the accumulate format computes. The result is
      ``s``.
    * :attr:`KAHAN`: Kahan's compensated summation. A correction ``c``, also
      in the accumulate format, carries the rounding error of each addition
      into the next step: ``y = round_acc(round_mul(a * b) - c)``,
      ``t = round_acc(s + y)``, ``c = round_acc(round_acc(t - s) - y)``,
      ``s = t``. The result is ``s``. Without an accumulate format this is
      ordinary Kahan summation in the carrier.
    * :attr:`BLOCK`: each run of ``block_size`` consecutive products is summed
      into ``s`` with the ``NAIVE`` step. At the end of each block (and after
      the last product) its sum is added to a total,
      ``tot = round_out(tot + s)``, and ``s`` restarts from zero. This allows
      two precisions: a narrow accumulate format inside each block, and a
      wider outer format across blocks. The result is ``tot``.
    * :attr:`TREE`: the rounded products ``p1, p2, ...`` of each block of
      ``block_size = 2**L`` are added in pairs, the pair sums in pairs, and so
      on, as a binary tree over the order in which they arrive:
      ``round_acc(round_acc(p1 + p2) + round_acc(p3 + p4))`` for four
      products. The block's sum is then added to the total as in ``BLOCK``. If
      the last block is incomplete, the complete subtrees it has are merged,
      smallest first. ``TREE`` needs each product as a separate term, so a
      fused multiply-add cannot use it.

    Under :attr:`RoundMode.SR`, every rounding to a format draws one random
    number, in the order the formulas above are written. The kernels never
    pad ``K`` with zero products, so a result does not depend on how a
    backend splits ``K`` into tiles: the CPU, CUDA and MPS kernels give
    bit-identical results in every deterministic rounding mode, except that
    the Apple GPU (MPS) flushes binary32 subnormals to zero in the arithmetic
    between roundings. Under SR an MPS result also equals the CPU's, since
    both draw their random bits from the CPU generator. ``KAHAN``, ``BLOCK``
    and ``TREE`` are available for the single-format and cross-family GEMMs,
    the convolutions and the block GEMM, in both carriers (only binary32 on
    MPS, which has no float64). A palette (a format per output element)
    supports only ``NAIVE``.

    Example::

        >>> from mptorch import AccumulateAlgorithm
        >>> AccumulateAlgorithm.NAIVE.value
        0
    """

    NAIVE = 0  #: Round the running sum after every addition.
    KAHAN = 1  #: Kahan's compensated summation, the correction in the accumulate format.
    #: Sum blocks in the accumulate format, and the block sums in the outer
    #: format (as in FABsum).
    BLOCK = 2
    #: Sum each block as a binary tree, and the block sums in the outer format.
    TREE = 3
