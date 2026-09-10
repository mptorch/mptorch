Formats and rounding
====================

This page is the theory the rest of the documentation relies on: what a
number format is, which formats MPTorch implements, what each rounding mode
does, and the equations the kernels follow. Everything here is exercised by
a run.

The simulation model
--------------------

MPTorch never computes *in* a narrow format. Every operation -- a product, an
addition, an activation -- is carried out in float32, and its result is then
**rounded** to the target format :math:`F` with a rounding mode :math:`\circ`:

.. math::

   \tilde{x} = Q_{F,\circ}(x).

For an elementwise step (rounding a weight tensor, say) that is the whole
story, and it is what the quantizers of :doc:`quantizers` do. For a
*reduction* -- a dot product -- the rounding can be applied to every
intermediate result, and which intermediates are rounded, to what, is the
arithmetic that :doc:`gemm` simulates:

.. math::

   c_{ij} = \sum_{k=1}^{K} a_{ik}\, b_{kj}
   \quad\longrightarrow\quad
   s_{k} = Q_{\text{acc}}\big(s_{k-1} + Q_{\text{mul}}(a_{ik} b_{kj})\big).

Because float32 has far more precision and range than any of the simulated
formats, computing in float32 and rounding once gives the correctly-rounded
result of the simulated operation -- the same number a machine working
natively in :math:`F` would produce.

BinaryK: the IEEE P3109 formats
-------------------------------

:class:`~mptorch.BinaryK` is the family of binary floating-point formats
defined by IEEE P3109, the upcoming *Standard for Arithmetic Formats for
Machine Learning* (`Fitzgibbon, Wintersteiger and Sarnoff
<https://arxiv.org/abs/2606.04028>`__ give an overview of the draft). P3109
parameterizes a format by its bitwidth :math:`K`, its precision :math:`P`,
its signedness and its domain -- whether it has infinities -- and names it
after them: ``Binary8p4se`` is the 8-bit signed (``s``) format with 4 bits
of precision in the extended (``e``) domain, and ``Binary8p4sf`` is its
finite-domain (``f``) twin. The bits are laid out the way IEEE 754 lays out a
binary float. A value has a sign :math:`s`, a biased exponent :math:`e` of
:math:`E` bits and a stored mantissa :math:`m` of :math:`P-1` bits; the
:math:`P`-th bit of precision is the implicit leading one. Its value is

.. math::

   x = (-1)^s \cdot \Big(1 + \frac{m}{2^{P-1}}\Big) \cdot 2^{\,e - \text{bias}}
   \qquad (1 \le e \le 2^E - 1)

for *normal* numbers, and

.. math::

   x = (-1)^s \cdot \frac{m}{2^{P-1}} \cdot 2^{\,1 - \text{bias}}
   \qquad (e = 0)

for *subnormal* numbers, which fill the gap between zero and the smallest
normal with the same absolute spacing. The two parameters you give are the
total width :math:`K` and the precision :math:`P`; the exponent width follows
as :math:`E = K - P` for a signed format and :math:`E = K - P + 1` for an
unsigned one, which has no sign bit to spend. ``is_signed`` is P3109's
signedness, and the domain comes with the saturation mode, below.

A few consequences worth having in mind:

- Within one *binade* :math:`[2^q, 2^{q+1})` there are :math:`2^{P-1}`
  representable values, spaced :math:`2^{q-P+1}` apart -- the *unit in the
  last place*, ulp. The relative spacing is therefore between
  :math:`2^{-P}` and :math:`2^{1-P}`: E4M3 (:math:`P=4`) represents numbers
  to about 6-12 %, E5M2 (:math:`P=3`) to about 12-25 %.
- The largest finite value is
  :math:`(2 - 2^{2-P}) \cdot 2^{\,2^E - 1 - \text{bias}}` in P3109's
  extended domain (``OVF_INF``, the default, and ``SAT_PROPAGATE``), where the
  very last code of the top binade is :math:`\infty`, and
  :math:`(2 - 2^{1-P}) \cdot 2^{\,2^E - 1 - \text{bias}}` in its finite
  domain (``SAT_FINITE``), where that code is a number. Those two are the
  signed formats with :math:`P \ge 3`; `Relation to IEEE P3109`_ has the rule
  for the rest.
- The smallest normal is :math:`2^{1 - \text{bias}}` and the smallest
  subnormal :math:`2^{1 - \text{bias} - (P-1)}`.
- ``bias`` defaults to P3109's, the middle of the exponent range:
  :math:`2^{E-1}`, that is :math:`2^{K-P-1}` signed and :math:`2^{K-P}`
  unsigned, which encodes 1.0 at the middle code point. IEEE 754 would give
  the same exponent width :math:`2^{E-1} - 1`, one less; P3109 fixes the bias
  rather than the largest exponent so that adding or removing the infinities
  moves no finite value. Pass ``bias=`` explicitly for a format outside
  P3109, such as the OCP 8-bit formats. A BinaryK format uses every exponent
  code for finite values where IEEE reserves the top one for infinities and
  NaNs, so with the IEEE bias it agrees with its namesake on every value up
  to the namesake's largest finite one (the :doc:`tutorial` checks this
  exhaustively for E4M3 and E5M2).

Some formats you may want, as BinaryK values -- first from P3109, then from
outside it:

=====================  ============================================  ======  ======  ======
format                 spelling                                      E       P-1     bias
=====================  ============================================  ======  ======  ======
``Binary8p4se``        ``BinaryK(8, 4)``                             4       3       8
``Binary8p4sf``        ``BinaryK(8, 4, saturation=SAT_FINITE)``      4       3       8
``Binary8p3se``        ``BinaryK(8, 3)``                             5       2       16
``Binary6p3se``        ``BinaryK(6, 3)``                             3       2       4
E4M3 (OCP / NVIDIA)    ``BinaryK(8, 4, bias=7)``                     4       3       7
E5M2 (OCP / NVIDIA)    ``BinaryK(8, 3, bias=15)``                    5       2       15
float16                ``BinaryK(16, 11, bias=15)``                  5       10      15
bfloat16               ``BinaryK(16, 8, bias=127)``                  8       7       127
float32                ``BinaryK(32, 24, bias=127)``                 8       23      127
=====================  ============================================  ======  ======  ======

The following run constructs P3109's two signed 8-bit formats with 4 and 3
bits of precision, derives their ranges from the formulas above, and confirms
them against the quantizer in both domains -- each boundary value is a fixed
point of the rounding in the domain that has it, and :math:`10^{30}`
overflows to :math:`\infty` in the extended domain and to the largest finite
value in the finite one.

.. literalinclude:: ../snippets/formats_binaryK.py
   :language: python
   :caption: docs/snippets/formats_binaryK.py

.. literalinclude:: ../snippets/formats_binaryK.out
   :language: text
   :caption: output

The last two lines show the grid: in :math:`[1, 2)` ``Binary8p4`` has the eight values
:math:`1, 1.125, \dots, 1.875`, and each input lands on the nearest one, ties
going to the even mantissa (:math:`1.0625 \to 1.0`, :math:`1.1875 \to 1.25`,
:math:`1.9375 \to 2.0`).

Subnormals, saturation and sign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two enums decide what happens at the ends of the range. Both are properties
of the *format*, so two formats in one computation can differ in them.

:class:`~mptorch.SubnormalsMode` -- the bottom of the range:

``SUBNORMALS``
   Gradual underflow, as in IEEE 754 and in P3109, whose formats all have
   subnormals once :math:`P > 1`: the exponent code 0 encodes the subnormal
   numbers, and a value below the smallest of them rounds to zero (or to it,
   under a directed mode).
``NORMALS``
   No subnormals, which takes the format outside P3109. Everything below the
   smallest normal, :math:`2^{1-\text{bias}}`, flushes to zero.
``EXTENDED_NORMALS``
   No subnormals either, but the exponent code they would have used encodes
   one more binade of *normal* numbers, :math:`[2^{-\text{bias}}, 2^{1-\text{bias}})`,
   with the full mantissa. Below that binade, flush to zero. Not P3109
   either.

:class:`~mptorch.SaturationMode` -- the top. The modes are P3109's saturation
modes, and they choose its domain as well: a finite-domain P3109 format admits
only ``SatFinite``, so ``SAT_FINITE`` is that domain and the other two are the
extended one.

``SAT_FINITE``
   P3109's ``SatFinite``, in the finite domain. Everything is clamped to the
   largest finite value, which here includes the top mantissa code: a finite
   input that overflows *and* an infinite input alike, so the output never
   contains an infinity. NaN inputs pass through unchanged.
``SAT_PROPAGATE``
   P3109's ``SatPropagate``, in the extended domain. A finite input that
   overflows is clamped to the largest finite value with the top mantissa
   code excluded -- one that rounds onto that code included, like 480 in the
   run below -- and an infinite input stays infinite.
``OVF_INF``
   P3109's ``SatNone``, in the extended domain, and the default: an input
   beyond the largest finite value becomes :math:`\pm\infty`. P3109 rounds
   before it saturates, so this happens under every rounding mode, including
   the ones for which IEEE 754 stops at the largest finite value, such as
   rounding toward zero (the last line of the run).

.. literalinclude:: ../snippets/formats_modes.py
   :language: python
   :caption: docs/snippets/formats_modes.py

.. literalinclude:: ../snippets/formats_modes.out
   :language: text
   :caption: output

An unsigned format (``is_signed=False``) has no sign bit; negative inputs
become zero and the bit goes to the exponent, which is why the unsigned
``BinaryK(8, 4)`` above has P3109's unsigned bias :math:`2^{K-P} = 16` and
reaches 3072.

Relation to IEEE P3109
~~~~~~~~~~~~~~~~~~~~~~

The correspondence in one place:

.. list-table::
   :header-rows: 1

   * - P3109
     - MPTorch
   * - bitwidth :math:`K`, precision :math:`P`, signedness
     - ``BinaryK(K, P, is_signed=...)``
   * - extended / finite domain
     - ``saturation``: ``OVF_INF`` or ``SAT_PROPAGATE`` / ``SAT_FINITE``
   * - exponent bias :math:`2^{K-P-1}`, unsigned :math:`2^{K-P}`
     - the default ``bias``
   * - ``SatNone``, ``SatPropagate``, ``SatFinite``
     - ``OVF_INF``, ``SAT_PROPAGATE``, ``SAT_FINITE``
   * - ``NearestTiesToEven``, ``NearestTiesToAway``
     - ``RNE``, ``RNA``
   * - ``TowardPositive``, ``TowardNegative``, ``TowardZero``
     - ``RU``, ``RD``, ``RZ``
   * - ``ToOdd``
     - ``RO``
   * - ``StochasticA`` with :math:`N` random bits
     - ``SR`` with ``prng_bits=N``

The top of the range is counted in P3109's code points. Of the top binade's
:math:`2^{P-1}` codes, the extended domain spends the last on :math:`+\infty`;
an unsigned format spends its last on NaN and, in the extended domain, the one
below that on :math:`+\infty`. The largest finite value is the highest code
left -- 224 for ``Binary8p4se``, 240 for ``Binary8p4sf``, 53248 for
``Binary8p4ue`` -- and with :math:`P \le 2` the reserved codes can outnumber
the top binade's, which moves it a binade or two lower. A finite result above
it saturates whatever the rounding mode.

Two test files hold this against the standard. ``tests/test_binaryk_quantize.py``
checks the six deterministic modes against gfloat, a reference implementation
of P3109, below the top of the range. ``tests/test_binaryk_p3109.py`` checks the
top itself -- every precision of the 3- to 8-bit formats, signed and unsigned,
in all three saturation modes -- against a transcription of the paper's
definitions, since gfloat overflows the IEEE 754 way under directed rounding.

MPTorch departs from the standard in two ways.

**By extension.** A ``bias`` other than the default, and the ``NORMALS`` and
``EXTENDED_NORMALS`` subnormal modes, give formats P3109 does not define; so
do the widths it excludes (it requires :math:`K \ge 3`, and :math:`P < K` for
a signed format). They are what reach E4M3, bfloat16 and the rest.

**By simulating values rather than code points.** A result is a float32
number, not an encoding: P3109's single NaN is whichever NaN came in, and its
single, unsigned zero can come out as :math:`-0.0` under ``RU``, ``RZ`` and
``RO`` (the two compare equal). And P3109's extended domain with
``SatFinite`` -- a format that has infinities, clamping an infinite input to
its largest finite value -- has no spelling, because ``SAT_FINITE`` also
selects the finite domain.

SuperFP: precision at the top, range below
------------------------------------------

:class:`~mptorch.SuperFP` is the second format family the kernels implement.
It has the same three fields as a binary float -- ``man_bits``, ``exp_bits``,
``bias`` -- but only the top ``normal_binades`` binades carry a mantissa. The
encodings of the remaining :math:`2^{\text{exp_bits}} - \text{normal_binades}`
binades, :math:`2^{\text{man_bits}}` codes each, are reinterpreted as that
many further *powers of two* below the normal region: numbers with an
implicit mantissa of exactly 1. Writing :math:`e_{\max} = 2^{\text{exp_bits}} - 1 - \text{bias}`,
the three regions are

.. math::

   \begin{aligned}
   \text{normal:}      &\quad [2^{\,e_{\max} - b + 1},\; 2^{\,e_{\max}+1}),
      && \text{full mantissa} \\
   \text{supernormal:} &\quad [2^{\,e_{\max} - b + 1 - n},\; 2^{\,e_{\max} - b + 1}),
      && \text{powers of two only} \\
   \text{underflow:}   &\quad \text{below that}, && \text{flushed to zero}
   \end{aligned}

with :math:`b = \text{normal_binades}` and
:math:`n = (2^{\text{exp_bits}} - b)\, 2^{\text{man_bits}}`. There are no
subnormals, and the bias has no default: it is part of the design of the
format and must be given. Rounding in the supernormal region is rounding of
the *exponent*: to nearest, :math:`3 = 2^{1.58}` becomes :math:`4`.

.. literalinclude:: ../snippets/formats_superfp.py
   :language: python
   :caption: docs/snippets/formats_superfp.py

.. literalinclude:: ../snippets/formats_superfp.out
   :language: text
   :caption: output

With the same 8 bits as E4M3, this format reaches down to :math:`2^{-104}`
instead of :math:`2^{-9}`, at the price of keeping its mantissa only in the
two top binades. It is a format for quantities whose *scale* matters more
than their digits.

Rounding modes
--------------

Every quantizer and every GEMM takes a :class:`~mptorch.RoundMode`. Writing
:math:`\lfloor x \rfloor_F` and :math:`\lceil x \rceil_F` for the nearest
representable values below and above :math:`x` (so
:math:`\lfloor x \rfloor_F \le x \le \lceil x \rceil_F`, with equality when
:math:`x` is representable):

=======  ===================================  ==========================================================
mode     name                                 :math:`Q(x)`
=======  ===================================  ==========================================================
``RNE``  round to nearest, ties to even       the nearer of the two; on a tie, the one with even mantissa
``RNA``  round to nearest, ties away          the nearer of the two; on a tie, the larger in magnitude
``RU``   round up (toward :math:`+\infty`)    :math:`\lceil x \rceil_F`
``RD``   round down (toward :math:`-\infty`)  :math:`\lfloor x \rfloor_F`
``RZ``   round toward zero (truncate)         :math:`\lfloor x \rfloor_F` if :math:`x > 0`, :math:`\lceil x \rceil_F` if :math:`x < 0`
``RO``   round to odd                         :math:`x` if representable; else the neighbour with odd mantissa
``SR``   stochastic rounding                  :math:`\lceil x \rceil_F` with probability :math:`p(x)`, else :math:`\lfloor x \rfloor_F`
=======  ===================================  ==========================================================

``RNE`` is what IEEE arithmetic does by default and is unbiased on average.
``RZ`` is what cheap hardware does; it is biased toward zero. ``RO`` is not a
mode you would compute in, but rounding to odd at an intermediate width and
then to nearest at the final one avoids the *double rounding* error that two
nearest roundings can commit.

.. literalinclude:: ../snippets/rounding_modes.py
   :language: python
   :caption: docs/snippets/rounding_modes.py

.. literalinclude:: ../snippets/rounding_modes.out
   :language: text
   :caption: output

Stochastic rounding
~~~~~~~~~~~~~~~~~~~

Under ``SR`` the rounding direction is random, with a probability
proportional to how far :math:`x` sits between its two neighbours. Writing
:math:`f = (x - \lfloor x \rfloor_F) / (\lceil x \rceil_F - \lfloor x \rfloor_F) \in [0, 1)`
for that fraction, the ideal rule is :math:`P(\text{up}) = f`, which makes
the rounding *unbiased*: :math:`\mathbb{E}[Q(x)] = x`.

The implementation draws ``prng_bits`` random bits and adds them just below
the last kept mantissa bit, then truncates -- so with :math:`p` random bits
the probability is quantized to steps of :math:`2^{-p}`:

.. math::

   P(\text{up}) = 1 - \frac{\lceil (1 - f)\, 2^{p} \rceil}{2^{p}}
   \;\xrightarrow{\;p \to \infty\;}\; f .

This is P3109's ``StochasticA`` rounding with :math:`N = p` random bits:
writing :math:`\eta` for the fraction of :math:`|x|` between its neighbours
(:math:`\eta = f` when :math:`x > 0`), it rounds away from zero when
:math:`\lfloor \eta\, 2^{N} \rfloor + R \ge 2^{N}` for a uniform integer
:math:`R \in [0, 2^{N})`.

``prng_bits`` lives on the format (``BinaryK(8, 4, prng_bits=8)``) and is
ignored by every other rounding mode. Two practical constraints follow from
where the bits are drawn: with ``prng_bits=0`` there is nothing random and
``SR`` degenerates into ``RZ`` (the ``SR`` row above, from a format with no
random bits, is identical to the ``RZ`` row); and the format's mantissa plus
its random bits must fit inside the mantissa of the tensor's dtype -- 23 for
float32, 10 for float16, 7 for bfloat16 -- which the quantizer checks.

The random streams are seeded from PyTorch's default generator, so
``torch.manual_seed`` makes a stochastic run reproducible. In a GEMM each
output element owns its own stream, keyed on its position, so the result does
not depend on how the work was split across threads.

.. literalinclude:: ../snippets/rounding_stochastic.py
   :language: python
   :caption: docs/snippets/rounding_stochastic.py

.. literalinclude:: ../snippets/rounding_stochastic.out
   :language: text
   :caption: output

The first table shows the probability converging to the exact fraction as the
random bits increase (with 1-3 bits, :math:`\lceil 0.4 \cdot 2^p \rceil / 2^p`
is :math:`1/2`, so the coin is fair whatever the residual). The second shows
why anyone accepts the extra variance: an accumulator whose increments are
smaller than half its spacing never moves under round-to-nearest -- the sum
of two thousand additions of :math:`0.001` to :math:`1.0` in a bfloat16-like
format is still :math:`1.0` -- while stochastic rounding advances it by the
right amount *on average*. This is the situation of a small weight update
against a large weight, and the :doc:`tutorial` shows it deciding whether a
network trains.

Where rounding is applied
-------------------------

A quantizer rounds a tensor; the arithmetic rounds inside a reduction. The
two compose, and MPTorch keeps them separate on purpose:

- :doc:`quantizers` -- ``binaryK_quantize``, ``superfp_quantize``,
  :class:`~mptorch.quant.Quant`, :class:`~mptorch.quant.Quantizer`: one
  rounding per element.
- :doc:`gemm` -- :class:`~mptorch.quant.SplitMac`,
  :class:`~mptorch.quant.FusedMac`, ``qmatmul``: one rounding per product
  and per addition, or per fused multiply-add.
- :doc:`layers` -- ``QAffineFormats`` and the layers: which quantizer is
  applied to which signal, and which arithmetic each of the layer's three
  matrix products runs in.
