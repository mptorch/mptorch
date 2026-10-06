Formats and rounding
====================

This page presents the fundamentals that the rest of the documentation relies on:
what a number format is, which formats MPTorch implements -- scalar ones, and
block formats that share a scale among their elements -- what each rounding
mode does, which arithmetic the simulation rounds in, and the equations the
kernels follow. Everything here is exercised by a run.

The simulation model
--------------------

MPTorch never computes *in* a narrow format. Every operation -- a product, an
addition, an activation -- is carried out in an IEEE binary float, the
*carrier*, and its result is then **rounded** to the target format :math:`F`
with a rounding mode :math:`\circ`:

.. math::

   \tilde{x} = Q_{F,\circ}(x).

For an elementwise step (rounding a weight tensor, say) that is the whole
story, and it is what the quantizers do (see :doc:`quantizers`). For a
*reduction* operation -- a dot product -- the rounding can be applied to every
intermediate result, and which intermediates are rounded, to what, is the
arithmetic that the provided kernels (see :doc:`kernels/index`) simulate. In
case of computing one output element when multiplying two matrices, this can 
look like:

.. math::

   c_{ij} = \sum_{k} a_{ik}\, b_{kj}
   \quad\longrightarrow\quad
   s_{k} = Q_{\text{acc}}\big(s_{k-1} + Q_{\text{mul}}(a_{ik} b_{kj})\big).

Because the carrier (see Carriers_) has far more precision and range than the simulated
format, computing in it and rounding once usually gives the correctly-rounded result
of the simulated operation -- the same number a machine working natively in
:math:`F` would produce -- for every format the carrier can hold.

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

A few consequences are worth having in mind:

- Within one *binade* :math:`[2^q, 2^{q+1})` there are :math:`2^{P-1}`
  representable values, spaced :math:`2^{q-P+1}` apart -- the *unit in the
  last place*, ulp. The relative spacing is therefore between
  :math:`2^{-P}` and :math:`2^{1-P}`: neighbouring values are about 6-12 %
  apart with :math:`P = 4` (E4M3's precision) and 12-25 % apart with
  :math:`P = 3` (E5M2's), and rounding to nearest moves a value by at most
  half of that.
- The largest finite value is
  :math:`(2 - 2^{2-P}) \cdot 2^{\,2^E - 1 - \text{bias}}` in P3109's
  extended domain (``OVF_INF``, the default, and ``SAT_PROPAGATE``), where the
  very last code of the top binade is :math:`\infty`, and
  :math:`(2 - 2^{1-P}) \cdot 2^{\,2^E - 1 - \text{bias}}` in its finite
  domain (``SAT_FINITE``), where that code is a number. Both are for a signed
  format with :math:`P \ge 3` (see `Relation to IEEE P3109`_ for the rules for
  the other formats).
- The smallest normal is :math:`2^{1 - \text{bias}}` and the smallest
  subnormal :math:`2^{1 - \text{bias} - (P-1)}`.
- ``bias`` defaults to P3109's, the middle of the exponent range:
  :math:`2^{E-1}`, that is :math:`2^{K-P-1}` for the signed format variant
  and :math:`2^{K-P}` for the unsigned one; either way 1.0 is encoded at the
  middle code point. For the same exponent width, IEEE 754's bias is
  :math:`2^{E-1} - 1`, one less than P3109's; P3109 fixes the bias rather
  than the largest exponent so that adding or removing the infinities moves
  no finite value. Pass ``bias=`` explicitly for a format outside P3109, such
  as the nearest spellings of the OCP 8-bit formats. A BinaryK format uses
  every exponent code for finite values where IEEE reserves the top one for
  infinities and NaNs, so with the IEEE bias a BinaryK format with the same
  exponent and mantissa widths as an IEEE 754 one (float16, float32) agrees
  with it on every value up to its largest finite one, and goes on a binade
  past it. That is why no BinaryK format *is* float16, bfloat16 or float32
  (see `The IEEE 754 dtypes are not P3109 formats`_). The same holds against
  the OCP formats, which the tutorial (see :doc:`tutorial`) checks exhaustively: E5M2 is
  laid out the IEEE way and is outrun likewise, while E4M3 spends its top
  exponent field on numbers and is matched on every finite value.

Some formats of interest can be approximately represented in terms of BinaryK:

===========================  ============================================  ======  ======  ======
format                       spelling                                      E       P-1     bias
===========================  ============================================  ======  ======  ======
``Binary8p4se``              ``BinaryK(8, 4)``                             4       3       8
``Binary8p4sf``              ``BinaryK(8, 4, saturation=SAT_FINITE)``      4       3       8
``Binary8p3se``              ``BinaryK(8, 3)``                             5       2       16
``Binary6p3se``              ``BinaryK(6, 3)``                             3       2       4
E4M3's layout (OCP, NVIDIA)  ``BinaryK(8, 4, bias=7)``                     4       3       7
E5M2's layout (OCP, NVIDIA)  ``BinaryK(8, 3, bias=15)``                    5       2       15
float16's layout             ``BinaryK(16, 11, bias=15)``                  5       10      15
bfloat16's layout            ``BinaryK(16, 8, bias=127)``                  8       7       127
float32's layout             ``BinaryK(32, 24, bias=127)``                 8       23      127
===========================  ============================================  ======  ======  ======

The last five rows have each format's field widths and bias, and none of them
*is* the format it is laid out as: whatever its bias, a BinaryK spends its
codes the way P3109 does. The E4M3 row comes closest -- it has exactly E4M3's
finite values and differs in the special ones, and so in what an overflow
returns -- and the other four have more values than their namesakes, a binade
past them (see `The IEEE 754 dtypes are not P3109 formats`_, which says how far
each is from the format it is named after). The two 8-bit rows are importable as
``mptorch.E4M3`` and ``mptorch.E5M2``; the MX block presets build on them with
OCP's largest values and NaN codes (see `Block formats`_).

The following run constructs P3109's two signed 8-bit formats with 4 and 3
bits of precision, derives their ranges from the formulas above, and confirms
them against the quantizer in both domains -- each boundary value is a fixed
point of the rounding in the domain that has it, and, for example, :math:`10^{30}`
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

The IEEE 754 dtypes are not P3109 formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

float16, bfloat16 and float32 are not exactly representable as P3109 formats --
nor as BinaryK formats with any other bias -- in either domain, because the two
standards spend the code points of the same layout differently:

- IEEE 754 reserves the whole top exponent field -- :math:`2^{P-1}` codes of
  each sign -- for :math:`\pm\infty` and the NaNs, and gives zero two
  encodings, :math:`+0.0` and :math:`-0.0`.
- P3109 spends one code of each sign on :math:`\pm\infty`, and only in the
  extended domain, and gives every other code of the top field a finite
  value. A signed format has a single, unsigned zero, and the encoding IEEE
  754 would read as :math:`-0.0` is its one NaN.
- P3109's default bias, :math:`2^{E-1}`, is one more than IEEE 754's for the
  same exponent width (:math:`2^{E-1}-1`).

So a bias can line up at most one end of the range with that of the closest IEEE 754 dtype, and neither special-value set ever matches between the formats. For float16, we have for instance:

.. list-table::
   :header-rows: 1

   * - format
     - largest finite value
     - smallest positive value
     - against float16's finite values
   * - float16
     - 65504
     - :math:`2^{-24}`
     -
   * - ``BinaryK(16, 11, bias=15)``
     - 130944
     - :math:`2^{-24}`
     - all of them, and 1023 more in :math:`[2^{16}, 130944]`
   * - ``Binary16p11se``, ``BinaryK(16, 11)``
     - 65472
     - :math:`2^{-25}`
     - all but 65504, whose code is :math:`+\infty`, and 1024 more below
       :math:`2^{-14}`
   * - ``Binary16p11sf``
     - 65504
     - :math:`2^{-25}`
     - all of them, and 1024 more below :math:`2^{-14}`; no :math:`\pm\infty`

None of the three has float16's :math:`-0.0` or its 2046 NaN codes; each has
one zero and one NaN. bfloat16 and float32 differ from their spellings in the
same way. With the IEEE 754 bias the BinaryK format reaches a binade past the
dtype, to about :math:`6.8 \cdot 10^{38}` against :math:`3.4 \cdot 10^{38}`,
with :math:`2^{P-1} - 1` extra values (:math:`127` for bfloat16, :math:`2^{23} - 1` 
for float32). With P3109's bias, ``Binary16p8`` and ``Binary32p24`` lose the
dtype's largest value to :math:`+\infty` in the extended domain, match it in
the finite one, and add :math:`2^{P-1}` values below the dtype's smallest
normal, down to :math:`2^{-134}` and :math:`2^{-150}`.

The two OCP 8-bit formats relate to their BinaryK rows differently.

**E5M2** reserves its top exponent field as IEEE 754 does, so
``BinaryK(8, 3, bias=15)`` is to it what ``BinaryK(16, 11, bias=15)`` is to
float16 above: it has every finite value of E5M2 and a binade more. E5M2's
largest finite value is :math:`57344`; the BinaryK format adds :math:`65536`,
:math:`81920` and :math:`98304` of each sign, and under ``SAT_FINITE`` also
:math:`114688`, the code that is :math:`\pm\infty` in the extended domain.

**E4M3** spends its top exponent field on finite values, as P3109 does, so
``BinaryK(8, 4, bias=7)`` has exactly E4M3's finite values, up to :math:`448`.
Only the special codes differ: the BinaryK format has :math:`\pm\infty` where
E4M3 has its two NaNs, and its one NaN where E4M3 has :math:`-0`. That shows
in what an overflow returns. E4M3 has no infinity, and PyTorch's conversion to
``torch.float8_e4m3fn`` saturates at :math:`\pm 448`;
``BinaryK(8, 4, bias=7)`` returns

- :math:`\pm\infty` under ``OVF_INF``, the default.
- :math:`\pm 448` under ``SAT_PROPAGATE``, which is PyTorch's result for
  every finite float32 input, except that a zero is always :math:`+0`.
- :math:`\pm 480` under ``SAT_FINITE``, whose domain turns the infinity's
  code into a number that E4M3 does not have.

In MPTorch these differences show up in two places:

- Quantizing to one of the dtype-layout rows is not the dtype's own
  conversion. A value between the dtype's largest finite value and the row's
  stays finite where the dtype would store :math:`\pm\infty`, and :math:`-0.0`
  comes back as :math:`+0.0`.
- The range checks flag these formats. ``BinaryK(16, 8, bias=127)`` and
  ``BinaryK(32, 24, bias=127)`` reach :math:`2^{128}` and warn, on a float32
  tensor, that they outrun binary32. ``Binary16p8`` and ``Binary32p24`` warn
  about their bottom there (see `What the carrier can hold`_).
  A float64 tensor holds all four. And
  ``BinaryK(16, 11, bias=15)``, as the format a GEMM's result holds over
  float16 operands, warns that it reaches past float16
  (see `What a narrower tensor can store`_).

Subnormals, saturation and sign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two enums decide what happens at the ends of the range. Both are properties
of the *format*, so two formats in one computation can differ in them.

:class:`~mptorch.SubnormalsMode` -- the bottom of the range:

``SUBNORMALS``
   Gradual underflow, as in IEEE 754 and in P3109, whose formats all have
   subnormals once :math:`P > 1`: the exponent code 0 encodes the subnormal
   numbers, down to :math:`2^{1-\text{bias}-(P-1)}`.
``NORMALS``
   No subnormals, which takes the format outside P3109. The exponent code 0
   encodes only the zero, and the smallest value is the smallest normal,
   :math:`2^{1-\text{bias}}`.
``EXTENDED_NORMALS``
   No subnormals either, but the exponent code they would have used encodes
   one more binade of *normal* numbers, above :math:`2^{-\text{bias}}`, with
   the full mantissa. Its first code is still the zero, so the smallest value
   is one step above the power of two, :math:`(1 + 2^{1-P})\, 2^{-\text{bias}}`.
   Not P3109 either.

Below its smallest value every mode rounds alike: to zero or to that value,
as the rounding mode decides (see `Relation to IEEE P3109`_ for the rule). A
value is flushed to zero only where the rounding mode says so -- under
round-to-nearest, when it is nearer to zero than to the smallest value.

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
   that rounds past the largest finite value becomes :math:`\pm\infty`. P3109 rounds
   before it saturates, so this happens under every rounding mode, including
   the ones for which IEEE 754 stops at the largest finite value, such as
   rounding toward zero (the run's ``OVF_INF, RZ`` line).

.. literalinclude:: ../snippets/formats_modes.py
   :language: python
   :caption: docs/snippets/formats_modes.py

.. literalinclude:: ../snippets/formats_modes.out
   :language: text
   :caption: output

An unsigned format (``is_signed=False``) has no sign bit: negative inputs
become zero, and the freed bit goes to the exponent field. The run's last
line uses the unsigned ``BinaryK(8, 4)``, P3109's ``Binary8p4ue``, which has
five exponent bits and the default bias :math:`2^{K-P} = 16`, where the signed
``Binary8p4se`` has four and :math:`2^{K-P-1} = 8`. Its normal exponents
therefore run from :math:`-15` to :math:`15` instead of :math:`-7` to
:math:`7`, and its largest finite value is :math:`1.625 \cdot 2^{15} = 53248`
against the signed format's :math:`1.75 \cdot 2^{7} = 224` (the unsigned top
binade ends in two reserved codes, :math:`+\infty` and NaN, the signed one in
one). The extra bit buys range, not precision: both formats have
:math:`P = 4`, so around :math:`3000` the unsigned format's values are 256 apart, 
and the run's input :math:`3000` comes back as :math:`3072`, the nearest of them.

Relation to IEEE P3109
~~~~~~~~~~~~~~~~~~~~~~

The sections above brought in P3109 one notion at a time: how it names a
format after its parameters, its default bias, its two domains and its three
saturation modes. This section puts them side by side and says in what sense a
:class:`~mptorch.BinaryK` *is* a P3109 format. It covers, in order:

- which MPTorch argument stands for which notion of the standard.
- the rule for the largest finite value, which the earlier sections stated
  only for a signed format with :math:`P \ge 3`.
- how the correspondence is tested.
- the two ways in which MPTorch departs from the standard.

Every parameter of a P3109 format, and each of its saturation and rounding
modes, has a counterpart among the arguments of ``BinaryK`` and MPTorch's
enums. A ``BinaryK(K, P)`` left at its defaults is therefore the P3109 format
with that bitwidth and precision, signed and in the extended domain, for every
width the standard admits. The rounding modes are only introduced further down 
(see `Rounding modes`_); their rows are listed here so that the whole correspondence 
is in one table.

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
left -- :math:`224` for ``Binary8p4se``, :math:`240` for ``Binary8p4sf``, 
:math:`53248` for ``Binary8p4ue`` -- and with :math:`P \le 2` the reserved codes 
can outnumber the top binade's, which moves it a binade or two lower. A finite 
result above it saturates whatever the rounding mode.

``dev/benchmarks/binaryK_values.py`` prints the whole code-point table for a
format -- every value, one per line, infinities and NaN included -- and with
``--verify`` quantizes each one to check the kernel returns it unchanged;
``dev/benchmarks/superfp_values.py`` is its SuperFP analgoue.

Two test files compare the formats to the standard. ``tests/test_binaryk_quantize.py``
checks the six deterministic modes against `gfloat <https://github.com/graphcore-research/gfloat>`__, a reference implementation
of P3109, below the top of the range. ``tests/test_binaryk_p3109.py`` checks the
top itself -- every precision of the 3- to 8-bit formats, signed and unsigned,
in all three saturation modes -- against a transcription of the paper's
definitions, since gfloat overflows the IEEE 754 way under directed rounding.

MPTorch departs from the standard in two ways.

**By extension.** A ``bias`` other than the default, and the ``NORMALS`` and
``EXTENDED_NORMALS`` subnormal modes, represent formats P3109 does not define; so
do the widths it excludes (it requires :math:`K \ge 3`, and :math:`P < K` for
a signed format). These are the options that allow one to reach E4M3's finite values,
and the nearest representations to E5M2 and the IEEE 754 dtypes (see `The IEEE 754 dtypes are not P3109 formats`_).

The two extra subnormal modes move the bottom of the range and nothing else.
Both concern the :math:`2^{P-1}` codes of each sign whose exponent field is
zero, which ``SUBNORMALS`` spends on the zero and the :math:`2^{P-1} - 1`
subnormals:

- ``NORMALS`` leaves all of them but the zero unused, so the smallest positive
  value is the smallest normal, :math:`2^{1-\text{bias}}`.
- ``EXTENDED_NORMALS`` spends them on one more binade of normals,
  :math:`[2^{-\text{bias}}, 2^{1-\text{bias}})`, directly below the smallest
  normal. The mantissa-zero code of that binade does not encode
  :math:`2^{-\text{bias}}`: it remains the zero (and, with the sign bit set,
  the NaN), exactly as in every other BinaryK format. The binade therefore
  holds :math:`2^{P-1} - 1` values, the smallest of which is one step above
  the power of two at its foot, :math:`(1 + 2^{1-P})\, 2^{-\text{bias}}`, and
  with :math:`P = 1` it holds none.

Write :math:`x_{\min}` for the smallest positive value of the format:
:math:`2^{1-\text{bias}-(P-1)}` under ``SUBNORMALS``, and the two values above
under the other modes. Below :math:`x_{\min}` all three modes round alike,
because the region has the same shape in each. An input :math:`x` with
:math:`0 < |x| < x_{\min}` has two candidates for its magnitude, zero and
:math:`x_{\min}`, and the rounding mode picks between them (see
`Rounding modes`_ for the modes):

- ``RNE`` and ``RNA`` take the nearer one. On a tie, :math:`|x| = x_{\min} / 2`,
  ``RNE`` takes the zero and ``RNA`` takes :math:`x_{\min}`.
- ``RU``, ``RD`` and ``RZ`` take the candidate in their own direction.
- ``RO`` takes :math:`x_{\min}`, the nonzero candidate.
- ``SR`` takes :math:`x_{\min}` with probability :math:`|x| / x_{\min}`, and
  the zero otherwise.

**By simulating values rather than code points.** A result is a value of
the carrier, not an encoding, so P3109's single NaN is whichever NaN came in. (Its
single zero is unsigned and gets no such latitude: every zero the kernels
return is :math:`+0.0`, whether it came from :math:`-0.0` or from a negative
value that rounded to zero, and the same holds for SuperFP.) And P3109's
extended domain with
``SatFinite`` -- a format that has infinities, clamping an infinite input to
its largest finite value -- has no spelling, because ``SAT_FINITE`` also
selects the finite domain.

SuperFP: precision at the top, range below
------------------------------------------

:class:`~mptorch.SuperFP` is the second format family the kernels implement.
It has the same three fields as a binary float -- ``man_bits``, ``exp_bits``,
``bias`` -- but only the top ``normal_binades`` binades carry a mantissa. The
encodings of the remaining :math:`2^{\text{exp\_bits}} - \text{normal\_binades}`
binades, :math:`2^{\text{man\_bits}}` codes each, are reinterpreted: the first
is the zero, and the others are that many further *powers of two* below the
normal region, numbers with an implicit mantissa of exactly 1. Writing :math:`e_{\max} = 2^{\text{exp\_bits}} - 1 - \text{bias}`,
the three regions are

.. math::

   \begin{aligned}
   \text{normal:}      &\quad [2^{\,e_{\max} - b + 1},\; 2^{\,e_{\max}+1}),
      && \text{full mantissa} \\
   \text{supernormal:} &\quad [2^{\,e_{\max} - b + 2 - n},\; 2^{\,e_{\max} - b + 1}),
      && \text{powers of two only} \\
   \text{underflow:}   &\quad \text{below that}, && \text{flushed to zero}
   \end{aligned}

with :math:`b = \text{normal\_binades}` and
:math:`n = (2^{\text{exp\_bits}} - b)\, 2^{\text{man\_bits}}` the number of
reinterpreted codes, of which :math:`n - 1` are supernormals. There are no
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

.. _block-formats:

Block formats
-------------

The formats above give every element an exponent of its own. A *block
format* shares one scale among a block of consecutive elements and stores
the elements as narrow codes, so a 4-bit element can carry values over the
whole range of a float32 tensor, one block at a time. The OCP Microscaling
(MX) formats and NVIDIA's NVFP4 are block formats, and they are what the
newest accelerators multiply natively.

A :class:`~mptorch.BlockFormat` names one: an element format, a scale format
and a block size, the element and the scale each a
:class:`~mptorch.BinaryK` or a :class:`~mptorch.SuperFP` (`Superfp elements
and scales`_, below). A tensor is split into blocks of ``block_size``
consecutive elements along one axis, its *packed* axis, and element
:math:`i` of block :math:`\kappa` stands for the value

.. math::

   x_i \;\approx\; e_i\, \sigma_\kappa ,

its element value :math:`e_i` times its block's scale :math:`\sigma_\kappa`:
a power of two with an integer exponent :math:`X_\kappa`,
:math:`\sigma_\kappa = 2^{X_\kappa}`, for an E8M0 scale, and for any other
scale the scale format's value :math:`s_\kappa` times a float32 scale per
tensor :math:`S_t`, :math:`\sigma_\kappa = s_\kappa S_t`, one float32 product. The
presets are OCP MX v1.0's and NVFP4:

=======================  ==============  ===============  =====  ==========  ==============
preset                   element         scale            block  largest     bytes/element
=======================  ==============  ===============  =====  ==========  ==============
``MXFP8_E4M3``           E4M3            E8M0             32     448         1.031
``MXFP8_E5M2``           E5M2            E8M0             32     57344       1.031
``MXFP6_E2M3``           E2M3            E8M0             32     7.5         0.781
``MXFP6_E3M2``           E3M2            E8M0             32     28          0.781
``MXFP4_E2M1``           E2M1            E8M0             32     6           0.531
``NVFP4``                E2M1            E4M3 x float32   16     6           0.562
=======================  ==============  ===============  =====  ==========  ==============

E8M0 is an unsigned exponent: code :math:`c` is the scale :math:`2^{c - 127}`,
and its code 255 is NaN. NVFP4's scale is an E4M3 value times a float32 scale
per tensor. The element formats' biases are OCP's (IEEE 754's), one below
P3109's default, which is why the presets spell them out (``E4M3 =
BinaryK(8, 4, bias=7)``); `Relation to IEEE P3109`_ says how
:class:`~mptorch.BinaryK` relates to the standard. How a block format is
stored, in the formats' own bytes, is :doc:`quantizers`' ("Block
quantization"); how two of them are multiplied is :doc:`kernels/block`'s.

How a block is quantized
~~~~~~~~~~~~~~~~~~~~~~~~

Each block's scale is chosen from its largest magnitude :math:`a` (NaNs
aside), by default as the format's own specification says (`Choosing the
scale`_, below, has the other rules), and every element of the block is
divided by it and rounded:

* **E8M0** (MX): :math:`s = 2^{\lfloor \log_2 a \rfloor - e_{\max}}`, where
  :math:`e_{\max} = \lfloor \log_2(\text{elem\_max}) \rfloor` is the exponent of
  the element format's largest value, clamped to E8M0's range, and code 0 for
  a block of zeros. This is OCP MX's rule, and dividing by a power of two is
  exact.
* **Any other scale** -- E4M3 with a tensor scale (NVFP4), or a superfp:
  :math:`s = \mathrm{RNE}_{\text{E4M3}}
  \big(a / (\text{elem\_max} \cdot S_t)\big)`, at most 448 and at least E4M3's
  smallest positive value, and an element is divided by :math:`s \cdot S_t`,
  in float32. :math:`S_t` defaults to :math:`\max|x| / (\text{elem\_max}
  \cdot 448)` over the tensor, so its largest block takes the largest scale;
  ``tensor_scale=`` gives a static one. A superfp scale rounds the same
  ratio, but by default takes the next scale up where rounding to nearest
  would saturate the block (below).

The element is then rounded by its format's :class:`~mptorch.BinaryK` cast in
the call's rounding mode, under ``SaturationMode.SAT_FINITE``, and clamped to
``elem_max``. The clamp is where OCP's formats part from P3109's: E4M3's
finite top is 480 and OCP spends that code on NaN, so ``MXFP8_E4M3`` has
``elem_max=448.0`` and ``nan_code=0x7F``. Every value above ``elem_max``
lands on it, in every rounding mode. The rounding mode applies to the
elements; the scale's rule never depends on it.

Decoding an element is one float32 product, its value times its block's
scale (times :math:`S_t`), which is exact for every OCP format. Two rules
are stated here once:

* **Non-finite values.** An infinity saturates to ``elem_max``. With an E8M0
  scale its block takes the largest scale, :math:`2^{127}`, and
  :math:`448 \cdot 2^{127}` is past binary32, so it decodes to an infinity
  again. A NaN takes the element's NaN code when it has one (E4M3, E5M2).
  Otherwise the block's *scale* becomes the scale's NaN and the whole block
  decodes to NaN, which for a 2D tile (below) is all of its up to 16,384
  elements. With neither a NaN code nor a scale, a NaN becomes
  ``elem_max``, a loss. OCP leaves the non-finite behaviour of the formats
  without one to the implementation; these are MPTorch's choices.
* **One zero.** An element code with only its sign bit set is OCP's
  :math:`-0`; MPTorch never writes it and reads it as :math:`+0`, so no block
  op returns ``-0.0``, like every other cast in the library.

.. _scale-rounding:

Choosing the scale
~~~~~~~~~~~~~~~~~~

The scale that takes a block's largest magnitude exactly onto ``elem_max`` is
:math:`r = a / \text{elem\_max}` (over :math:`S_t` for a cast scale), and it is
rarely a value of the scale format. A scale below :math:`r` puts the block's
largest element above ``elem_max``, where it saturates to ``elem_max``; a
scale above it leaves every other element of the block on a coarser grid.
``BlockFormat(..., scale_rounding=)`` takes one of four rules,
:class:`~mptorch.ScaleRounding`:

* ``OCP``: :math:`2^{\lfloor \log_2 a \rfloor - e_{\max}}`, OCP MX's rule, for a
  power-of-two scale only;
* ``NEAREST``: :math:`r` rounded to nearest even in the scale format, NVFP4's
  rule;
* ``UP``: the smallest scale at least :math:`r`, the rule NVIDIA's MXFP8
  pretraining recipe uses for E8M0 (arXiv 2506.08027), because saturated
  values hurt its convergence;
* ``SELECTIVE``: ``NEAREST``, or the next scale up where the block's largest
  element would then land above :math:`T`, ``elem_max`` plus half the element
  format's step above it: the largest value that rounding to nearest would
  still take to ``elem_max``. The clamp then never moves the block's largest
  element further than rounding it to nearest would.

From a power-of-two scale, where the four differ most, a block's largest
element lands in E2M1's units in

=============  ====================  =========================================
rule           :math:`a / s`         the block's largest element
=============  ====================  =========================================
``OCP``        :math:`[4, 8)`        saturates above 7, up to 25% low
``NEAREST``    :math:`[4.5, 9]`      saturates above 7, up to 33% low
``UP``         :math:`(3, 6]`        never saturates
``SELECTIVE``  :math:`(3.5, 7]`      within E2M1's own rounding
=============  ====================  =========================================

and ``UP`` pays for its guarantee where the largest element was going to
round to 6 anyway, with a grid twice as coarse for the rest of the block.
Where the scale has mantissa bits the choice matters less: ``NEAREST`` falls
at most half a step below :math:`r`, a factor of 1.0625 in E4M3's normals,
which E2M1's top step (4 to 6) absorbs, so ``SELECTIVE`` leaves NVFP4's
scales alone there. An element with a finer top step (E2M3, E4M3) saturates
by that factor too, and ``SELECTIVE`` moves those blocks' scales up.

The default is the scale's own rule: ``OCP`` for a power of two, ``NEAREST``
for a binaryK with a mantissa, and ``SELECTIVE`` for a superfp, whose values
below its normal binades are powers of two (`Superfp elements and scales`_).
The rule in effect is the format's ``scale_rule``, which formats compare by.
Quantizing a decoded tensor again changes nothing under ``OCP``, ``UP`` and
``SELECTIVE`` from a power of two. ``NEAREST`` from one can move values: a
largest element that rounded down gives a ratio that rounds a binade down,
under which it saturates. So can ``UP`` from a scale with mantissa bits, or
any rule from E4M3 with an element finer than E2M1.

.. literalinclude:: ../snippets/block_scale_rounding.py
   :language: python
   :caption: docs/snippets/block_scale_rounding.py

.. literalinclude:: ../snippets/block_scale_rounding.out
   :language: text
   :caption: output

Superfp elements and scales
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~mptorch.SuperFP` element or scale spends the codes a binaryK spends
on subnormals, and more, on powers of two below its normal binades: the
*supernormals*. As a block element it trades the uniform steps near the top of
a block for reach at the bottom, so small elements of a block with a wide
spread survive where E2M1 flushes them; as a scale it reaches far below E4M3's
:math:`2^{-9}` in the same byte. Its codes are laid out as the format lays
them out -- zero, then the supernormals from code 1, then the normal binades
as a binaryK's -- and it rounds by the superfp casts, so a tie between two
supernormals goes to the power of two with an even exponent, as
:func:`~mptorch.quant.superfp_quantize` rounds it. A superfp scale is cast
like E4M3 and multiplied by a tensor scale, spends its all-ones code on NaN,
and is chosen by ``ScaleRounding.SELECTIVE`` unless the format names another
rule, for the reason below.

.. literalinclude:: ../snippets/block_superfp.py
   :language: python
   :caption: docs/snippets/block_superfp.py

.. literalinclude:: ../snippets/block_superfp.out
   :language: text
   :caption: output

Rounding a scale to nearest costs more here. Below its normal binades a
superfp scale holds only powers of two, so the nearest scale can fall a
factor of 1.5 below :math:`a / \text{elem\_max}`: a block whose ratio is 0.35
gets the scale 0.25 under ``NEAREST``, its largest element is 8.4 in E2M1's
units, saturates to 6, and decodes 29% low (33% at worst). ``SELECTIVE``
takes 0.5 there, since 8.4 is past 7, and the element decodes 5% low.
E4M3's subnormals, below :math:`2^{-6}`, behave the same, which is what
NVFP4's tensor scale keeps blocks away from. The block GEMM decodes superfp
operands like any other, so their element codes are at most 8 bits there too
(:doc:`kernels/block`).

2D tiles
~~~~~~~~

With ``block_rows > 1`` a scale is shared by a *tile* of ``block_rows`` rows
of ``block_size`` elements: rows along the axis before the packed one, never
across a batch dimension. A tile changes only where the scale comes from;
the codes and their layout are the 1D format's, and ``scales`` is coarser,
one row per tile. ``dataclasses.replace(NVFP4, block_rows=16)`` is NVFP4
with 16 x 16 tiles, the weight format of NVIDIA's NVFP4 pretraining recipe
(which scales activations and gradients 1 x 16), and 128 x 128 tiles are
DeepSeek-V3's weight blocking, here with an E8M0 scale (a scale is one byte).

A *square* tile covers the same elements whichever of a matrix's two axes is
packed, so it quantizes the matrix to the same values either way, in every
deterministic rounding mode. That is what lets one packed weight serve a
layer's forward pass and its input gradient, which reads it transposed; a 1D
block runs along one axis only and quantizes the two orientations
differently. :doc:`kernels/block` says why a matrix product needs its
operands blocked along the dimension it sums over, and so why that matters.

.. literalinclude:: ../snippets/block_tiles.py
   :language: python
   :caption: docs/snippets/block_tiles.py

.. literalinclude:: ../snippets/block_tiles.out
   :language: text
   :caption: output

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
``SR``   stochastic rounding                  :math:`\lceil x \rceil_F` with probability :math:`\Pr(\text{up})`, else :math:`\lfloor x \rfloor_F`
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
for that fraction, the ideal rule is :math:`\Pr(\text{up}) = f`, which makes
the rounding *unbiased*: :math:`\mathbb{E}[Q(x)] = x`.

The implementation draws :math:`N` random bits (``prng_bits``) and adds them
just below the last kept mantissa bit, then truncates -- so the probability is
quantized to steps of :math:`2^{-N}`:

.. math::

   \Pr(\text{up}) = 1 - \frac{\lceil (1 - f)\, 2^{N} \rceil}{2^{N}}
   \;\xrightarrow{\;N \to \infty\;}\; f .

This is P3109's ``StochasticA`` rounding with :math:`N` random bits:
writing :math:`\eta` for the fraction of :math:`|x|` between its neighbours
(:math:`\eta = f` when :math:`x > 0`), it rounds away from zero when
:math:`\lfloor \eta\, 2^{N} \rfloor + R \ge 2^{N}` for a uniform integer
:math:`R \in [0, 2^{N})`.

``prng_bits`` lives on the format (``BinaryK(8, 4, prng_bits=8)``) and is
ignored by every other rounding mode. Two practical constraints follow from
where the bits are drawn: with ``prng_bits=0`` there is nothing random and
``SR`` degenerates into ``RZ`` (the ``SR`` row above, from a format with no
random bits, is identical to the ``RZ`` row); and the format's mantissa plus
its random bits must fit in the carrier's mantissa -- 23 bits in binary32, 52
in binary64 -- which each call checks. A float64 operand is rounded, and its
bits drawn, in binary64, two random words per draw.

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
random bits increase (with 1-3 bits, :math:`\lceil 0.4 \cdot 2^N \rceil / 2^N`
is :math:`1/2`, so the coin is fair whatever the residual). The second shows
why anyone accepts the extra variance: an accumulator whose increments are
smaller than half its spacing never moves under round-to-nearest -- the sum
of two thousand additions of :math:`0.001` to :math:`1.0` in ``Binary16p8``,
which has bfloat16's field widths, is still :math:`1.0` -- while stochastic rounding advances it by the
right amount *on average*. This is the situation of a small weight update
against a large weight, and the :doc:`tutorial` shows it deciding whether a
network trains.

Carriers
--------

Every operation is computed in a *carrier*, an IEEE binary float with far
more precision and range than the format it simulates. There are two
carriers, binary32 and binary64, and by default a tensor's dtype picks one:

- a **float32**, **float16** or **bfloat16** tensor is computed in
  **binary32** -- a float16 or bfloat16 result is then stored back in its own
  dtype, which is a second rounding (`What a narrower tensor can store`_);
- a **float64** tensor is computed in **binary64**, throughout: an elementwise
  quantizer rounds each value once, directly, and a matrix product computes
  every partial product and every running sum in binary64 before rounding it.

So moving a model to float64 does more than widen its storage. Formats that
binary32 cannot hold -- up to 53 bits of precision, ten exponent bits, values
down to :math:`2^{-1021}` -- become simulable, and a value binary32 cannot hold
is rounded on its own bits rather than on the nearest float32. The two
carriers agree wherever both hold every intermediate exactly: rounding a
float64 tensor of float32 values to a format binary32 holds gives exactly the
float32 answer, and so does a matrix product whose every product and sum is
exact in binary32 -- which is usual once a quantizer has rounded its operands
to a narrow format.

Every function and object that computes takes a ``carrier`` keyword to say
otherwise, as the ``torch.dtype`` whose arithmetic the carrier is:
``torch.float64`` for binary64, ``torch.float32`` for binary32, and ``None``,
the default, for the tensor's own. ``carrier=torch.float64`` gives a float32,
float16 or bfloat16 tensor binary64's arithmetic: its operands are widened to
float64, computed and rounded in binary64, and the result is narrowed back to
the tensor's dtype with a single rounding -- bit for bit the float64 call on
the widened tensor, then stored. The result keeps the tensor's dtype either
way. A carrier is never narrower than its tensor, so ``carrier=torch.float32``
on a float64 tensor raises: narrowing the operands would round every input
once before the format does. The quantizers
(:func:`~mptorch.quant.binaryK_quantize`, :func:`~mptorch.quant.superfp_quantize`,
:class:`~mptorch.quant.Quant`), the matrix products
(:class:`~mptorch.quant.SplitMac`, :class:`~mptorch.quant.FusedMac` and the
flat ``*_matmul`` functions) and the layer factories
(:func:`~mptorch.quant.binaryK_gemm_formats` and its siblings) all take it.

===========================================  ==========================  ==========================
carrier                                      binary32                    binary64
===========================================  ==========================  ==========================
``carrier=``                                 ``torch.float32``           ``torch.float64``
tensors computed in it by default            float32, float16, bfloat16  float64
tensors that can name it                     float32, float16, bfloat16  every floating dtype
precision, :math:`P` (``man_bits + 1``)      :math:`\le 24`              :math:`\le 53`
stochastic bits, ``man_bits + prng_bits``    :math:`\le 23`              :math:`\le 52`
random words per stochastic rounding         1                           2
largest finite value                         :math:`< 2^{128}`           :math:`< 2^{1024}`
finest step                                  :math:`2^{-149}`            :math:`2^{-1074}`
smallest value, ``SUBNORMALS`` or SuperFP    :math:`\ge 2^{-125}`        :math:`\ge 2^{-1021}`
smallest value, the other two modes (\*)     :math:`\ge 2^{-126}`        :math:`\ge 2^{-1022}`
exponent bits, at P3109's bias               :math:`\le 7`               :math:`\le 10`
matrix products on a consumer GPU            fastest                     about 3x slower
===========================================  ==========================  ==========================

(\*) ``NORMALS`` and ``EXTENDED_NORMALS``, except a binade higher --
:math:`2^{-125}` and :math:`2^{-1021}` -- at :math:`P = 1`, and for
``EXTENDED_NORMALS`` at the carrier's full precision, :math:`P = 24` and
:math:`P = 53`.

The last row is a cost worth knowing: on a GPU, binary64 arithmetic is
several times slower than binary32, so a float64 model's matrix products are
too (on the CPU the two are within a few percent). A narrower tensor that names
binary64 pays that, plus a float64 copy of each operand and a narrowed copy of
the result. The other way round is a choice of dtype rather than of carrier: a
float64 model that needs only formats binary32 can carry gets binary32's speed
as a float32 model.

The rest of this section derives the table's bounds, and says what happens to
a format that outruns them.

What the carrier can hold
~~~~~~~~~~~~~~~~~~~~~~~~~

"Far more" precision and range is true of every format in this guide, but it
is a bound on the parameters rather than a fact about all of them, and the
bound is the carrier's. A format outside it is quantized *partially*: the
result is still a tensor of plausible numbers, and some of the format's values
are simply never among them. Three limits, in the format's own terms, with
binary64's in parentheses:

* **Precision.** :math:`P \le 24` (53) for :class:`~mptorch.BinaryK`,
  ``man_bits <= 23`` (52) for :class:`~mptorch.SuperFP` -- the carrier's
  significand bits, and it cannot hold a finer grid.
* **The top.** The largest finite value must be at most the carrier's, which
  is :math:`\text{bias} \ge 2^E - 128` (:math:`2^E - 1024`), with :math:`E`
  the exponent width (``exp_bits`` for a SuperFP). Above that the cast saturates
  at the carrier's largest value on the format's grid, and the codes over it
  are unreachable.
* **The bottom.** A subnormal's exponent field is 0 whatever its magnitude, so
  any part of a cast that reads that field is blind below the carrier's
  smallest normal, :math:`2^{-126}` (:math:`2^{-1022}`) -- and which parts do
  is what sets the limit. BinaryK's subnormals and SuperFP's supernormals are
  both placed on their grid *by* that field, so their smallest value must be
  at least :math:`2^{-125}` (:math:`2^{-1021}`), a binade clear of it:
  :math:`\text{bias} + P \le 127` (1023) for the first, and
  :math:`e_{\max} - b + 2 - n \ge -125` (:math:`-1021`) for the second, in the
  notation of `SuperFP: precision at the top, range below`_. The two non-P3109 subnormal modes decide
  their bottom by comparing magnitudes instead, which reads no field, so
  ``NORMALS`` and ``EXTENDED_NORMALS`` are exact down to the smallest normal
  -- and no further, because below that the floor leaves the carrier's
  normals and nothing flushes at all. Two of their formats stop a binade
  short, because the compare also needs half the floor and the carrier cannot
  place it there: at :math:`P = 1` the rounding ahead of the compare works on
  the exponent field and reads :math:`2^{-127}` (:math:`2^{-1023}`) as a tie
  between binades, and an ``EXTENDED_NORMALS`` format at the carrier's full
  precision, :math:`P = 24` (53), has a half-floor finer than its spacing.

Together the last two say a simulated format may span at most 253 binades
(2045), so **seven exponent bits is the most either family can carry in
binary32, and ten in binary64** -- at P3109's default bias, :math:`K - P \le 7`
(10) for a signed BinaryK and :math:`K - P \le 6` (9) for an unsigned one.
The 8-bit and 6-bit formats of this guide are well inside binary32's bounds. A
format with eight exponent bits is not, whatever its bias: ``Binary16p8`` --
which the kernel pages use as an accumulator, silencing the warning described
below -- and ``Binary32p24`` reach under binary32's floor, and the bfloat16 and
float32 layouts past its top as well. binary64 holds all four.

Which carrier a format meets belongs to the tensor, not to the format, so the
limits are checked on every call, against the carrier that call rounds in --
by the quantizers of :doc:`quantizers`, the kernels of :doc:`kernels/index` and the
layers built on them. :class:`~mptorch.BinaryK` and :class:`~mptorch.SuperFP`
check only what neither carrier can do when they are built, and the
plain-integer wrappers, which never see a format object, check the same when
they resolve one -- one rule, so ``binaryK_matmul(a, b, mul_K=16, mul_P=8)``
says what ``qmatmul(a, b, BinaryK(16, 8))`` says. What separates the two
severities is not which limit is missed but whether the format still works:

* a format whose **range** outruns the carrier -- above its largest finite
  value, spaced finer than its smallest subnormal (:math:`2^{-149}`,
  :math:`2^{-1074}`), or with a smallest value below whichever floor the
  bullet above gives it -- warns with :class:`~mptorch.FormatRangeWarning`,
  naming the line that made the call. It quantizes correctly over the part
  the carrier holds, and some of its values are simply never returned. A wide
  BinaryK used as a precision-only target is exactly this, so the warning can
  be silenced with :func:`warnings.simplefilter` when that is what is meant.
* a format that **cannot function** raises :exc:`ValueError`: more precision
  than the carrier's, stochastic-rounding bits with no significand left to
  draw from, an exponent field wider than the kernels shift by, a SuperFP
  whose ``normal_binades`` leaves no supernormal codes, or a format whose
  largest finite value is below the carrier's normals, where every nonzero
  result overflows.

The run below shows both carriers on the same inputs and formats: a float64
value rounded once in binary64, and the float32 value nearest it rounded in
binary32 and, with ``carrier=torch.float64``, in binary64; four formats, each
checked against each carrier when a tensor meets it; a 30-bit format computed
exactly on a float64 tensor; and a product of float32 operands whose rounding
only binary64 gets right.

.. literalinclude:: ../snippets/formats_carriers.py
   :language: python
   :caption: docs/snippets/formats_carriers.py

.. literalinclude:: ../snippets/formats_carriers.out
   :language: text
   :caption: output

A ``carrier=torch.float64`` call is held to binary64's bounds even on a
float32 tensor, since binary64 is what rounds it -- and its result to what the
tensor's dtype can store, which the next section gives.

``dev/benchmarks/format_limits.py`` reports all of this for a given format,
in binary32 or with ``--carrier binary64``, and with ``--audit`` checks the
kernels against the format's value set to say so; its ``sweep --audit`` walks
every bound in the table above across its edge, and measured each where this
section puts it.

What a narrower tensor can store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A float16 or bfloat16 tensor is rounded in binary32, its default carrier, and
the result is converted back when it is stored -- and so, under
``carrier=torch.float64``, is a float32 one, whose result is rounded in
binary64. That conversion is a second rounding, onto the dtype's grid, correctly
rounded to nearest (for float16 and bfloat16 torch's own conversion from
float64 goes through float32, rounding twice, so MPTorch narrows those itself).
A format whose results are not all values of the dtype is
quantized partially in the same way -- with one difference worth knowing: a
result above the dtype's largest finite value is stored as :math:`\pm\infty`
*whatever the format's saturation mode*, so a ``SAT_FINITE`` format overflows
too. The dtype belongs to the tensor, not the format, so this is checked on
every call, against the format whose values the result holds, and by a rule
that depends on what the call rounded.

**A GEMM's result** is its last rounding -- the accumulate format of a
:class:`~mptorch.quant.SplitMac`, the fused format of a
:class:`~mptorch.quant.FusedMac` -- applied to products and sums that can land
anywhere, so every value of that format is some result and all of them must
be values of the dtype. For a BinaryK:

===========  ======================================  ===============  =======================================  ========
bound        float16                                 bfloat16         float32 (under binary64)                 severity
===========  ======================================  ===============  =======================================  ========
precision    :math:`P \le 11`                        :math:`P \le 8`  :math:`P \le 24`                         raises
top          largest value below :math:`2^{16}`      float32's        largest value below :math:`2^{128}`      warns
finest step  :math:`\text{bias} + P \le 26`          float32's        :math:`\text{bias} + P \le 151`          warns
             (``EXTENDED_NORMALS``: :math:`\le 25`)                   (``EXTENDED_NORMALS``: :math:`\le 150`)
===========  ======================================  ===============  =======================================  ========

A SuperFP is held to the same three, in its own parameters, and the rule does
not depend on the carrier that rounded the result. In binary32, for bfloat16
only the precision is new: its range is float32's to within its precision, so
a format that passes the float32 checks with at most eight bits of precision is
inside it at both ends. In binary64 the carrier passes ranges no narrower
dtype holds, so every row can speak. A SplitMac's multiply format is never stored -- its
products are intermediates in the carrier -- and a step left unrounded stores
no format's values at all.

**An elementwise quantizer's result** is the format's answer to a value that
is already the dtype's. Wherever the format is at least as fine as the dtype
that answer is the input itself, so neither precision nor a fine bottom costs
anything, and what can land off the dtype's grid is at the edges of the
format's range. Each of these warns:

* the format reaches past the dtype's largest finite value, and that value is
  not on the format's grid, so an input near it rounds up out of the dtype's
  range;
* the format's largest finite value is inside the dtype's range but not a
  value of it, and the format saturates onto it;
* an ``EXTENDED_NORMALS`` format's hole, :math:`2^{-\text{bias}}`, is a value of
  the dtype, and the first value above it, where such an input rounds, is not.

Either kind of call raises for a format whose largest finite value is below
the dtype's smallest, which stores nothing of the format but zero.

**A block format's decoded values** are an element value times its block's
scale, which the block kernels compute in binary32 only so far: a float64
tensor, or ``carrier=torch.float64``, raises, and a
:class:`~mptorch.BlockFormat`'s element and scale must be formats binary32
holds entirely, which it checks when it is built. A float16 or bfloat16
result rounds the decoded values a second time where the dtype does not hold
them, which :class:`~mptorch.FormatRangeWarning` reports: always for a format
with a tensor scale (NVFP4), whose decoded values are float32 products, and
for an E8M0 format only where its largest element value has more significant
bits than the dtype.

.. code-block:: python

   x = torch.ones(4, 4, dtype=torch.float16)
   binaryK_quantize(x, 8, 4)                # fine
   binaryK_quantize(x, 8, 3, bias=15)       # FormatRangeWarning: rounds up past 65504
   binaryK_quantize(x, 16, 12, bias=8)      # fine: float16's values come back as they are
   binaryK_matmul_fma(x, x, fma_K=16, fma_P=12, fma_bias=8)  # ValueError: 12 bits
   qmatmul(x, x, SplitMac(BinaryK(8, 4), BinaryK(16, 11)))    # FormatRangeWarning: 2**-25

Both rules were measured against the kernels, not only derived:
``format_limits.py --dtype float16 --audit`` runs a format through a fused GEMM
over operands of the dtype and through the quantizer over every value of it,
and ``sweep --dtype float16 --audit`` walks each boundary above -- with
``--carrier binary64`` too, through the widened operands and the narrowed
result.

.. _apple-gpu:

On an Apple GPU
~~~~~~~~~~~~~~~

On macOS every op also runs on ``"mps"`` tensors, as Metal kernels compiled
from the casts and GEMM policies the CPU runs. MPS has no float64, so there
the carrier is always binary32: float32, float16 and bfloat16 tensors compute
exactly as they do on the CPU, and ``carrier=torch.float64`` is refused, since
the float64 copy of the operands it needs cannot be made on the device.

The results are the CPU's, bit for bit, stochastic rounding included: an MPS
call draws its one seed from the CPU generator, as a CPU call does, and hands
every element the same random word, so ``torch.manual_seed`` reproduces a run
across the two devices and not only on one. (CUDA draws from its own
generator.) Two things the GPU's float unit does its own way, both in what a
GEMM computes *between* its roundings rather than in the roundings:

* **Subnormals.** Apple GPUs flush binary32 subnormals to zero, operands and
  results, in every mode Metal offers. The casts round on the word and are not
  touched by it, but a product, a running sum or a fused multiply-add whose
  exact value is below :math:`2^{-126}` in magnitude is zero before the format
  sees it. Under ``RZ`` that is what the CPU returns too, and under ``RNE`` and
  ``RNA`` so it is for every format whose smallest value is at least
  :math:`2^{-125}`. Under ``RU``, ``RD``, ``RO`` and ``SR`` the CPU can round
  such a value to the format's smallest where the GPU gives zero, and a sum
  left unrounded (``accumulate_quant=False``) can hold a subnormal the GPU
  drops. The quantizers are not affected: they round every input exactly,
  subnormal ones included, in every mode.
* **NaN payloads.** A NaN that reaches a product or a sum comes out of the
  GPU's float unit as its canonical NaN, where the CPU keeps the operand's
  payload. The casts pass a NaN through whole on every device, so a
  quantizer's NaNs keep theirs.

Each format's first call compiles its kernel, which takes from about 20 ms
to a few hundred; later calls with the same format, rounding mode and dtype
reuse it.

Where rounding is applied
-------------------------

A quantizer rounds a tensor; a kernel rounds inside a reduction. The two
compose, and MPTorch keeps them separate on purpose:

- :doc:`quantizers` -- ``binaryK_quantize``, ``superfp_quantize``,
  :class:`~mptorch.quant.Quant`, :class:`~mptorch.quant.Quantizer`, and the
  block quantizers: one rounding per element, and for a block format one
  scale per block.
- :doc:`kernels/index` -- :class:`~mptorch.quant.SplitMac`,
  :class:`~mptorch.quant.FusedMac`, :class:`~mptorch.quant.BlockMac`,
  ``qmatmul``, the convolutions: one rounding per product and per addition,
  or per fused multiply-add, inside one matrix-product routine.
- :doc:`layers` -- ``QAffineFormats`` and the layers: which quantizer is
  applied to which signal, and which arithmetic each of the layer's three
  matrix products runs in.
