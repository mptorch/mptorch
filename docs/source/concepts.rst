Formats and rounding
====================

This page presents the fundamentals that the rest of the documentation relies
on:

- what a number format is;
- which formats MPTorch implements: scalar formats, and block formats that
  share a scale among their elements;
- what each rounding mode does;
- which arithmetic the simulation rounds in;
- the equations the kernels follow.

Every example on this page comes from a run.

The simulation model
--------------------

MPTorch never computes *in* a narrow format. Every operation (a product, an
addition, an activation) is carried out in an IEEE binary float, the
*carrier*. Its result is then **rounded** to the target format :math:`F`
with a rounding mode :math:`\circ`:

.. math::

   \tilde{x} = Q_{F,\circ}(x).

For an elementwise step, such as rounding a weight tensor, that is all there
is, and it is what the quantizers do (see :doc:`quantizers`). A *reduction*
such as a dot product has intermediate results, and each of them can be
rounded. Which intermediates are rounded, and to which format, is the
arithmetic that the kernels simulate (see :doc:`kernels/index`). For one
output element of a matrix product, this can look like:

.. math::

   c_{ij} = \sum_{k} a_{ik}\, b_{kj}
   \quad\longrightarrow\quad
   s_{k} = Q_{\text{acc}}\big(s_{k-1} + Q_{\text{mul}}(a_{ik} b_{kj})\big).

The carrier (see Carriers_) has far more precision and range than the
simulated format. Computing in it and rounding once therefore usually gives
the correctly rounded result of the simulated operation, the number a machine
working natively in :math:`F` would produce. This holds for every format the
carrier can hold.

BinaryK: the IEEE P3109 formats
-------------------------------

:class:`~mptorch.BinaryK` is the family of binary floating-point formats
defined by IEEE P3109, the upcoming *Standard for Arithmetic Formats for
Machine Learning* (`Fitzgibbon, Wintersteiger and Sarnoff
<https://arxiv.org/abs/2606.04028>`__ give an overview of the draft). P3109
parameterizes a format by four things: its bitwidth :math:`K`, its precision
:math:`P`, its signedness, and its domain, which says whether it has
infinities. A format is named after them. ``Binary8p4se`` is the 8-bit
signed (``s``) format with 4 bits of precision in the extended (``e``)
domain, and ``Binary8p4sf`` is its finite-domain (``f``) twin.

The bits are laid out as in an IEEE 754 binary float. A value has a sign
:math:`s`, a biased exponent :math:`e` of :math:`E` bits and a stored
mantissa :math:`m` of :math:`P-1` bits. The :math:`P`-th bit of precision is
the implicit leading one. Its value is

.. math::

   x = (-1)^s \cdot \Big(1 + \frac{m}{2^{P-1}}\Big) \cdot 2^{\,e - \text{bias}}
   \qquad (1 \le e \le 2^E - 1)

for *normal* numbers, and

.. math::

   x = (-1)^s \cdot \frac{m}{2^{P-1}} \cdot 2^{\,1 - \text{bias}}
   \qquad (e = 0)

for *subnormal* numbers, which fill the gap between zero and the smallest
normal with the same absolute spacing. You give the total width :math:`K`
and the precision :math:`P`. The exponent width follows: :math:`E = K - P`
for a signed format, and :math:`E = K - P + 1` for an unsigned one, which has
no sign bit. ``is_signed`` is P3109's signedness, and the domain is set
together with the saturation mode (below).

A few consequences are worth keeping in mind:

- Within one *binade* :math:`[2^q, 2^{q+1})` there are :math:`2^{P-1}`
  representable values, spaced :math:`2^{q-P+1}` apart. This spacing is the
  *unit in the last place* (ulp). The relative spacing is therefore between
  :math:`2^{-P}` and :math:`2^{1-P}`: neighbouring values are about 6-12 %
  apart with :math:`P = 4` (E4M3's precision) and 12-25 % apart with
  :math:`P = 3` (E5M2's). Rounding to nearest moves a value by at most half
  of that.
- The largest finite value depends on the domain. In P3109's extended domain
  (the default) the last code of the top binade is :math:`\infty`, and the
  largest finite value is
  :math:`(2 - 2^{2-P}) \cdot 2^{\,2^E - 1 - \text{bias}}`. In the finite
  domain that code is a number, and the largest finite value is
  :math:`(2 - 2^{1-P}) \cdot 2^{\,2^E - 1 - \text{bias}}`. Both hold for a
  signed format with :math:`P \ge 3`. `Subnormals, saturation and sign`_ says
  how to choose the domain, and `Relation to IEEE P3109`_ gives the rules for
  the other formats.
- The smallest normal is :math:`2^{1 - \text{bias}}` and the smallest
  subnormal :math:`2^{1 - \text{bias} - (P-1)}`.
- ``bias`` defaults to P3109's, the middle of the exponent range:
  :math:`2^{E-1}`, which is :math:`2^{K-P-1}` for a signed format and
  :math:`2^{K-P}` for an unsigned one. Either way 1.0 is encoded at the
  middle code point. For the same exponent width, IEEE 754's bias is
  :math:`2^{E-1} - 1`, one less. P3109 fixes the bias rather than the largest
  exponent, so that adding or removing the infinities moves no finite value.
  Pass ``bias=`` explicitly for a format outside P3109, such as the nearest
  spellings of the OCP 8-bit formats.
- A BinaryK format uses every exponent code for finite values, where IEEE 754
  reserves the top one for infinities and NaNs. Take a BinaryK format with
  the IEEE bias and the same exponent and mantissa widths as an IEEE 754
  format (float16, float32). It agrees with that format on every value up to
  its largest finite one, and then continues for one more binade. So no
  BinaryK format *is* float16, bfloat16 or float32 (see `The IEEE 754 dtypes
  are not P3109 formats`_). The OCP formats compare the same way, and the
  tutorial checks it exhaustively (see :doc:`tutorial`): E5M2 is laid out the
  IEEE way, so its BinaryK spelling also goes a binade past it, while E4M3
  spends its top exponent field on numbers and is matched on every finite
  value.

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

The last five rows have each format's field widths and bias, but none of
them *is* the format it is laid out as: whatever its bias, a BinaryK spends
its codes the way P3109 does.

- The E4M3 row comes closest. It has exactly E4M3's finite values and differs
  in the special ones, and so in what an overflow returns.
- The other four have more values than their namesakes, one binade past
  them.

`The IEEE 754 dtypes are not P3109 formats`_ says how far each row is from
the format it is named after. The two 8-bit rows can be imported as
``mptorch.E4M3`` and ``mptorch.E5M2``. The MX block presets build on them
with OCP's largest values and NaN codes (see `Block formats`_).

The following run constructs P3109's two signed 8-bit formats with 4 and 3
bits of precision, derives their ranges from the formulas above, and checks
them against the quantizer in both domains. Each boundary value is a fixed
point of the rounding in the domain that has it. For example,
:math:`10^{30}` overflows to :math:`\infty` in the extended domain and to
the largest finite value in the finite one.

.. literalinclude:: ../snippets/formats_binaryK.py
   :language: python
   :caption: docs/snippets/formats_binaryK.py

.. literalinclude:: ../snippets/formats_binaryK.out
   :language: text
   :caption: output

The last two lines show the grid. In :math:`[1, 2)` ``Binary8p4`` has the
eight values :math:`1, 1.125, \dots, 1.875`, and each input lands on the
nearest one. Ties go to the even mantissa (:math:`1.0625 \to 1.0`,
:math:`1.1875 \to 1.25`, :math:`1.9375 \to 2.0`).

The IEEE 754 dtypes are not P3109 formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

float16, bfloat16 and float32 are not exactly representable as P3109 formats
in either domain, nor as BinaryK formats with any other bias. The two
standards spend the code points of the same layout differently:

- IEEE 754 reserves the whole top exponent field (:math:`2^{P-1}` codes of
  each sign) for :math:`\pm\infty` and the NaNs. It gives zero two encodings,
  :math:`+0.0` and :math:`-0.0`.
- P3109 spends one code of each sign on :math:`\pm\infty`, and only in the
  extended domain. Every other code of the top field is a finite value. A
  signed format has a single, unsigned zero, and the encoding IEEE 754 would
  read as :math:`-0.0` is its one NaN.
- P3109's default bias, :math:`2^{E-1}`, is one more than IEEE 754's for the
  same exponent width (:math:`2^{E-1}-1`).

A bias can therefore line up at most one end of the range with that of the
closest IEEE 754 dtype, and the special values never match. For float16, for
instance:

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
same way:

- With the IEEE 754 bias, the BinaryK format reaches a binade past the dtype
  (about :math:`6.8 \cdot 10^{38}` against :math:`3.4 \cdot 10^{38}`), with
  :math:`2^{P-1} - 1` extra values: 127 for bfloat16 and :math:`2^{23} - 1`
  for float32.
- With P3109's bias, ``Binary16p8`` and ``Binary32p24`` lose the dtype's
  largest value to :math:`+\infty` in the extended domain and match it in the
  finite one. They add :math:`2^{P-1}` values below the dtype's smallest
  normal, down to :math:`2^{-134}` and :math:`2^{-150}`.

The two OCP 8-bit formats relate to their BinaryK rows differently.

**E5M2** reserves its top exponent field as IEEE 754 does. So
``BinaryK(8, 3, bias=15)`` relates to it as ``BinaryK(16, 11, bias=15)``
relates to float16 above: it has every finite value of E5M2 and one binade
more. E5M2's largest finite value is :math:`57344`. The BinaryK format adds
:math:`65536`, :math:`81920` and :math:`98304` of each sign. Under
``SAT_FINITE`` it also adds :math:`114688`, the code that is
:math:`\pm\infty` in the extended domain.

**E4M3** spends its top exponent field on finite values, as P3109 does. So
``BinaryK(8, 4, bias=7)`` has exactly E4M3's finite values, up to
:math:`448`. Only the special codes differ: the BinaryK format has
:math:`\pm\infty` where E4M3 has its two NaNs, and its one NaN where E4M3 has
:math:`-0`. The difference shows in what an overflow returns. E4M3 has no
infinity, and PyTorch's conversion to ``torch.float8_e4m3fn`` saturates at
:math:`\pm 448`. ``BinaryK(8, 4, bias=7)`` returns:

- :math:`\pm\infty` under ``SAT_NONE``, the default.
- :math:`\pm 448` for a finite input under ``SAT_PROPAGATE``, which leaves an
  infinite input infinite.
- :math:`\pm 448` for every input under ``SAT_FINITE_EXTENDED``, infinite
  ones included. This is PyTorch's result for every float32 input, except
  that a zero is always :math:`+0`.
- :math:`\pm 480` under ``SAT_FINITE``. Its finite domain turns the
  infinity's code into a number, which E4M3 does not have.

In MPTorch these differences show up in two places:

- Quantizing to one of the dtype-layout rows is not the dtype's own
  conversion. A value between the dtype's largest finite value and the row's
  stays finite, where the dtype would store :math:`\pm\infty`. And
  :math:`-0.0` comes back as :math:`+0.0`.
- The range checks flag these formats:

  - ``BinaryK(16, 8, bias=127)`` and ``BinaryK(32, 24, bias=127)`` reach
    :math:`2^{128}`. On a float32 tensor they warn that they outrun binary32.
  - ``Binary16p8`` and ``Binary32p24`` warn about their bottom there (see
    `What the carrier can hold`_). A float64 tensor holds all four.
  - ``BinaryK(16, 11, bias=15)``, as the format of a GEMM's result over
    float16 operands, warns that it reaches past float16 (see `What a
    narrower tensor can store`_).

Subnormals, saturation and sign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two enums decide what happens at the ends of the range. Both are properties
of the *format*, so two formats in one computation can differ in them.

:class:`~mptorch.SubnormalsMode` decides the bottom of the range:

``SUBNORMALS``
   Gradual underflow, as in IEEE 754 and in P3109, whose formats all have
   subnormals once :math:`P > 1`. The exponent code 0 encodes the subnormal
   numbers, down to :math:`2^{1-\text{bias}-(P-1)}`.
``NORMALS``
   No subnormals, which takes the format outside P3109. The exponent code 0
   encodes only the zero, and the smallest value is the smallest normal,
   :math:`2^{1-\text{bias}}`.
``EXTENDED_NORMALS``
   No subnormals either, and also outside P3109. The exponent code 0 encodes
   one more binade of *normal* numbers, above :math:`2^{-\text{bias}}`, with
   the full mantissa. Its first code is still the zero, so the smallest value
   is one step above the power of two,
   :math:`(1 + 2^{1-P})\, 2^{-\text{bias}}`.

Below its smallest value every mode rounds the same way: to zero or to that
value, as the rounding mode decides (`Relation to IEEE P3109`_ gives the
rule). A value is flushed to zero only if the rounding mode says so. Under
round-to-nearest, that is when it is nearer to zero than to the smallest
value.

:class:`~mptorch.SaturationMode` and :class:`~mptorch.Domain` decide the top.
P3109 chooses two things here, separately:

- the *domain*: whether the format spends its top code of each sign on
  :math:`\pm\infty` (extended) or on a number (finite);
- the *saturation mode*: what an operation does with a result beyond the
  largest finite value.

The finite domain admits only ``SatFinite``, which leaves four valid pairs.
Each pair is one member of ``SaturationMode``:

``SAT_FINITE``
   P3109's ``SatFinite``, in the finite domain. Everything is clamped to the
   largest finite value, which here includes the top mantissa code. This
   applies to a finite input that overflows and to an infinite input, so the
   output never contains an infinity. NaN inputs pass through unchanged.
``SAT_FINITE_EXTENDED``
   P3109's ``SatFinite``, in the extended domain. The format keeps its
   infinity codes, so its largest finite value is the same as under
   ``SAT_PROPAGATE`` and ``SAT_NONE``. Everything is clamped to it, infinite
   inputs included, as under ``SAT_FINITE``.
``SAT_PROPAGATE``
   P3109's ``SatPropagate``, in the extended domain. A finite input that
   overflows is clamped to the largest finite value, which excludes the top
   mantissa code. This includes an input that rounds onto that code, like 480
   in the run below. An infinite input stays infinite.
``SAT_NONE``
   P3109's ``SatNone``, in the extended domain, and the default. An input
   that rounds past the largest finite value becomes :math:`\pm\infty`. P3109
   rounds before it saturates, so this happens under every rounding mode,
   including those for which IEEE 754 stops at the largest finite value, such
   as rounding toward zero (the run's ``SAT_NONE, RZ`` line).

A :class:`~mptorch.BinaryK` or :class:`~mptorch.SuperFP` can also name the
pair as P3109 does, with ``domain=`` next to ``saturation=``.
``BinaryK(8, 4, saturation=SAT_FINITE, domain=Domain.EXTENDED)`` is stored as
``saturation=SAT_FINITE_EXTENDED``, and the two spellings make equal formats.
If ``domain`` is left out, it follows from the mode: finite for
``SAT_FINITE`` and extended otherwise. ``domain=Domain.FINITE`` with any
other mode raises. The flat functions of the schema tier take no format
object, so they name the pair with the ``SaturationMode`` member alone.

.. literalinclude:: ../snippets/formats_modes.py
   :language: python
   :caption: docs/snippets/formats_modes.py

.. literalinclude:: ../snippets/formats_modes.out
   :language: text
   :caption: output

An unsigned format (``is_signed=False``) has no sign bit. Negative inputs
become zero, and the freed bit goes to the exponent field. The run's last
line uses the unsigned ``BinaryK(8, 4)``, P3109's ``Binary8p4ue``. Compared
with the signed ``Binary8p4se``:

- It has five exponent bits and the default bias :math:`2^{K-P} = 16`, where
  the signed format has four and :math:`2^{K-P-1} = 8`. Its normal exponents
  therefore run from :math:`-15` to :math:`15` instead of :math:`-7` to
  :math:`7`.
- Its largest finite value is :math:`1.625 \cdot 2^{15} = 53248`, against
  :math:`1.75 \cdot 2^{7} = 224`. The unsigned top binade ends in two
  reserved codes, :math:`+\infty` and NaN, and the signed one in one.

The extra bit buys range, not precision. Both formats have :math:`P = 4`, so
around :math:`3000` the unsigned format's values are 256 apart, and the run's
input :math:`3000` comes back as :math:`3072`, the nearest of them.

Relation to IEEE P3109
~~~~~~~~~~~~~~~~~~~~~~

The sections above introduced P3109 one notion at a time: how it names a
format after its parameters, its default bias, its two domains and its three
saturation modes. This section puts them side by side and says in what sense
a :class:`~mptorch.BinaryK` *is* a P3109 format. It covers, in order:

- which MPTorch argument stands for which notion of the standard.
- the rule for the largest finite value, which the earlier sections stated
  only for a signed format with :math:`P \ge 3`.
- how the correspondence is tested.
- the two ways in which MPTorch departs from the standard.

Every parameter of a P3109 format, and each of its saturation and rounding
modes, has a counterpart among the arguments of ``BinaryK`` and MPTorch's
enums. A ``BinaryK(K, P)`` left at its defaults is therefore the P3109 format
with that bitwidth and precision, signed and in the extended domain, for
every width the standard admits. The rounding modes are introduced further
down (see `Rounding modes`_); their rows are listed here so that the whole
correspondence is in one table.

.. list-table::
   :header-rows: 1

   * - P3109
     - MPTorch
   * - bitwidth :math:`K`, precision :math:`P`, signedness
     - ``BinaryK(K, P, is_signed=...)``
   * - extended / finite domain
     - ``domain=Domain.EXTENDED`` / ``Domain.FINITE``
   * - exponent bias :math:`2^{K-P-1}`, unsigned :math:`2^{K-P}`
     - the default ``bias``
   * - ``SatNone``, ``SatPropagate``, ``SatFinite``
     - ``saturation=SAT_NONE``, ``SAT_PROPAGATE``, ``SAT_FINITE``
   * - extended domain with ``SatFinite`` (one ``SaturationMode`` member, for
       the flat functions)
     - ``SAT_FINITE_EXTENDED``
   * - ``NearestTiesToEven``, ``NearestTiesToAway``
     - ``RNE``, ``RNA``
   * - ``TowardPositive``, ``TowardNegative``, ``TowardZero``
     - ``RU``, ``RD``, ``RZ``
   * - ``ToOdd``
     - ``RO``
   * - ``StochasticA`` with :math:`N` random bits
     - ``SR`` with ``prng_bits=N``

The top of the range is counted in P3109's code points. The top binade has
:math:`2^{P-1}` codes:

- In the extended domain, the last one is :math:`+\infty`.
- An unsigned format spends its last one on NaN and, in the extended domain,
  the one below that on :math:`+\infty`.

The largest finite value is the highest code left: :math:`224` for
``Binary8p4se``, :math:`240` for ``Binary8p4sf`` and :math:`53248` for
``Binary8p4ue``. With :math:`P \le 2` the reserved codes can outnumber the
top binade's, which moves the largest finite value a binade or two lower. A
finite result above it saturates in every rounding mode.

``dev/benchmarks/binaryK_values.py`` prints the whole code-point table of a
format: every value, one per line, infinities and NaN included. With
``--verify`` it quantizes each value and checks that the kernel returns it
unchanged. ``dev/benchmarks/superfp_values.py`` does the same for SuperFP.

Two test files compare the formats to the standard:

- ``tests/test_binaryk_quantize.py`` checks the six deterministic modes
  against `gfloat <https://github.com/graphcore-research/gfloat>`__, a
  reference implementation of P3109, below the top of the range.
- ``tests/test_binaryk_p3109.py`` checks the top itself, against a
  transcription of the paper's definitions: every precision of the 3- to
  8-bit formats, signed and unsigned, in all three saturation modes. gfloat
  is not used there because it overflows the IEEE 754 way under directed
  rounding.

MPTorch departs from the standard in two ways.

**By extension.** Three options describe formats that P3109 does not define:
a ``bias`` other than the default, the ``NORMALS`` and ``EXTENDED_NORMALS``
subnormal modes, and the widths the standard excludes (it requires
:math:`K \ge 3`, and :math:`P < K` for a signed format). These options are
what make E4M3's finite values reachable, as well as the nearest
representations of E5M2 and the IEEE 754 dtypes (see `The IEEE 754 dtypes are
not P3109 formats`_).

The two extra subnormal modes change only the bottom of the range. Both
concern the :math:`2^{P-1}` codes of each sign whose exponent field is zero.
``SUBNORMALS`` spends these on the zero and the :math:`2^{P-1} - 1`
subnormals. The other two modes use them differently:

- ``NORMALS`` leaves all of them unused except the zero, so the smallest
  positive value is the smallest normal, :math:`2^{1-\text{bias}}`.
- ``EXTENDED_NORMALS`` spends them on one more binade of normals,
  :math:`[2^{-\text{bias}}, 2^{1-\text{bias}})`, directly below the smallest
  normal. The mantissa-zero code of that binade does not encode
  :math:`2^{-\text{bias}}`. It remains the zero (and, with the sign bit set,
  the NaN), as in every other BinaryK format. The binade therefore holds
  :math:`2^{P-1} - 1` values. The smallest is one step above the power of two
  at its foot, :math:`(1 + 2^{1-P})\, 2^{-\text{bias}}`. With :math:`P = 1`
  the binade holds no values.

Let :math:`x_{\min}` be the smallest positive value of the format:
:math:`2^{1-\text{bias}-(P-1)}` under ``SUBNORMALS``, and the two values in
the list above under the other modes. Below :math:`x_{\min}` all three modes
round the same way. An input :math:`x` with :math:`0 < |x| < x_{\min}` has
two candidates for its magnitude, zero and :math:`x_{\min}`, and the
rounding mode picks between them (see `Rounding modes`_):

- ``RNE`` and ``RNA`` take the nearer one. On a tie, :math:`|x| = x_{\min} / 2`,
  ``RNE`` takes the zero and ``RNA`` takes :math:`x_{\min}`.
- ``RU``, ``RD`` and ``RZ`` take the candidate in their own direction.
- ``RO`` takes :math:`x_{\min}`, the nonzero candidate.
- ``SR`` takes :math:`x_{\min}` with probability :math:`|x| / x_{\min}`, and
  the zero otherwise.

**By simulating values rather than code points.** A result is a value of the
carrier, not an encoding, so P3109's single NaN is whichever NaN came in.
P3109's single zero is unsigned, and here MPTorch does follow the standard:
every zero the kernels return is :math:`+0.0`, whether it came from
:math:`-0.0` or from a negative value that rounded to zero. The same holds
for SuperFP.

SuperFP: precision at the top, range below
------------------------------------------

:class:`~mptorch.SuperFP` is the second format family the kernels implement.
It has the same three fields as a binary float (``man_bits``, ``exp_bits``,
``bias``), but only the top ``normal_binades`` binades carry a mantissa. The
codes of the remaining
:math:`2^{\text{exp\_bits}} - \text{normal\_binades}` binades,
:math:`2^{\text{man\_bits}}` codes each, are reinterpreted. The first is the
zero. The others are further *powers of two* below the normal region, one per
code: numbers with an implicit mantissa of exactly 1. With
:math:`e_{\max} = 2^{\text{exp\_bits}} - 1 - \text{bias}`, the three regions
are

.. math::

   \begin{aligned}
   \text{normal:}      &\quad [2^{\,e_{\max} - b + 1},\; 2^{\,e_{\max}+1}),
      && \text{full mantissa} \\
   \text{supernormal:} &\quad [2^{\,e_{\max} - b + 2 - n},\; 2^{\,e_{\max} - b + 1}),
      && \text{powers of two only} \\
   \text{underflow:}   &\quad \text{below that}, && \text{flushed to zero}
   \end{aligned}

where :math:`b = \text{normal\_binades}` and
:math:`n = (2^{\text{exp\_bits}} - b)\, 2^{\text{man\_bits}}` is the number
of reinterpreted codes, of which :math:`n - 1` are supernormals. There are no
subnormals. The bias has no default: it is part of the format's design and
must be given. In the supernormal region, rounding acts on the *exponent*:
to nearest, :math:`3 = 2^{1.58}` becomes :math:`4`.

.. literalinclude:: ../snippets/formats_superfp.py
   :language: python
   :caption: docs/snippets/formats_superfp.py

.. literalinclude:: ../snippets/formats_superfp.out
   :language: text
   :caption: output

With the same 8 bits as E4M3, this format reaches down to :math:`2^{-104}`
instead of :math:`2^{-9}`. The price is that it keeps its mantissa only in
the two top binades. It suits quantities whose *scale* matters more than
their digits.

.. _block-formats:

Block formats
-------------

The formats above give every element an exponent of its own. A *block
format* shares one scale among a block of consecutive elements and stores
the elements as narrow codes, so a 4-bit element can carry values over the
whole range of a float32 tensor, one block at a time. The OCP Microscaling
(MX) formats and NVIDIA's NVFP4 are block formats, and they are what the
newest accelerators multiply natively. The MX formats are defined by the `OCP
Microscaling Formats (MX) Specification
<https://opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
(version 1.0, September 2023) and evaluated by `Rouhani et al.
<https://arxiv.org/abs/2310.10537>`__; NVFP4 is described in `Pretraining
Large Language Models with NVFP4 <https://arxiv.org/abs/2509.25149>`__
(Section 2 and Appendix B).

A :class:`~mptorch.BlockFormat` is defined by an element format, a scale format
and a block size, with the element and the scale each a
:class:`~mptorch.BinaryK` or a :class:`~mptorch.SuperFP` (see `SuperFP elements
and scales`_, below). A tensor is split into blocks of ``block_size``
consecutive elements along one axis, its *packed* axis, and each block stores
one scale code beside its element codes. Element :math:`i` of block
:math:`\kappa` stands for the value

.. math::

   x_i \;\approx\; e_i\, \sigma_\kappa ,

its element value :math:`e_i` times its block's scale :math:`\sigma_\kappa`.

As an example, in the MX formats the scale format is E8M0, an unsigned exponent 
with no mantissa: code :math:`c` is the power of two :math:`2^{c - 127}`, and 
code 255 is NaN. Writing :math:`X_\kappa` for the block's exponent,

.. math::

   \sigma_\kappa = 2^{X_\kappa}, \qquad -127 \le X_\kappa \le 127 ,

so the block scale alone places a block anywhere in float32's range, and the
format needs nothing else.

Tensor scales
~~~~~~~~~~~~~

A scale format with a mantissa -- NVFP4's E4M3, or a SuperFP -- follows a
block's largest magnitude more closely than a power of two can, but over a
much shorter range: E4M3's positive values run from :math:`2^{-9}` to
:math:`448`. By itself it could represent only a block whose largest magnitude
is between :math:`2^{-9}` and :math:`448` times the element format's largest
value. Such a format can therefore use a second scale, one float32 number
:math:`S_t` for the whole tensor. This tensor scale can usually bring back the 
tensor into the scale format's range, with the block scale differentiating the 
blocks:

.. math::

   \sigma_\kappa = s_\kappa\, S_t ,

where :math:`s_\kappa` is the value of the block's scale code and the product
is computed in float32. :math:`S_t` is one number for the whole tensor, kept
beside the codes (the ``tensor_scale`` of a
:class:`~mptorch.quant.BlockPacked`), so it adds nothing to the bytes per
element. `How a block is quantized`_ gives its default value.

MPTorch currently pairs the tensor scale to the kind of block scale being
used; it is not an option of its own. A :class:`~mptorch.BlockFormat`
therefore can be in one of three states:

- **Block scale only**: a power-of-two scale, as in the MX presets. Such a
  format has no tensor scale, and passing ``tensor_scale=`` to it raises.
- **Block scale and tensor scale**: any scale with a mantissa, as in NVFP4.
  A static ``tensor_scale=1.0`` leaves the block scale alone, which is block
  scaling with a mantissa scale.
- **No scale**: ``scale=None`` stores the element codes as they are.

A tensor scale without block scales is not available.

The presets
~~~~~~~~~~~

MPTorch includes as presets five of the six concrete formats of OCP MX v1.0,
and NVIDIA's NVFP4:

==============  =======  ===========  ============  =====  ===============  =============
preset          element  block scale  tensor scale  block  largest element  bytes/element
==============  =======  ===========  ============  =====  ===============  =============
``MXFP8_E4M3``  E4M3     E8M0         none          32     448              1.031
``MXFP8_E5M2``  E5M2     E8M0         none          32     57344            1.031
``MXFP6_E2M3``  E2M3     E8M0         none          32     7.5              0.781
``MXFP6_E3M2``  E3M2     E8M0         none          32     28               0.781
``MXFP4_E2M1``  E2M1     E8M0         none          32     6                0.531
``NVFP4``       E2M1     E4M3         float32       16     6                0.562
==============  =======  ===========  ============  =====  ===============  =============

"Largest element" is the format's ``elem_max``, the largest element value
:math:`e_i`. The largest value a block can stand for is ``elem_max`` times
its largest scale. OCP's sixth format, MXINT8, has 8-bit integer elements and
no preset, because an element here is a :class:`~mptorch.BinaryK` or a
:class:`~mptorch.SuperFP`. The element formats use OCP's biases (IEEE 754's),
one below P3109's default, so the presets spell them out (``E4M3 =
BinaryK(8, 4, bias=7)``). `Relation to IEEE P3109`_ explains how
:class:`~mptorch.BinaryK` relates to the standard.

.. _scale-rounding:

How a block is quantized
~~~~~~~~~~~~~~~~~~~~~~~~

A block format stores each element :math:`x_i` of a block as an element value
:math:`e_i`, next to one scale :math:`s` for the whole block, so that

.. math::

   x_i \approx e_i\, s .

Quantizing a block means choosing both. The scale :math:`s` has to be a value
of the scale format, and each :math:`e_i` a value of the element format. Once
the scale is fixed the elements follow: :math:`e_i` is :math:`x_i / s`
rounded to the element format, and the block then stands for the *decoded*
values :math:`\hat{x}_i = e_i\, s`. The main question is which scale to take,
and the answer depends on the format. A tensor is quantized one block at a
time.

The scale a block asks for
^^^^^^^^^^^^^^^^^^^^^^^^^^

One scale serves the whole block, so it can put at least one element exactly
where we want it, and the natural choice is the largest. Let
:math:`a = \max_i |x_i|` be the block's largest magnitude (NaNs aside) and
``elem_max`` the element format's largest value: 6 for E2M1, whose
magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4 and 6, and 448 for E4M3. An element
above ``elem_max`` cannot be stored, so the codes are used best when the
block's largest element sits exactly at the top, :math:`a =
\text{elem\_max} \cdot s`. That gives the scale

.. math::

   r = \frac{a}{\text{elem\_max}} .

But :math:`r` is rarely a value of the scale format. Let
:math:`\lfloor r \rfloor` and :math:`\lceil r \rceil` be the scale format's
values nearest to :math:`r` from below and from above; for E8M0 these are
the two powers of two around :math:`r`. They are the two most common
candidates, and each has a cost:

- :math:`s = \lfloor r \rfloor` is too small. The block's largest element
  lands at :math:`a / \lfloor r \rfloor = \text{elem\_max} \cdot r / \lfloor
  r \rfloor`, above ``elem_max``. No code holds such a value, so it is clipped
  to ``elem_max`` (it *saturates*), in every rounding mode, and decodes as

  .. math::

     \hat{a} \;=\; \text{elem\_max} \cdot \lfloor r \rfloor
              \;=\; a \cdot \frac{\lfloor r \rfloor}{r} \;<\; a .

  The block's largest value comes back too small by the factor
  :math:`\lfloor r \rfloor / r`.
- :math:`s = \lceil r \rceil` is too large. The block's largest element
  lands at :math:`\text{elem\_max} \cdot r / \lceil r \rceil`, below
  ``elem_max``, so every element is in range. But two neighbouring decoded
  values are one element step :math:`\delta` times the scale apart, and that
  distance is now

  .. math::

     \delta \cdot \lceil r \rceil \;>\; \delta \cdot r .

  The block's grid is coarser than necessary by the factor
  :math:`\lceil r \rceil / r`. Every element is rounded more coarsely, and
  small values reach zero sooner: rounded to nearest, an element becomes zero
  once its magnitude is below half the element format's smallest value times
  the scale, :math:`0.25\, \lceil r \rceil` for E2M1.

The run at the end of this section shows both choices on one MXFP4 block
whose largest magnitude is 3.8. It asks for :math:`r = 3.8 / 6 \approx 0.63`,
so :math:`\lfloor r \rfloor = 1/2` and :math:`\lceil r \rceil = 1`:

- Under the scale 1/2, 3.8 is 7.6 in the element's units. It is clipped to 6
  and decodes as :math:`6 \cdot 1/2 = 3.0`, which is 21% low:
  :math:`\lfloor r \rfloor / r = 0.5 / 0.63 = 0.79`.
- Under the scale 1, it is 3.8, rounds to 4 and decodes as 4.0, which is 5%
  high. But the grid is twice as coarse: the block's 0.3 decodes as 0.5
  instead of 0.25, and anything below 0.25 would decode as zero.

The four rules: one dividing line each
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A block gets one scale, chosen once from the block's own values and stored
with it. A *rule* makes this choice: given a block, it returns the lower
candidate or the upper one. Blocks hold different values, so the same rule
gives some blocks of a tensor the lower candidate and others the upper one.

``BlockFormat(..., scale_rounding=)`` names the rule, a member of
:class:`~mptorch.ScaleRounding`. Four of the rules choose between the two
candidates and are described here; the other three also consider a third
scale option (see `Four Over Six`_, below).

A rule's answer depends on where the scale the block asks for, :math:`r`,
lies between the block's two candidates:

.. math::

   \rho = \frac{r}{\lfloor r \rfloor} .

Under the lower candidate the block's largest element lands at
:math:`\text{elem\_max} \cdot \rho`, so :math:`\rho` also measures how much
that candidate clips:

- :math:`\rho = 1`: the lower candidate is the scale the block asks for, and
  nothing is clipped.
- A larger :math:`\rho`: the lower candidate clips more, and the case for
  the upper one is stronger. Between two powers of two, :math:`\rho` is below
  2.

Each of the four rules has a *dividing line* :math:`\theta`, and gives a
block

- the lower candidate, :math:`\lfloor r \rfloor`, if the block's
  :math:`\rho` is below the line: there is little to clip, so the block keeps
  the finer grid;
- the upper candidate, :math:`\lceil r \rceil`, if the block's :math:`\rho`
  is above the line: too much would be clipped, so the block takes the
  coarser grid instead.

The rules differ only in where they put the line. The table below has four
MXFP4 blocks (E2M1 elements, power-of-two scale) with different largest
magnitudes. Each asks for a scale between 1/2 and 1, its two candidates, and
the last four columns give the scale each rule chooses:

.. list-table::
   :header-rows: 1
   :widths: 16 12 10 18 9 13 9 13

   * - largest magnitude :math:`a`
     - :math:`r = a / 6`
     - :math:`\rho`
     - lands at, under 1/2
     - ``UP``
     - ``SELECTIVE``
     - ``OCP``
     - ``NEAREST``
   * - 3.1
     - 0.52
     - 1.03
     - 6.2
     - 1
     - 1/2
     - 1/2
     - 1/2
   * - 3.6
     - 0.60
     - 1.20
     - 7.2
     - 1
     - 1
     - 1/2
     - 1/2
   * - 4.2
     - 0.70
     - 1.40
     - 8.4
     - 1
     - 1
     - 1
     - 1/2
   * - 5.0
     - 0.83
     - 1.67
     - 10.0
     - 1
     - 1
     - 1
     - 1

A row is one block under the four rules. A column is one rule, whose answer
changes once, where :math:`\rho` passes its line: ``UP``'s line is below all
four blocks, ``SELECTIVE``'s is between the first and the second, ``OCP``'s
between the second and the third, and ``NEAREST``'s between the third and the
fourth.

The next table gives each rule's line, lowest first. Its last three columns
are for E2M1 under a power-of-two scale, where the rules differ most, and
:math:`T` is explained under ``SELECTIVE`` below.

.. list-table::
   :header-rows: 1
   :widths: 14 38 12 14 22

   * - rule
     - gives the upper candidate when
     - dividing line :math:`\theta`
     - largest element lands in
     - which then
   * - ``UP``
     - :math:`r` is not itself a scale value
     - 1
     - :math:`(3, 6]`
     - is never clipped
   * - ``SELECTIVE``
     - ``NEAREST`` does, or the lower candidate would put the largest
       element above :math:`T`
     - 7/6
     - :math:`(3.5, 7]`
     - moves no further than rounding moves it
   * - ``OCP``
     - the lower candidate would put the largest element past the element
       format's top binade
     - 4/3
     - :math:`[4, 8)`
     - is clipped above 7, up to 25% low
   * - ``NEAREST``
     - :math:`r` is nearer to the upper candidate
     - 3/2
     - :math:`[4.5, 9]`
     - is clipped above 7, up to 33% low

A rule with a low line gives most blocks the upper candidate: it clips
little, and many blocks get the coarser grid. A rule with a high line does
the reverse. The earlier example block (largest magnitude 3.8) has
:math:`\rho = 0.63 / 0.5 = 1.27`, above the lines of ``UP`` and ``SELECTIVE``
and below those of ``OCP`` and ``NEAREST``. The first two give it the scale
1 and the other two the scale 1/2, as the run shows.

``OCP``
   This rule never computes :math:`r`. It rounds :math:`a` and ``elem_max``
   each down to a power of two, and divides:

   .. math::

      s = 2^{\lfloor \log_2 a \rfloor - e_{\max}}, \qquad
      e_{\max} = \lfloor \log_2(\text{elem\_max}) \rfloor .

   This is the conversion that the `OCP Microscaling Formats (MX)
   Specification
   <https://opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
   gives (version 1.0, Section 6.3). It needs only the exponent of :math:`a`,
   and no division. The result is still one of the two candidates: the power
   of two that puts the block's largest element in the element format's top
   binade, :math:`[2^{e_{\max}}, 2^{e_{\max}+1})`. For E2M1 that binade is
   :math:`[4, 8)`, but E2M1 stops at 6. The rule therefore takes
   :math:`\lfloor r \rfloor` whenever the largest element then stays below 8,
   and that element is clipped if it is above 6, by up to a quarter of its
   value. In general the dividing line is
   :math:`2^{e_{\max}+1} / \text{elem\_max}`: 4/3 for E2M1 and 8/7 for E4M3.
   The rule exists for a power-of-two scale only.

``NEAREST``
   Take the candidate nearer to :math:`r`, ties to even. This is NVFP4's
   rule: `Pretraining Large Language Models with NVFP4
   <https://arxiv.org/abs/2509.25149>`__ (Appendix B.2) rounds each block's
   scale to E4M3 this way. How much it clips depends on how far apart the two
   candidates are. In E4M3's normal range they differ by at most a factor of
   1.125, so the scale is at most a factor of 1.0625 below :math:`r`. The
   largest element then lands between 5.65 and 6.375, and E2M1 takes it to 6
   either way. Two powers of two differ by a factor of 2: the dividing line
   is at 3/2, and the largest element is clipped by up to a third.

``UP``
   Always take :math:`\lceil r \rceil`. No element ever passes ``elem_max``.
   This is the rule that NVIDIA's `Recipes for Pre-training LLMs with MXFP8
   <https://arxiv.org/abs/2506.08027>`__ (Section 3.3, Algorithm 1) uses for
   E8M0. Its authors describe the specification's rule as rounding the scale
   down. They report training issues under it, which they hypothesize come
   from the clipped values, and none once the scale is rounded up. The price
   is the other cost, for every block: from a power of two, a grid up to
   twice as coarse as the block needs. The NVFP4 paper (Appendix B.4) counts
   this for a 4-bit element: in the worst case no element of the block
   reaches the values 4 and 6, and the block uses 2.58 of E2M1's 3.58
   binades. ``UP`` pays this even when the block's largest element would have
   landed between 6 and 7 under :math:`\lfloor r \rfloor`, and so rounded to
   6 without being clipped.

``SELECTIVE``
   Take what ``NEAREST`` takes, except for blocks that the lower candidate
   would really clip; those get the upper one. Clipping does not always lose
   something. If E2M1's values went on past 6, the next one would be 8. An
   element that lands between 6 and 7 would then still round to 6, so
   clipping it to 6 is no worse than rounding it. An element above 7 would
   round to 8, which does not exist, and clipping it to 6 moves it further
   than any rounding would.

   The rule therefore uses a threshold :math:`T`: ``elem_max`` plus half the
   element format's step above it (7 for E2M1). If the block's largest
   element lands at or below :math:`T` under ``NEAREST``'s choice, the rule
   keeps that choice. Otherwise it takes :math:`\lceil r \rceil`. From a
   power of two its dividing line is :math:`T / \text{elem\_max}`, which is
   7/6 for E2M1. In the run, the block with 3.8 lands on 7.6 under the scale
   1/2 and gets the scale 1, while the block with 3.3 lands on 6.6 and keeps
   the scale 1/2. This rule comes from no specification; it is MPTorch's
   combination of the two before it.

Every rule's result is kept within the scale format's codes, and a block of
zeros takes the smallest scale (code 0 for E8M0).

A rule need not stay between the two candidates. The three `Four Over Six`_
rules below may take a scale about 1.5 times :math:`r`.

By default ``scale_rounding`` is the rule of the scale format's own
specification: ``OCP`` for a power of two and ``NEAREST`` for a BinaryK with
a mantissa. A :class:`~mptorch.SuperFP` scale defaults to ``SELECTIVE``,
because its values below its normal binades are powers of two, where
rounding to nearest clips the most (see `SuperFP elements and scales`_). The
format's ``scale_rule`` is the rule in effect, and formats compare by it.

Scales with a mantissa, and the tensor scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A scale format with a mantissa (NVFP4's E4M3, or a SuperFP) comes with a
tensor scale :math:`S_t` (see `Tensor scales`_). An element is then stored as
:math:`x_i \approx e_i\, s\, S_t`, and the scale the block asks for is
measured relative to :math:`S_t`:

.. math::

   r = \frac{a}{\text{elem\_max} \cdot S_t} .

The rules above choose between its two neighbours in the scale format,
:math:`\lfloor r \rfloor` and :math:`\lceil r \rceil`, as before. The chosen
:math:`s` stays inside the scale format: at most its largest value (448 for
E4M3) and at least its smallest positive one. The block's scale is the
float32 product :math:`s \cdot S_t`.

By default :math:`S_t` is computed from the tensor :math:`x` itself:

.. math::

   S_t = \frac{\max|x|}{\text{elem\_max} \cdot \text{scale\_max}} ,

where ``scale_max`` is the scale format's largest value, so
:math:`S_t = \max|x| / (6 \cdot 448)` for NVFP4. The block that holds the
tensor's largest magnitude then asks for exactly ``scale_max``. It takes the
largest scale and every other block a smaller one, so the scale format's
whole range lies below the tensor's largest block. This is the NVFP4 paper's
choice (Appendix B.1 writes its reciprocal, the "global encode scale"
:math:`6 \cdot 448 / \max|x|`). Passing ``tensor_scale=`` gives a fixed value
instead.

That range is finite. A block gets a scale of its own only while the scale
it asks for is at least the scale format's smallest positive value,
:math:`s_{\min}`:

.. math::

   r \ge s_{\min}
   \quad\Longleftrightarrow\quad
   a \;\ge\; \text{elem\_max} \cdot s_{\min} \cdot S_t
     \;=\; \max|x| \cdot \frac{s_{\min}}{\text{scale\_max}} ,

where the last step uses the default :math:`S_t`. For E4M3,
:math:`s_{\min} = 2^{-9}` and ``scale_max`` is 448, so a block's largest
magnitude may be up to :math:`448 \cdot 2^{9} \approx 2.3 \cdot 10^{5}` times
smaller than the tensor's. A block smaller than that still takes
:math:`s_{\min}`. Its largest element then lands at
:math:`\text{elem\_max} \cdot r / s_{\min}`, below ``elem_max``, and its
elements are rounded coarsely or to zero.

With a mantissa the choice of rule matters less than with a power of two,
because the two candidates are close together. In E4M3's normal range, when
``NEAREST`` takes :math:`\lfloor r \rfloor` it is at most a factor of 1.0625
below :math:`r`. E2M1's top step (4 to 6) absorbs that, so ``SELECTIVE``
leaves NVFP4's scales alone there. An element format with a finer top step
(E2M3, E4M3) is clipped by that factor, and ``SELECTIVE`` gives those blocks
:math:`\lceil r \rceil` instead.

Four Over Six
^^^^^^^^^^^^^

The rules so far all aim the block's largest element at ``elem_max`` and
choose the scale from the block's largest magnitude alone. `Four Over Six
<https://arxiv.org/abs/2512.02010>`__ (Cook et al.) questions that aim. E2M1's
largest values are 3, 4 and 6. In a block whose largest element sits on 6,
nothing between 67% and 100% of that element can be represented, and an
element near 5 is off by a sixth of the block's largest value after
rounding. The paper finds that these near-maximal elements account for most
of NVFP4's quantization error (its Section 2.2). If the block's largest
element sits on 4 instead, the block gives up the value 6 but gains 3 at 75%
of its largest value, and its other values are spaced more evenly.

A block therefore has a third candidate scale, and the rule chooses per
block between two of the three:

- the scale ``NEAREST`` takes, the candidate nearer to :math:`r`, under which
  the block's largest element lands on ``elem_max`` (6 for E2M1);
- the scale nearest to :math:`1.5\, r`, under which it lands on 4. In
  general the factor is ``elem_max`` over the largest power of two not above
  it, so 1.875 for E2M3 (7.5 and 4).

The rule quantizes the block under both scales in the call's rounding mode,
decodes both results, and measures each one's error against the block's
original values. Let :math:`\hat{x}^{(6)}_i` and :math:`\hat{x}^{(4)}_i` be
the decoded values under the first and the second scale, and
:math:`E^{(6)}`, :math:`E^{(4)}` their errors. The block takes the second
scale if

.. math::

   E^{(4)} < E^{(6)} ,

and keeps the first otherwise, including on a tie. There are three rules,
one per error measure, and the paper reports which worked best where:

``FOUR_OVER_SIX_MSE``
   The sum of squared errors,
   :math:`E^{(k)} = \sum_i \big(\hat{x}^{(k)}_i - x_i\big)^2`. Best in most
   of the paper's post-training quantization experiments (its Section 3.1).
``FOUR_OVER_SIX_MAE``
   The sum of absolute errors,
   :math:`E^{(k)} = \sum_i \big|\hat{x}^{(k)}_i - x_i\big|`. Better than MSE
   in its pre-training experiments (its Appendix A.1).
``FOUR_OVER_SIX_ABS_MAX``
   The largest absolute error of the block,
   :math:`E^{(k)} = \max_i \big|\hat{x}^{(k)}_i - x_i\big|`.

In the paper's example (its Table 2, and the run below), the block
:math:`[10, 20, 30, 40]` has the scale 6.5 under ``NEAREST`` and decodes as
:math:`[9.75, 19.5, 26, 39]`. Its second candidate is the scale 10, under
which it is exact, so every measure takes it.

Four things are specific to these rules:

- **The scale format needs a mantissa.** E4M3 can hold two scales a factor
  of 1.5 apart and E8M0 cannot, so a Four Over Six rule on a power-of-two
  scale raises. The paper has the same limit: the method does not apply to
  MXFP4 (its Section 5.2).
- **The default tensor scale changes.** It gives the tensor's largest block
  the scale 256 instead of 448, so that this block also has a second
  candidate: 256 is the largest scale whose multiple by 1.5 is still an E4M3
  value (384). This follows the paper's Appendix A.2. The format's
  ``top_scale`` is that scale.
- **Errors are counted in fixed point.** In the sums above, each
  :math:`|\hat{x}^{(k)}_i - x_i|` is first truncated to a multiple of
  :math:`2^{\lfloor \log_2 a \rfloor - 22}`, at most a four millionth of the
  block's largest magnitude, and the measure is added up in integers. A
  block's error is then exact. The CPU and a GPU add the elements up in
  different orders, and with exact errors they always choose the same scale;
  with a sum of floats they could differ at a near tie.
- **Stochastic rounding uses one draw per element.** Under ``SR`` both
  candidates round an element with the same random draw, the one the element
  is stored with. The choice is made on the rounded values, as in the paper.
  The paper notes that this favours the smaller error and so reintroduces a
  bias (its Appendix A.3).

The rule rounds every element a second time in each block whose two scales
differ. On a CPU, ``block_quantize`` takes about 2.2 times as long as under
``NEAREST``. For pre-training the paper recommends the rule for activations
only (its Appendix A.4).

With `2D tiles`_ (below) the block is the whole tile. The tile has one pair
of candidate scales, from its largest magnitude; the error is taken over all
of its elements; and it makes one choice. The paper's `reference
implementation <https://github.com/mit-han-lab/fouroversix>`__ does the same
for 16 x 16 tiles (its ``block_scale_2d``). Since the errors are integers,
the choice does not depend on the order in which they are added up, and a
square tile makes the same choice whichever axis the tensor is packed along.

Rounding the elements
^^^^^^^^^^^^^^^^^^^^^

Each element is divided by its block's scale. The division is exact for a
power of two and is done in float32 for a scale with a mantissa. The quotient
is rounded by the element format's cast (a :class:`~mptorch.BinaryK`'s or a
:class:`~mptorch.SuperFP`'s) in the call's rounding mode, under
``SaturationMode.SAT_FINITE``, and then clamped to ``elem_max``. With
:math:`Q` for that cast and :math:`\sigma` for the block's scale (:math:`s`,
or :math:`s\, S_t` for a scale with a mantissa):

.. math::

   e_i = \operatorname{sign}(x_i) \cdot
         \min\!\big( |Q(x_i / \sigma)|,\ \text{elem\_max} \big) .

Every value above ``elem_max`` lands on it, in every rounding mode. The
rounding mode applies to the elements only; the rule for the scale never
depends on it.

The clamp is where OCP's element formats differ from P3109's. As a P3109
format, E4M3 has the largest finite value 480, but OCP spends that code on
NaN. ``MXFP8_E4M3`` therefore has ``elem_max=448.0`` and ``nan_code=0x7F``.

Decoding, and the special values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Decoding an element is one float32 product, its value times its block's
scale:

.. math::

   \hat{x}_i = e_i\, \sigma, \qquad
   \sigma = s \ \text{ for a power-of-two scale}, \quad
   \sigma = s\, S_t \ \text{ for a scale with a mantissa},

where :math:`s\, S_t` is itself a float32 product, computed once per block.
The product :math:`e_i\, \sigma` is exact for every OCP format. Two rules
hold for every block format:

* **Non-finite values.** An infinity saturates to ``elem_max``. With an E8M0
  scale its block takes the largest scale, :math:`2^{127}`. Since
  :math:`448 \cdot 2^{127}` is past binary32, the element decodes to an
  infinity again. A NaN takes the element's NaN code when the format has one
  (E4M3, E5M2). Otherwise the block's *scale* becomes the scale's NaN and the
  whole block decodes to NaN; for a 2D tile (below) that is up to 16,384
  elements. With neither a NaN code nor a scale, a NaN becomes ``elem_max``
  and is lost. The OCP specification leaves the conversion of a NaN to the
  implementation for the formats without a NaN code (Sections 5.3.2 and
  5.3.3); these are MPTorch's choices.
* **One zero.** An element code with only its sign bit set is OCP's
  :math:`-0`. MPTorch never writes it and reads it as :math:`+0`, so no block
  op returns ``-0.0``, like every other cast in the library.

Quantizing twice
^^^^^^^^^^^^^^^^

Quantizing an already quantized tensor does not always return it unchanged,
because the second pass chooses its scales from the rounded values:

- With a power-of-two scale, ``OCP``, ``UP`` and ``SELECTIVE`` change nothing
  the second time.
- ``NEAREST`` with a power-of-two scale can move values. If the largest
  element was rounded down, the block now asks for a smaller scale, which can
  round to the next power of two down. Under that scale the element
  saturates.
- ``UP`` with a mantissa scale can also move values, and so can any rule
  with a mantissa scale when the element format is finer than E2M1.
- A Four Over Six rule changes nothing the second time on NVFP4. A block
  comes back with its largest element on the value its scale was chosen for,
  and under that scale it has no error at all.

The run
^^^^^^^

The first four lines of the output show the example block above (largest
magnitude 3.8, :math:`\rho = 1.27`) and a second block whose largest
magnitude is 3.3, so :math:`\rho = 1.1`. That is below every rule's dividing
line except ``UP``'s. Under the scale 1/2 its largest element is 6.6, which
rounds to 6 without being clipped, so only ``UP`` gives this block the scale
1. The table after them is the mean relative error of a block under each
rule, over Gaussian blocks of every magnitude.

The last part is Four Over Six on NVFP4. It shows the paper's block under
``NEAREST`` and under ``FOUR_OVER_SIX_MSE``. Then, over Gaussian blocks, it
shows how often each measure takes the second scale and the mean relative
error that leaves, next to ``NEAREST``'s. The same table follows for 16 x 16
tiles of Gaussian values. A tile of 256 such values has most of them far
below its largest, where the first scale's grid is the finer one, so the MSE
and MAE measures rarely take the second scale there; the last column says
that a tile's choice is the same along either axis.

.. literalinclude:: ../snippets/block_scale_rounding.py
   :language: python
   :caption: docs/snippets/block_scale_rounding.py

.. literalinclude:: ../snippets/block_scale_rounding.out
   :language: text
   :caption: output

SuperFP elements and scales
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~mptorch.SuperFP` element or scale spends the codes that a binaryK
spends on subnormals, and more, on powers of two below its normal binades:
the *supernormals*.

- **As a block element**, it trades the uniform steps near the top of a block
  for reach at the bottom. Small elements of a block with a wide spread
  survive where E2M1 flushes them to zero.
- **As a scale**, it reaches far below E4M3's :math:`2^{-9}` in the same
  byte.

Its codes follow the format's own layout: zero, then the supernormals from
code 1, then the normal binades as in a binaryK. It rounds by the superfp
casts, so a tie between two supernormals goes to the power of two with an
even exponent, as in :func:`~mptorch.quant.superfp_quantize`. A superfp
scale is cast like E4M3 and multiplied by a tensor scale, and its all-ones
code is NaN. Its default rule is ``ScaleRounding.SELECTIVE``, because below
its normal binades a superfp scale holds only powers of two, and rounding
such a scale to nearest can clip a block's largest element by up to a third.
The paragraph after the run gives the details.

.. literalinclude:: ../snippets/block_superfp.py
   :language: python
   :caption: docs/snippets/block_superfp.py

.. literalinclude:: ../snippets/block_superfp.out
   :language: text
   :caption: output

Rounding a scale to nearest costs more here. Below its normal binades a
superfp scale holds only powers of two, so the nearest scale can be a factor
of 1.5 below :math:`a / \text{elem\_max}`. Take a block whose ratio is 0.35:

- Under ``NEAREST`` it gets the scale 0.25. Its largest element is 8.4 in
  E2M1's units, saturates to 6, and decodes 29% low (33% at worst).
- Under ``SELECTIVE`` it gets 0.5, because 8.4 is past 7, and the element
  decodes 5% low.

E4M3's subnormals, below :math:`2^{-6}`, behave the same way, and NVFP4's
tensor scale keeps blocks away from them. The block GEMM decodes superfp
operands like any others, so their element codes are at most 8 bits there
too (see :doc:`kernels/block`).

2D tiles
~~~~~~~~

With ``block_rows > 1`` a scale is shared by a *tile* of ``block_rows`` rows
of ``block_size`` elements. The rows run along the axis before the packed
one, never across a batch dimension. A tile changes only where the scale
comes from: the codes and their layout are the 1D format's, and ``scales``
has one row per tile. Two examples:

- ``dataclasses.replace(NVFP4, block_rows=16)`` is NVFP4 with 16 x 16 tiles,
  the weight format of NVIDIA's NVFP4 pretraining recipe (which scales
  activations and gradients 1 x 16).
- 128 x 128 tiles are DeepSeek-V3's weight blocking, here with an E8M0 scale
  (a scale is one byte).

A *square* tile covers the same elements whichever of a matrix's two axes is
packed, so it quantizes the matrix to the same values either way, in every
deterministic rounding mode and under every scale rule, the `Four Over
Six`_ rules included. One packed weight can therefore serve both a
layer's forward pass and its input gradient, which reads the weight
transposed. A 1D block runs along one axis only and quantizes the two
orientations differently. A matrix product needs its operands blocked along
the dimension it sums over, because only then can each block's scales be
factored out of that block's part of the sum, which is how block-format
hardware computes it (see :doc:`kernels/block`, "Why the blocks run along K").

.. literalinclude:: ../snippets/block_tiles.py
   :language: python
   :caption: docs/snippets/block_tiles.py

.. literalinclude:: ../snippets/block_tiles.out
   :language: text
   :caption: output

Using block formats
~~~~~~~~~~~~~~~~~~~

Two other pages cover how block formats are used:

- :doc:`quantizers`, in its section "Block quantization and packed storage",
  shows the functions that round a tensor to a block format, and how its
  element and scale codes are packed into bytes.
- :doc:`kernels/block` shows how two block-format tensors are multiplied,
  including where the product applies their tensor scales.

Rounding modes
--------------

Every quantizer and every GEMM takes a :class:`~mptorch.RoundMode`. Let
:math:`\lfloor x \rfloor_F` and :math:`\lceil x \rceil_F` be the nearest
representable values below and above :math:`x`. Then
:math:`\lfloor x \rfloor_F \le x \le \lceil x \rceil_F`, with equality when
:math:`x` is representable. The modes are:

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

``RNE`` is the default of IEEE arithmetic and is unbiased on average. ``RZ``
is what cheap hardware does, and it is biased toward zero. ``RO`` is not a
mode to compute in. Its use is in two-step rounding: rounding to odd at an
intermediate width and then to nearest at the final one avoids the *double
rounding* error that two nearest roundings can make.

.. literalinclude:: ../snippets/rounding_modes.py
   :language: python
   :caption: docs/snippets/rounding_modes.py

.. literalinclude:: ../snippets/rounding_modes.out
   :language: text
   :caption: output

Stochastic rounding
~~~~~~~~~~~~~~~~~~~

Under ``SR`` the rounding direction is random, with a probability that
depends on where :math:`x` sits between its two neighbours. Let
:math:`f = (x - \lfloor x \rfloor_F) / (\lceil x \rceil_F - \lfloor x \rfloor_F) \in [0, 1)`
be that fraction. The ideal rule is :math:`\Pr(\text{up}) = f`, which makes
the rounding *unbiased*: :math:`\mathbb{E}[Q(x)] = x`.

The implementation draws :math:`N` random bits (``prng_bits``), adds them
just below the last kept mantissa bit, and truncates. The probability is
therefore quantized to steps of :math:`2^{-N}`:

.. math::

   \Pr(\text{up}) = 1 - \frac{\lceil (1 - f)\, 2^{N} \rceil}{2^{N}}
   \;\xrightarrow{\;N \to \infty\;}\; f .

This is P3109's ``StochasticA`` rounding with :math:`N` random bits. Let
:math:`\eta` be the fraction of :math:`|x|` between its neighbours
(:math:`\eta = f` when :math:`x > 0`). The value is rounded away from zero
when :math:`\lfloor \eta\, 2^{N} \rfloor + R \ge 2^{N}`, for a uniform
integer :math:`R \in [0, 2^{N})`.

``prng_bits`` is set on the format (``BinaryK(8, 4, prng_bits=8)``) and is
ignored by every other rounding mode. Where the bits are drawn has two
practical consequences:

- With ``prng_bits=0`` nothing is random, and ``SR`` becomes ``RZ``. The
  ``SR`` row above comes from a format with no random bits and is identical
  to the ``RZ`` row.
- The format's mantissa plus its random bits must fit in the carrier's
  mantissa: 23 bits in binary32, 52 in binary64. Each call checks this.

A float64 operand is rounded in binary64, and its bits are drawn there too,
two random words per draw.

The random streams are seeded from PyTorch's default generator, so
``torch.manual_seed`` makes a stochastic run reproducible. In a GEMM each
output element has its own stream, keyed on its position, so the result does
not depend on how the work is split across threads.

.. literalinclude:: ../snippets/rounding_stochastic.py
   :language: python
   :caption: docs/snippets/rounding_stochastic.py

.. literalinclude:: ../snippets/rounding_stochastic.out
   :language: text
   :caption: output

The output has two parts. The table shows the probability converging to the
exact fraction as the number of random bits grows. With 1 to 3 bits,
:math:`\lceil 0.4 \cdot 2^N \rceil / 2^N` is :math:`1/2`, so the coin is fair
whatever the residual.

The lines after the table show why the extra variance can be worth it. They
add :math:`0.001` to :math:`1.0` two thousand times in ``Binary16p8``, which
has bfloat16's field widths, in 1000 independent accumulators. The exact sum
is 3. An accumulator whose increments are smaller than half its spacing
never moves under round-to-nearest, so the ``RNE`` accumulators all stay at
:math:`1.0`. Stochastic rounding advances the sum by the right amount *on
average*: the ``SR`` accumulators have a mean of 2.95, and range from 2.5 to
3.47. A small weight update applied to a large weight is the same situation (see :doc:`tutorial` for an example).

Carriers
--------

Every operation is computed in a *carrier*, an IEEE binary float with far
more precision and range than the format it simulates. There are two
carriers, binary32 and binary64, and by default a tensor's dtype picks one:

- A **float32**, **float16** or **bfloat16** tensor is computed in
  **binary32**. A float16 or bfloat16 result is then stored back in its own
  dtype, which is a second rounding (see `What a narrower tensor can store`_).
- A **float64** tensor is computed in **binary64** throughout. An elementwise
  quantizer rounds each value once, directly. A matrix product computes every
  partial product and every running sum in binary64 before rounding it.

Moving a model to float64 therefore does more than widen its storage:

- Formats that binary32 cannot hold become simulable: up to 53 bits of
  precision, ten exponent bits, values down to :math:`2^{-1021}`.
- A value that binary32 cannot hold is rounded on its own bits, not on the
  nearest float32.

The two carriers agree wherever both hold every intermediate exactly.
Rounding a float64 tensor of float32 values to a format binary32 holds gives
exactly the float32 answer. So does a matrix product whose every product and
sum is exact in binary32, which is usual once a quantizer has rounded its
operands to a narrow format.

Every function and object that computes takes a ``carrier`` keyword to
override the default. Its value is the ``torch.dtype`` whose arithmetic the
carrier uses: ``torch.float64`` for binary64, ``torch.float32`` for binary32,
and ``None`` (the default) for the tensor's own.

``carrier=torch.float64`` gives a float32, float16 or bfloat16 tensor
binary64's arithmetic. Its operands are widened to float64, computed and
rounded in binary64, and the result is narrowed back to the tensor's dtype
with a single rounding. This is bit for bit the float64 call on the widened
tensor, then stored. The result keeps the tensor's dtype either way.

A carrier is never narrower than its tensor, so ``carrier=torch.float32`` on
a float64 tensor raises: narrowing the operands would round every input once
before the format does.

The keyword is taken by the quantizers
(:func:`~mptorch.quant.binaryK_quantize`,
:func:`~mptorch.quant.superfp_quantize`, :class:`~mptorch.quant.Quant`), the
matrix products (:class:`~mptorch.quant.SplitMac`,
:class:`~mptorch.quant.FusedMac` and the flat ``*_matmul`` functions) and the
layer factories (:func:`~mptorch.quant.binaryK_gemm_formats` and its
siblings).

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

(\*) ``NORMALS`` and ``EXTENDED_NORMALS``. The bound is one binade higher
(:math:`2^{-125}` and :math:`2^{-1021}`) at :math:`P = 1`, and for
``EXTENDED_NORMALS`` at the carrier's full precision, :math:`P = 24` and
:math:`P = 53`.

The last row is a cost worth knowing. On a GPU, binary64 arithmetic is
several times slower than binary32, so a float64 model's matrix products are
too (on the CPU the two are within a few percent). A narrower tensor that
names binary64 pays that cost, plus a float64 copy of each operand and a
narrowed copy of the result. In the other direction the choice is one of
dtype, not of carrier: a float64 model that needs only formats binary32 can
carry gets binary32's speed by being a float32 model.

The rest of this section derives the table's bounds and says what happens to
a format that exceeds them.

What the carrier can hold
~~~~~~~~~~~~~~~~~~~~~~~~~

The carrier has "far more" precision and range than every format in this
guide, but not than every possible format. The carrier sets a bound on the
format's parameters. A format outside that bound is quantized *partially*:
the result is still a tensor of plausible numbers, but some of the format's
values never appear in it. There are three limits, stated here in the
format's own terms, with binary64's in parentheses:

* **Precision.** :math:`P \le 24` (53) for :class:`~mptorch.BinaryK`, and
  ``man_bits <= 23`` (52) for :class:`~mptorch.SuperFP`. These are the
  carrier's significand bits; it cannot hold a finer grid.
* **The top.** The largest finite value must be at most the carrier's. That
  is :math:`\text{bias} \ge 2^E - 128` (:math:`2^E - 1024`), where :math:`E`
  is the exponent width (``exp_bits`` for a SuperFP). Above that the cast
  saturates at the carrier's largest value on the format's grid, and the
  codes above it are unreachable.
* **The bottom.** A subnormal carrier value has exponent field 0 whatever its
  magnitude. Any part of a cast that reads that field therefore cannot tell
  values apart below the carrier's smallest normal, :math:`2^{-126}`
  (:math:`2^{-1022}`). The limit depends on which parts of the cast read it:

  - BinaryK's subnormals and SuperFP's supernormals are placed on their grid
    using that field. Their smallest value must be at least :math:`2^{-125}`
    (:math:`2^{-1021}`), one binade above the carrier's smallest normal. That
    is :math:`\text{bias} + P \le 127` (1023) for BinaryK, and
    :math:`e_{\max} - b + 2 - n \ge -125` (:math:`-1021`) for SuperFP (using 
    the notations from `SuperFP: precision at the top, range below`_).
  - The two non-P3109 subnormal modes, ``NORMALS`` and ``EXTENDED_NORMALS``,
    find their bottom by comparing the input's magnitude with the format's
    smallest value :math:`x_{\min}`, which reads no field. They are exact as
    long as :math:`x_{\min}` is at least the carrier's smallest normal.
    Below that, :math:`x_{\min}` is not a normal carrier value and nothing is
    flushed to zero at all.
  - In two cases :math:`x_{\min}` must be one binade higher, at least
    :math:`2^{-125}` (:math:`2^{-1021}`). Under rounding to nearest, the
    comparison also needs the midpoint :math:`x_{\min} / 2` between zero and
    :math:`x_{\min}`, and in these two cases the carrier cannot place it:

    - **At** :math:`P = 1`. The format has one value per binade, and the
      rounding before the comparison works on the exponent field. With
      :math:`x_{\min} = 2^{-126}`, the midpoint :math:`2^{-127}`
      (:math:`2^{-1023}`) is a carrier subnormal, in the exponent field that
      all subnormals share. The rounding reads it as a tie between two
      binades and returns :math:`x_{\min}`, where round-to-nearest-even
      should return zero.
    - **An** ``EXTENDED_NORMALS`` **format at the carrier's full precision**,
      :math:`P = 24` (53). With its lowest binade starting at
      :math:`2^{-126}`, its smallest value is
      :math:`x_{\min} = 2^{-126}\,(1 + 2^{-23})`. The midpoint,
      :math:`2^{-127}\,(1 + 2^{-23})`, needs a step of :math:`2^{-150}`
      (:math:`2^{-1075}`), which is finer than the carrier's finest step,
      :math:`2^{-149}` (:math:`2^{-1074}`). The carrier holds
      :math:`2^{-127}` in its place. An input of exactly :math:`2^{-127}` is
      below the true midpoint and should round to zero, but it now looks
      like a tie, and round-to-nearest-away returns :math:`x_{\min}`.

The last two limits together mean that a simulated format may span at most
253 binades (2045). So **seven exponent bits is the most either family can
carry in binary32, and ten in binary64**. At P3109's default bias that is
:math:`K - P \le 7` (10) for a signed BinaryK and :math:`K - P \le 6` (9) for
an unsigned one.

The 8-bit and 6-bit formats of this guide are well inside binary32's bounds.
A format with eight exponent bits is not, whatever its bias:

- ``Binary16p8`` and ``Binary32p24`` reach below binary32's floor. The kernel
  pages use ``Binary16p8`` as an accumulator and silence the warning
  described below.
- The bfloat16 and float32 layouts also reach past its top.

binary64 holds all four.

The carrier depends on the tensor, not on the format, so the limits are
checked on every call, against the carrier that the call rounds in. The
quantizers of :doc:`quantizers`, the kernels of :doc:`kernels/index` and the
layers built on them all do this. When a :class:`~mptorch.BinaryK` or
:class:`~mptorch.SuperFP` is built, it checks only what neither carrier can
do. The plain-integer wrappers never see a format object, and check the same
thing when they resolve their format. There is one rule for both, so
``binaryK_matmul(a, b, mul_K=16, mul_P=8)`` reports what
``qmatmul(a, b, BinaryK(16, 8))`` reports.

There are two severities. What separates them is whether the format still
works, not which limit it misses:

* A format whose **range** exceeds the carrier's warns with
  :class:`~mptorch.FormatRangeWarning`, naming the line that made the call.
  This covers a format that reaches above the carrier's largest finite value,
  is spaced finer than its smallest subnormal (:math:`2^{-149}`,
  :math:`2^{-1074}`), or has a smallest value below the floor given in the
  list above. Such a format quantizes correctly over the part the carrier
  holds, and some of its values are never returned. A wide BinaryK used as a
  precision-only target is this case, so the warning can be silenced with
  :func:`warnings.simplefilter` when that is intended.
* A format that **cannot function** raises :exc:`ValueError`. This covers
  more precision than the carrier's, stochastic-rounding bits with no
  significand left to draw from, an exponent field wider than the kernels
  shift by, a SuperFP whose ``normal_binades`` leaves no supernormal codes,
  and a format whose largest finite value is below the carrier's normals,
  where every nonzero result overflows.

The run below shows both carriers on the same inputs and formats:

- a float64 value rounded once in binary64, and the nearest float32 value
  rounded in binary32 and, with ``carrier=torch.float64``, in binary64;
- four formats, each checked against each carrier when a tensor meets it;
- a 30-bit format computed exactly on a float64 tensor;
- a product of float32 operands whose rounding only binary64 gets right.

.. literalinclude:: ../snippets/formats_carriers.py
   :language: python
   :caption: docs/snippets/formats_carriers.py

.. literalinclude:: ../snippets/formats_carriers.out
   :language: text
   :caption: output

A ``carrier=torch.float64`` call is held to binary64's bounds even on a
float32 tensor, because binary64 is what rounds it. Its result is also held
to what the tensor's dtype can store, which the next section describes.

``dev/benchmarks/format_limits.py`` reports all of this for a given format,
in binary32 or with ``--carrier binary64``. With ``--audit`` it checks the
kernels against the format's value set. Its ``sweep --audit`` walks every
bound in the table above across its edge; each was measured where this
section puts it.

What a narrower tensor can store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A float16 or bfloat16 tensor is rounded in binary32, its default carrier, and
the result is converted back when it is stored. The same happens to a float32
tensor under ``carrier=torch.float64``, whose result is rounded in binary64.
That conversion is a second rounding, onto the dtype's grid, and MPTorch
rounds it correctly to nearest.

For a float64 result stored as float16 or bfloat16, MPTorch rounds once,
directly from float64 onto the dtype's grid. It does not use PyTorch's
``.to()``, which converts float64 to bfloat16, and on most platforms to
float16, in two steps: first to float32, then to the dtype. Two roundings
can give a different result from one. Take the float64 value
:math:`1 + 2^{-8} + 2^{-30}`. It lies just above the midpoint of the
bfloat16 values :math:`1` and :math:`1 + 2^{-7}`, so it should become
:math:`1 + 2^{-7}`. PyTorch's first step rounds it to :math:`1 + 2^{-8}`,
which is exactly the midpoint, and its second step then rounds that tie to
:math:`1`.

A format whose results are not all values of the dtype is quantized
partially, as in the previous section, with one difference: a result above
the dtype's largest finite value is stored as :math:`\pm\infty` *whatever the
format's saturation mode*. A ``SAT_FINITE`` format overflows too.

The dtype depends on the tensor, not on the format, so this is checked on
every call. The check is made against the format whose values the result
holds, and the rule depends on what the call rounded.

**A GEMM's result** is the output of its last rounding: the accumulate
format of a :class:`~mptorch.quant.SplitMac`, or the fused format of a
:class:`~mptorch.quant.FusedMac`. That rounding is applied to products and
sums that can land anywhere, so every value of the format can occur as a
result, and all of them must be values of the dtype. For a BinaryK:

===========  ======================================  ===============  =======================================  ========
bound        float16                                 bfloat16         float32 (under binary64)                 severity
===========  ======================================  ===============  =======================================  ========
precision    :math:`P \le 11`                        :math:`P \le 8`  :math:`P \le 24`                         raises
top          largest value below :math:`2^{16}`      float32's        largest value below :math:`2^{128}`      warns
finest step  :math:`\text{bias} + P \le 26`          float32's        :math:`\text{bias} + P \le 151`          warns
             (``EXTENDED_NORMALS``: :math:`\le 25`)                   (``EXTENDED_NORMALS``: :math:`\le 150`)
===========  ======================================  ===============  =======================================  ========

A SuperFP is held to the same three bounds, in its own parameters. The rule
does not depend on the carrier that rounded the result.

- In binary32, only the precision bound is new for bfloat16. Its range is
  float32's to within its precision, so a format that passes the float32
  checks with at most eight bits of precision is inside it at both ends.
- In binary64, the carrier accepts ranges that no narrower dtype holds, so
  every row of the table can apply.

A SplitMac's multiply format is never stored, because its products are
intermediates in the carrier. A step left unrounded stores no format's values
at all.

**An elementwise quantizer's result** is the format's rounding of a value
that is already a value of the dtype. Wherever the format is at least as fine
as the dtype, the result is the input itself. So neither high precision nor a
fine bottom causes a problem; only the edges of the format's range can land
off the dtype's grid. Each of these cases warns:

* The format reaches past the dtype's largest finite value, and that value is
  not on the format's grid. An input near it then rounds up, out of the
  dtype's range.
* The format's largest finite value is inside the dtype's range but is not a
  value of the dtype, and the format saturates onto it.
* An ``EXTENDED_NORMALS`` format's hole, :math:`2^{-\text{bias}}`, is a value
  of the dtype, but the first value above it, where such an input rounds, is
  not.

Either kind of call raises for a format whose largest finite value is below
the dtype's smallest. Such a tensor could store nothing of the format but
zero.

**A block format's decoded values** are an element value times its block's
scale, computed in the carrier: binary64 for a float64 tensor or
``carrier=torch.float64``, binary32 otherwise.

- A :class:`~mptorch.BlockFormat`'s element and cast scale must be formats
  the carrier holds entirely, and a power-of-two scale's codes must lie in
  its range (:math:`2^{\pm 127}` in binary32, :math:`2^{\pm 1023}` in
  binary64). Building the format checks binary64's bounds. A call that
  rounds in binary32 checks binary32's, so an element with nine or ten
  exponent bits is a format of float64 tensors.
- A packed tensor remembers the carrier it was packed in, and unpacking or
  multiplying it runs in that carrier or a wider one.

A result narrower than its carrier (float16 or bfloat16, and float32 too in
binary64) rounds the decoded values a second time wherever the dtype does not
hold them, and :class:`~mptorch.FormatRangeWarning` reports it. That is always
the case for a format with a tensor scale (NVFP4), whose decoded values are
products in the carrier. For an E8M0 format it happens only
when its largest element value has more significant bits than the dtype.

.. code-block:: python

   x = torch.ones(4, 4, dtype=torch.float16)
   binaryK_quantize(x, 8, 4)                # fine
   binaryK_quantize(x, 8, 3, bias=15)       # FormatRangeWarning: rounds up past 65504
   binaryK_quantize(x, 16, 12, bias=8)      # fine: float16's values come back as they are
   binaryK_matmul_fma(x, x, fma_K=16, fma_P=12, fma_bias=8)  # ValueError: 12 bits
   qmatmul(x, x, SplitMac(BinaryK(8, 4), BinaryK(16, 11)))    # FormatRangeWarning: 2**-25

Both rules were measured against the kernels, not only derived.
``format_limits.py --dtype float16 --audit`` runs a format through a fused
GEMM over operands of the dtype, and through the quantizer over every value
of the dtype. ``sweep --dtype float16 --audit`` walks each boundary above.
With ``--carrier binary64`` both also go through the widened operands and the
narrowed result.

.. _apple-gpu:

On an Apple GPU
~~~~~~~~~~~~~~~

On macOS every op also runs on ``"mps"`` tensors, as Metal kernels compiled
from the same casts and GEMM policies that the CPU runs. MPS has no float64,
so the carrier there is always binary32. float32, float16 and bfloat16
tensors compute exactly as on the CPU. ``carrier=torch.float64`` is refused,
because the float64 copy of the operands cannot be made on the device.

The results are the CPU's, bit for bit, stochastic rounding included. An MPS
call draws its one seed from the CPU generator, as a CPU call does, and gives
every element the same random word. ``torch.manual_seed`` therefore
reproduces a run across the two devices, not only on one. (CUDA draws from
its own generator.)

The GPU's float unit differs from the CPU's in two ways. Both affect what a
GEMM computes *between* its roundings, not the roundings themselves:

* **Subnormals.** Apple GPUs flush binary32 subnormals to zero, in operands
  and results, in every mode Metal offers. The casts round on the word and
  are not affected. But a product, a running sum or a fused multiply-add
  whose exact value is below :math:`2^{-126}` in magnitude becomes zero
  before the format sees it. The effect depends on the rounding mode:

  - Under ``RZ`` the CPU returns zero too.
  - Under ``RNE`` and ``RNA`` it does as well, for every format whose
    smallest value is at least :math:`2^{-125}`.
  - Under ``RU``, ``RD``, ``RO`` and ``SR`` the CPU can round such a value to
    the format's smallest value where the GPU gives zero.
  - A sum left unrounded (``accumulate_quant=False``) can hold a subnormal on
    the CPU that the GPU drops.

  The quantizers are not affected. They round every input exactly, subnormal
  ones included, in every mode.
* **NaN payloads.** A NaN that reaches a product or a sum comes out of the
  GPU's float unit as its canonical NaN, where the CPU keeps the operand's
  payload. The casts pass a NaN through unchanged on every device, so a
  quantizer's NaNs keep their payload.

The first call with a given format compiles its kernel, which takes from
about 20 ms to a few hundred. Later calls with the same format, rounding mode
and dtype reuse it.

Where rounding is applied
-------------------------

A quantizer rounds a tensor, and a kernel rounds inside a reduction. The two
compose, and MPTorch keeps them separate on purpose:

- :doc:`quantizers` covers the functions and objects that round a tensor:

  - elementwise: :func:`~mptorch.quant.binaryK_quantize`,
    :func:`~mptorch.quant.superfp_quantize`, :class:`~mptorch.quant.Quant`
    and :class:`~mptorch.quant.Quantizer`, which apply one rounding per
    element;
  - by blocks: :func:`~mptorch.quant.block_quantize`,
    :func:`~mptorch.quant.block_pack`, :func:`~mptorch.quant.block_unpack`
    and :class:`~mptorch.quant.BlockQuant`, which choose one scale per block
    and then apply one rounding per element.

- :doc:`kernels/index` covers :class:`~mptorch.quant.SplitMac`,
  :class:`~mptorch.quant.FusedMac`, :class:`~mptorch.quant.BlockMac`,
  :func:`~mptorch.quant.qmatmul`, :func:`~mptorch.quant.block_matmul` and the
  convolutions. These apply one rounding per product and per addition, or
  per fused multiply-add, inside one matrix-product routine.
- :doc:`layers` covers ``QAffineFormats`` and the layers: which quantizer is
  applied to which signal, and which arithmetic each of a layer's three
  matrix products runs in.
