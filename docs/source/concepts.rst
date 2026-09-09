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

BinaryK: K bits, P of them precision
------------------------------------

:class:`~mptorch.BinaryK` describes a binary floating-point format the way
IEEE 754 does. A value has a sign :math:`s`, a biased exponent :math:`e`
of :math:`E` bits and a stored mantissa :math:`m` of :math:`P-1` bits; the
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
unsigned one, which has no sign bit to spend.

A few consequences worth having in mind:

- Within one *binade* :math:`[2^q, 2^{q+1})` there are :math:`2^{P-1}`
  representable values, spaced :math:`2^{q-P+1}` apart -- the *unit in the
  last place*, ulp. The relative spacing is therefore between
  :math:`2^{-P}` and :math:`2^{1-P}`: E4M3 (:math:`P=4`) represents numbers
  to about 6-12 %, E5M2 (:math:`P=3`) to about 12-25 %.
- The largest finite value is
  :math:`(2 - 2^{2-P}) \cdot 2^{\,2^E - 1 - \text{bias}}` under the
  ``OVF_INF`` and ``SAT_PROPAGATE`` overflow policies -- the very last code
  of the top binade is treated as non-finite, as in the OCP E4M3 format --
  and :math:`(2 - 2^{1-P}) \cdot 2^{\,2^E - 1 - \text{bias}}` under
  ``SAT_FINITE``, which uses it.
- The smallest normal is :math:`2^{1 - \text{bias}}` and the smallest
  subnormal :math:`2^{1 - \text{bias} - (P-1)}`.
- ``bias`` defaults to the middle of the exponent range, :math:`2^{E-1}`.
  IEEE formats use :math:`2^{E-1} - 1`; pass ``bias=`` explicitly when you
  want to match one exactly. A BinaryK format uses every exponent code for
  finite values where IEEE reserves the top one for infinities and NaNs, so
  with the IEEE bias it agrees with its namesake on every value up to the
  namesake's largest finite one (the :doc:`tutorial` checks this
  exhaustively for E4M3 and E5M2).

Some formats you may want, as BinaryK values:

=====================  ==============================  ======  ======  ======
format                 spelling                        E       P-1     bias
=====================  ==============================  ======  ======  ======
E4M3 (OCP / NVIDIA)    ``BinaryK(8, 4, bias=7)``       4       3       7
E5M2 (OCP / NVIDIA)    ``BinaryK(8, 3, bias=15)``      5       2       15
float16                ``BinaryK(16, 11, bias=15)``    5       10      15
bfloat16               ``BinaryK(16, 8, bias=127)``    8       7       127
float32                ``BinaryK(32, 24, bias=127)``   8       23      127
a 6-bit E3M2           ``BinaryK(6, 3)``               3       2       4
a 4-bit E2M1           ``BinaryK(4, 2)``               2       1       2
=====================  ==============================  ======  ======  ======

The following run constructs the two 8-bit formats with their default bias,
derives their ranges from the formulas above, and confirms them against the
quantizer -- each boundary value is a fixed point of the rounding, and
:math:`10^{30}` clamps to the largest finite value under ``SAT_FINITE``.

.. literalinclude:: ../snippets/formats_binaryK.py
   :language: python
   :caption: docs/snippets/formats_binaryK.py

.. literalinclude:: ../snippets/formats_binaryK.out
   :language: text
   :caption: output

The last two lines show the grid: in :math:`[1, 2)` E4M3 has the eight values
:math:`1, 1.125, \dots, 1.875`, and each input lands on the nearest one, ties
going to the even mantissa (:math:`1.0625 \to 1.0`, :math:`1.1875 \to 1.25`,
:math:`1.9375 \to 2.0`).

Subnormals, saturation and sign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two enums decide what happens at the ends of the range. Both are properties
of the *format*, so two formats in one computation can differ in them.

:class:`~mptorch.SubnormalsMode` -- the bottom of the range:

``SUBNORMALS``
   Gradual underflow, as in IEEE 754: the exponent code 0 encodes the
   subnormal numbers, and a value below the smallest of them rounds to zero
   (or to it, under a directed mode).
``NORMALS``
   No subnormals. Everything below the smallest normal, :math:`2^{1-\text{bias}}`,
   flushes to zero.
``EXTENDED_NORMALS``
   No subnormals either, but the exponent code they would have used encodes
   one more binade of *normal* numbers, :math:`[2^{-\text{bias}}, 2^{1-\text{bias}})`,
   with the full mantissa. Below that binade, flush to zero.

:class:`~mptorch.SaturationMode` -- the top:

``SAT_FINITE``
   Everything is clamped to the largest finite value, which here includes
   the top mantissa code: a finite input that overflows *and* an infinite
   input alike, so the output never contains an infinity. NaN inputs pass
   through unchanged.
``SAT_PROPAGATE``
   A finite input that overflows is clamped to the largest finite value with
   the top mantissa code excluded; an infinite input stays infinite.
``OVF_INF``
   IEEE behaviour: an input beyond the largest finite value becomes
   :math:`\pm\infty` (the default).

.. literalinclude:: ../snippets/formats_modes.py
   :language: python
   :caption: docs/snippets/formats_modes.py

.. literalinclude:: ../snippets/formats_modes.out
   :language: text
   :caption: output

An unsigned format (``is_signed=False``) has no sign bit; negative inputs
become zero and the bit goes to the exponent, which is why the unsigned
``BinaryK(8, 4)`` above has a bias of 16 and reaches 3072.

SuperFP: precision at the top, range below
------------------------------------------

:class:`~mptorch.SuperFP` is the second format family the kernels implement.
It has the same three fields as a binary float -- ``man_bits``, ``exp_bits``,
``bias`` -- but only the top ``normal_binades`` binades carry a mantissa. The
encodings of the remaining :math:`2^{\text{exp\_bits}} - \text{normal\_binades}`
binades, :math:`2^{\text{man\_bits}}` codes each, are reinterpreted as that
many further *powers of two* below the normal region: numbers with an
implicit mantissa of exactly 1. Writing :math:`e_{\max} = 2^{\text{exp\_bits}} - 1 - \text{bias}`,
the three regions are

.. math::

   \begin{aligned}
   \text{normal:}      &\quad [2^{\,e_{\max} - b + 1},\; 2^{\,e_{\max}+1}),
      && \text{full mantissa} \\
   \text{supernormal:} &\quad [2^{\,e_{\max} - b + 1 - n},\; 2^{\,e_{\max} - b + 1}),
      && \text{powers of two only} \\
   \text{underflow:}   &\quad \text{below that}, && \text{flushed to zero}
   \end{aligned}

with :math:`b = \text{normal\_binades}` and
:math:`n = (2^{\text{exp\_bits}} - b)\, 2^{\text{man\_bits}}`. There are no
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
