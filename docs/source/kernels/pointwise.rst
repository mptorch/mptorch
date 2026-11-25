Pointwise ops and activations
=============================

The pointwise ops are not matrix products, and they do not run on the core
routine of :doc:`index`. Each one computes with PyTorch's own arithmetic in a
*carrier*, binary32 or binary64, and then rounds the result once with a
quantizer. They cover the arithmetic a model writes in its own code, besides
its layers:

- the elementary ops :func:`~mptorch.quant.qadd`,
  :func:`~mptorch.quant.qsub`, :func:`~mptorch.quant.qmul`,
  :func:`~mptorch.quant.qdiv`, :func:`~mptorch.quant.qsqrt`,
  :func:`~mptorch.quant.qrsqrt`, :func:`~mptorch.quant.qexp` and
  :func:`~mptorch.quant.qpow` (a number exponent). The binary ones take two
  tensors of one float dtype, or a tensor and a Python number, broadcast as
  PyTorch broadcasts them;
- the activations :func:`~mptorch.quant.qgelu`,
  :func:`~mptorch.quant.qsilu`, :func:`~mptorch.quant.qsigmoid`,
  :func:`~mptorch.quant.qtanh`, :func:`~mptorch.quant.qhardswish`,
  :func:`~mptorch.quant.qhardsigmoid` and :func:`~mptorch.quant.qrelu`;
- :func:`~mptorch.quant.qembedding`, a gather whose rows are rounded.

:doc:`/layers` shows the modules built on them, and :doc:`/instrument` how a
recipe reaches a model's own arithmetic with them.

Slots
-----

What a step rounds to is its *slot*: a :class:`~mptorch.quant.Quant`, any
other callable ``Tensor -> Tensor``, or ``None``. An elementary op has two,
``quant`` for its result and ``grad_quant`` for the gradient of each tensor
operand:

.. code-block:: python

    from mptorch import BinaryK
    from mptorch.quant import Quant, qmul

    z = qmul(x, y, Quant(BinaryK(8, 4)), Quant(BinaryK(16, 11)))

An activation takes an :class:`~mptorch.quant.ActivationFormats`, whose
``inner`` rounds the gate of the three gated activations (GELU's
:math:`\operatorname{erf}(x/\sqrt 2)`, or its tanh approximation; SiLU's
:math:`\sigma(x)`; hardswish's hardsigmoid), ``out`` the result and ``grad``
the gradient. With a rounded gate the result is computed from it, as
:math:`\tfrac12 x(1 + g)` or :math:`x g`. A recipe's ``elementwise`` and
``embedding`` kinds take a :class:`~mptorch.quant.PointwiseFormats`, which
holds an elementary op's four arguments.

A step whose slot is ``None`` is PyTorch's own computation in the tensor's
dtype, so with every slot ``None`` an op is exactly PyTorch's function,
forward and backward.

What the carrier rounds
-----------------------

A step with a slot computes in the carrier: binary32 for float32, float16
and bfloat16 tensors (the narrow ones widened first), binary64 for float64
tensors, and binary64 whatever the dtype under ``carrier=torch.float64``. The
slot rounds the carrier's result, and the result is stored in the tensor's
dtype: from binary32 with one conversion, from binary64 with one rounding
(:doc:`/concepts`, "Carriers"). A Python number operand is converted to the
carrier first, as PyTorch converts it.

So the carrier rounds first, to nearest, and the format second, as in a
step of a matrix product (:doc:`arithmetic`, "What the carrier rounds"). The
result is the format's rounding of the exact value whenever the carrier's
result is on the exact value's side of every number the format's rounding
compares it with. Those *boundaries* have :math:`k` significant bits:

.. list-table::
   :header-rows: 1

   * - rounding mode
     - boundaries
     - :math:`k`
   * - ``RU``, ``RD``, ``RZ``, ``RO``
     - the format's values
     - :math:`P`
   * - ``RNE``, ``RNA``
     - the midpoints between them
     - :math:`P + 1`
   * - ``SR`` (StochasticA)
     - the values :math:`N` random bits below them
     - :math:`P + N`
   * - ``SR_B``, ``SR_C``
     - one bit further
     - :math:`P + N + 1`

for a format of precision :math:`P` and :math:`N` = ``prng_bits``. With
:math:`p` the operands' precision (24 for float32, 11 for float16, 8 for
bfloat16, or less for values of a narrower format), the result is the exact
value's rounding when:

.. list-table::
   :header-rows: 1

   * - op
     - binary32 (the default)
     - binary64
     - ``carrier_rounding=RoundMode.RO``
   * - ``qmul``
     - :math:`2p \le 24` (the product is exact)
     - always (exact)
     - always (exact)
   * - ``qadd``, ``qsub``
     - the exponents differ by at most :math:`23 - p` (the sum is exact)
     - they differ by at most :math:`52 - p`
     - always
   * - ``qdiv``
     - :math:`k \le 24 - p`
     - :math:`k \le 53 - p` (29 for float32 operands)
     - as binary64, or exactly in binary32
   * - ``qsqrt``
     - :math:`k \le 11`
     - :math:`k \le 25`
     - as binary64, or exactly in binary32
   * - ``qrsqrt``, ``qexp``, ``qpow``, the activations
     - not claimed
     - not claimed
     - refused

A quotient of float32 operands is more than :math:`2^{-(24+k)}` from a
:math:`k`-bit number in relative terms, unless it equals one, and a root more
than :math:`2^{-(2k+1)}`, while binary64 is off by at most :math:`2^{-53}`
(a root within an ulp, :math:`2^{-52}`, still meets the bound). The test
suite checks the table on every pair of an 8-bit format's values, and the
quotient's bound on the worst cases there are for it. ``qrsqrt``, ``qexp``,
``qpow`` and the activations run PyTorch's library functions, which are not
correctly rounded and differ between the CPU, CUDA and MPS: the format rounds
a value that may already be off by an ulp of the carrier.

Outside those conditions the carrier rounds first. The directed modes and
stochastic rounding can then return the wrong neighbour, and so can RNE on
operands that are not values of the format; on operands that are, RNE stays
correct in binary32 for a format of at most 11 bits (Figueroa's bound,
:math:`2P + 2 \le 24`). The snippet shows the default carrier's two
roundings, and the two ways to avoid them:

.. literalinclude:: ../../snippets/pointwise_exact.py
   :language: python
   :caption: docs/snippets/pointwise_exact.py

.. literalinclude:: ../../snippets/pointwise_exact.out
   :language: text
   :caption: output

PyTorch's eager square root on the CPU is not correctly rounded: it can be an
ulp off, in float32 and in float64, where CUDA's is correctly rounded. A
binary32 root is therefore computed through float64 on the CPU, whose root
within an ulp narrows to the correctly rounded float32 one, so that the
binary32 step is the same on every device.

Rounding to odd
~~~~~~~~~~~~~~~

``carrier_rounding=RoundMode.RO`` replaces the carrier's nearest step by the
exact result rounded to odd, as the matrix products' odd ops do
(:doc:`arithmetic`). It computes in float64: the product exactly, the sum
with the error 2Sum finds, moved one step toward it when the sum's last bit
is even, and the quotient and the root to nearest, which the bounds above
make sufficient. In binary32 that value is rounded to odd once more, at 24
bits. The cast then rounds the exact value, for every format that the
round-to-odd step's bounds admit (:math:`P \le 22` under RNE in binary32,
:math:`P \le 51` in binary64; the matrix products' odd ops state the others).
It takes float32, float16 and bfloat16 operands, whose results float64 holds,
and a number operand is first rounded to their dtype; float64 operands, MPS
tensors (which have no float64), a slot that is not a ``Quant`` of a
``BinaryK``, ``SuperFP`` or ``FixedPoint``, and ``qrsqrt``, ``qexp`` and
``qpow`` raise.

Gradients
---------

Each op's backward is the derivative of the unrounded function, PyTorch's
own formula (``derivatives.yaml``, or the activation's ``*_backward`` op),
computed in the carrier and rounded once by the gradient slot. The forward's
roundings pass the gradient through unchanged, as a straight-through
estimator does. A broadcast operand's gradient is summed over the dimensions
it was broadcast along, in the carrier, before that rounding. With no
gradient slot the backward is PyTorch's formula in the gradient's dtype,
which is the gradient autograd gives. Round to odd applies to the forward
step only.

An embedding's weight gradient is PyTorch's (``embedding_backward``), summed
over repeated indices in the carrier, then rounded once.

The result in place
-------------------

A ``Quant`` rounds an op's result in place, since the result is the op's own
tensor, and out of place when it is not contiguous (it keeps its operands'
layout). An op never writes its inputs. Each op is an autograd Function,
hidden from torch function modes as the layers' Functions are, and compiles
with ``fullgraph=True``. Under Inductor, PyTorch's arithmetic around the
roundings is Inductor's, which fuses an activation's multiply-adds.

What it costs
-------------

An op with no slot costs what PyTorch's does. With a slot, it costs the op
plus one pass of the quantizer: 1.7-2x PyTorch's forward in float32 on an
RTX 4060, and 3-5.5x on one Ryzen CPU thread (the square root more, since it
is computed through float64 there). A narrow dtype adds the widening
and the store, a rounded gate its own pass, binary64 the float64 arithmetic,
and rounding to odd a dozen float64 passes
(``dev/benchmarks/timing/pointwise_ops.py``).

Lookup-table activations
------------------------

A :class:`~mptorch.quant.PWLTable` is a continuous piecewise-linear function
on uniform knots: one entry at each of the ``segments + 1`` knots of
``[lo, hi]``. ``ActivationFormats(table=...)`` makes an activation evaluate
the table in place of its function, and :func:`~mptorch.quant.qpwl` and
:class:`~mptorch.quant.QPWLActivation` evaluate a table of a function of
your own.

**Evaluation.** In the carrier, ``x`` is evaluated as

.. math::

   u = (x - lo) \cdot \frac{n}{hi - lo}, \quad
   i = \operatorname{clamp}(\lfloor u \rfloor, 0, n - 1), \quad
   f = u - i, \quad
   y = e_i + f\,(e_{i+1} - e_i),

each operation rounded by the carrier and written out rather than taken from
``torch.lerp``, whose formula changes above :math:`f = 1/2`. So every device
gives the same words. ``out`` rounds :math:`y`. Outside ``[lo, hi]``,
``tails`` gives ``"extend"`` (the end segments' lines), ``"clamp"`` (the end
entries), or a callable's value: ``silu_table`` and ``gelu_table`` use
``torch.relu``, as LoFloat's ``PWLSiLU`` does.

**The fits.** :meth:`~mptorch.quant.PWLTable.fit` places the entries in one
of three ways.

- ``"chord"``: the function at each knot.
- ``"balanced"``: LoFloat's correction (its ``init_lut``, which it calls
  minimax). Each entry moves by the mean of its two segments' midpoint
  errors, halved at the ends. ``silu_table(fit="balanced")`` is LoFloat's
  ``PWLSiLU``.
- ``"minimax"`` (the default): the entries whose largest error over 64
  points a segment, knots included, is the smallest any entries reach.

  - **How it is found.** For an error bound :math:`E`, each point bounds
    its segment's two entries to a strip. The feasible values of each entry
    form an interval, carried from knot to knot. Bisection on :math:`E`
    finds the smallest bound that leaves every interval nonempty, and a walk
    back from the last knot picks the entries.
  - **What it guarantees.** The fit is the optimum on its grid, to a
    relative 1e-13, and it is proven, not converged to.
  - **Error measures.** ``error="relative"`` holds the relative error
    instead, which :meth:`~mptorch.quant.PWLTable.rsqrt`'s table uses.

``entry_quant`` rounds the fitted entries; a minimax table so rounded is no
longer minimax. The largest errors, measured on 1,024 points a segment, are
in the run below. Balanced and minimax come close at 64 segments, where
LoFloat's correction is near the optimum.

**The gradient** is the activation's own: the derivative of the function the
table approximates, computed in the carrier and rounded by ``grad``. A
table's own slope is piecewise constant and a poor gradient. A ``qpwl``
table's gradient is ``derivative(g, x)``, or autograd's derivative of the
table's ``fn``.

An activation refuses a table of another activation (a SiLU table given to
``qgelu``), a table together with ``inner``, and a table of 1/sqrt, which is a
norm's ``inverse`` (:doc:`reductions`). A table of a function of your own,
which has no ``name``, goes anywhere.

**What a table costs.** A table is evaluated with PyTorch's own ops: the
index, the fraction, two gathers, the interpolation and the tails, then the
slot's cast. Eager, that is about a dozen passes over the tensor. Compile it
for speed: ``torch.compile`` fuses the evaluation into one kernel, whose cost
is PyTorch's fused activation's, and only the rounding stays a pass of its
own. On ``[4096, 4096]`` float32, against ``F.silu``'s 0.65 ms on CUDA (an
RTX 4060 laptop GPU) and 0.78 ms on the CPU (16 threads, ``[2048, 2048]``),
in ms per call:

====================================  ==========  =============  =========  ============
``qsilu`` with ``silu_table()``       CUDA eager  CUDA compiled  CPU eager  CPU compiled
====================================  ==========  =============  =========  ============
the table                             15.8        0.71           59.1       1.09
the table, ``out`` rounded            14.5        1.28           63.5       2.18
====================================  ==========  =============  =========  ============

.. literalinclude:: ../../snippets/pointwise_tables.py
   :language: python
   :caption: docs/snippets/pointwise_tables.py

.. literalinclude:: ../../snippets/pointwise_tables.out
   :language: text
   :caption: output
