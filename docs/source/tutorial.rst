Tutorial: training in 8-bit floating point
==========================================

Since the Hopper and Ada generations, NVIDIA GPUs multiply matrices in two
8-bit floating-point formats: **E4M3** (four exponent bits, three mantissa
bits) and **E5M2** (five and two). They are also the formats of the Open
Compute Project's FP8 specification, and AMD and Intel accelerators implement
the same pair. The recipe that makes training in them work -- E4M3 for the
forward pass, E5M2 for the gradients, 32-bit accumulation, a scale per
tensor -- was worked out on hardware that already had the tensor cores.
MPTorch lets you take that recipe apart on a GPU without them (or on a CPU),
change any one ingredient, and watch what it does to a training run.

This tutorial builds up from the formats to a trained network. Every code
block is a file under ``docs/snippets/tutorial/`` and every output is
recorded from a run; the last section's training script took about three
minutes on a laptop GPU.

1. The formats
--------------

An FP8 number is a sign, an exponent and a mantissa, like any binary float
(:doc:`concepts` has the general form). With the OCP biases:

.. math::

   \text{E4M3:}\quad x = (-1)^s \Big(1 + \frac{m}{8}\Big) 2^{\,e-7}, \qquad
   \text{E5M2:}\quad x = (-1)^s \Big(1 + \frac{m}{4}\Big) 2^{\,e-15}.

E4M3 spends its bits on precision: eight values per binade, a largest finite
value of 448 and a smallest subnormal of :math:`2^{-9}`. E5M2 spends them on
range: four values per binade, up to 57344 and down to :math:`2^{-16}` --
the same exponent range as float16. In MPTorch they are
``BinaryK(8, 4, bias=7)`` and ``BinaryK(8, 3, bias=15)`` -- the bias has to
be given, because a ``BinaryK`` otherwise takes IEEE P3109's, which is one
larger (``BinaryK(8, 4)`` is P3109's ``Binary8p4``, not OCP's E4M3) -- and PyTorch's own
``torch.float8_e4m3fn`` and ``torch.float8_e5m2`` dtypes serve as the
reference. The run tabulates the ranges from both sides, then rounds *every
one of the* :math:`2^{32}` *float32 values* with both and counts where they
disagree:

.. literalinclude:: ../snippets/tutorial/fp8_01_formats.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_01_formats.py

.. literalinclude:: ../snippets/tutorial/fp8_01_formats.out
   :language: text
   :caption: output

Up to each format's largest finite value the two agree on every single
input. Beyond it they differ in convention only: E4M3 in PyTorch has no
infinity and turns an overflow into NaN, where a BinaryK format under its
default ``OVF_INF`` policy produces :math:`\pm\infty` (or clamps, under
``SAT_FINITE``); and PyTorch's E5M2 reserves its top exponent for infinities,
where BinaryK, like P3109, keeps it for numbers, so 61440 rounds to 65536 rather than
overflowing. A model that keeps its values in range never sees either.

2. What rounding to FP8 costs
-----------------------------

Rounding a tensor to a format with three mantissa bits replaces each value by
one within half a spacing of it, and the spacing in :math:`[2^q, 2^{q+1})` is
:math:`2^{q-3}` for E4M3 and :math:`2^{q-2}` for E5M2. The relative error
per element is therefore bounded by :math:`2^{-4}` and :math:`2^{-3}`, and
its root-mean-square over a Gaussian tensor comes out near 2.7 % and 5.3 %.
The rounding *mode* decides the sign of the error: round-to-nearest is
unbiased, truncation is biased toward zero, rounding up is biased upward,
and stochastic rounding is unbiased with more variance.

.. literalinclude:: ../snippets/tutorial/fp8_02_rounding.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_02_rounding.py

.. literalinclude:: ../snippets/tutorial/fp8_02_rounding.out
   :language: text
   :caption: output

The second table is the reason there are two formats. Over nine decades of
magnitude, E4M3 keeps its 2.5 % accuracy only between about :math:`10^{-2}`
and :math:`10^{2}`: below that its subnormals run out of digits and then
flush to zero (a relative error of 1.0 means the value *became* zero), and
above 448 it overflows. E5M2 is twice as coarse everywhere but holds its
5 % accuracy from :math:`10^{-4}` to :math:`10^{4}`. Weights and activations
live within a couple of decades of one; gradients, which shrink layer by
layer and step by step, do not -- and that is the whole argument for E4M3
forward, E5M2 backward.

3. One matrix product, several arithmetics
------------------------------------------

An FP8 tensor core takes two 8-bit operands, multiplies them *exactly* --
the product of two 4-bit significands has 8 bits, which anything wider than
E4M3 holds -- and adds the product to a running sum kept in a wider format.
On Hopper that accumulator is nominally float32, though it has been measured
to keep only about 14 mantissa bits for FP8 inputs; on other hardware it may
be float16 or bfloat16. In MPTorch's vocabulary (:doc:`gemm`) that is a
``SplitMac`` whose ``mul`` is any format wide enough to hold the product
exactly and whose ``acc`` is the accumulator under study:

.. math::

   s_k = Q_{\text{acc}}\big(s_{k-1} + a_{ik} b_{kj}\big), \qquad
   a_{ik}, b_{kj} \in \text{E4M3}.

The run quantizes the operands of a :math:`512 \times 1024 \times 512`
product to E4M3 once, then computes the product in eight arithmetics and
measures each against a float64 reference:

.. literalinclude:: ../snippets/tutorial/fp8_03_matmul.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_03_matmul.py

.. literalinclude:: ../snippets/tutorial/fp8_03_matmul.out
   :language: text
   :caption: output

Reading down the table: rounding the operands to E4M3 costs 3.9 % relative
error, and *nothing the accumulator does above float16 changes that* -- the
float32 and 14-bit sums land on the same number and float16 a hair behind.
A bfloat16
accumulator (7 mantissa bits, for a sum of 1024 terms) adds visibly to the
error, stochastic rounding of that sum adds variance rather than removing a
bias (the terms have random signs), rounding the *products* to E4M3 as well
costs a further point, and accumulating in E4M3 is not a matrix product any
more. This is the experiment to run whenever a new accelerator's accumulator
width is in question: change ``acc`` and read the row.

4. One layer, the FP8 recipe
----------------------------

A linear layer has four signals and three matrix products
(:doc:`layers`), and the standard FP8 training recipe assigns them so:

.. math::

   \begin{aligned}
   y &= \big[\, Q_{\text{E4M3}}(x)\; Q_{\text{E4M3}}(W)^{\top} \,\big]_{\text{fp32 acc}} + Q_{\text{E4M3}}(b) \\[4pt]
   \frac{\partial L}{\partial x} &= \big[\, Q_{\text{E5M2}}(G)\; Q_{\text{E4M3}}(W) \,\big]_{\text{fp32 acc}} \\[4pt]
   \frac{\partial L}{\partial W} &= \big[\, Q_{\text{E5M2}}(G)^{\top}\; Q_{\text{E4M3}}(x) \,\big]_{\text{fp32 acc}}
   \end{aligned}

-- everything the forward pass reads in E4M3, every gradient in E5M2, and
products exact with a wide sum. As a ``QAffineFormats`` that is a factory
call for the arithmetic and six quantizer assignments:

.. literalinclude:: ../snippets/tutorial/fp8_04_linear.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_04_linear.py

.. literalinclude:: ../snippets/tutorial/fp8_04_linear.out
   :language: text
   :caption: output

The first block confirms the layer is close to its float32 twin in all three
directions. The second is the range argument of section 2 made concrete on a
gradient of realistic size: two thirds of it vanishes in E4M3, almost none
in E5M2. The third is the last ingredient of the recipe, **scaling**: a
tensor is multiplied into the format's range before rounding and divided
back afterwards, so that its largest element uses the top of the range and
its smallest elements are not lost at the bottom. Hardware FP8 keeps one
such scale per tensor (Transformer Engine's "amax" scaling), and here it is
eight lines of Python, because a quantizer slot takes any callable. With it,
even E4M3 holds the gradient.

5. Training the network
-----------------------

The last script trains a 784-128-96-10 multilayer perceptron on MNIST --
the example of the original MPTorch tutorial, on the current interface --
in nine configurations from the same initial weights and the same shuffles,
plain SGD, five epochs each. Three things vary:

- the **signals**: float32, or E4M3 on everything the forward pass reads and
  E5M2 on every gradient, with the logits rounded by a ``Quantizer`` (E4M3
  forward, E5M2 backward) because there is no ``output_quant`` slot;
- the **matmuls**: float32 (no math hooks), or a ``binaryK_gemm_formats``
  core with exact products (``mul_K=16, mul_P=8``, wide enough to hold any
  FP8 product) and a running sum in a 14-bit accumulator, in bfloat16, in
  bfloat16 with stochastic rounding, or in E4M3;
- the **master weights**: float32, or rounded to E4M3 after every
  optimizer step, with round-to-nearest or stochastic rounding. The update
  :math:`w \leftarrow Q(w - \eta\, g)` is applied by hand after
  ``opt.step()``, which is all a low-precision master copy is.

One configuration also scales the loss by 1024 (and the gradients back
down), the classic remedy for gradient underflow.

.. literalinclude:: ../snippets/tutorial/fp8_05_mlp_mnist.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_05_mlp_mnist.py

.. literalinclude:: ../snippets/tutorial/fp8_05_mlp_mnist.out
   :language: text
   :caption: output

Epoch-to-epoch the test accuracy of any one run moves by a few tenths of a
percent, so differences of that size between rows are noise. What the table
does say:

- **FP8 signals cost almost nothing here.** Rounding every activation,
  weight and gradient of this network to eight bits, with float32 products
  and sums, trains to the same accuracy as float32. Loss scaling changes
  nothing on this problem: an MLP three layers deep has no gradient small
  enough to underflow E5M2, as section 4 already suggested.
- **The accumulator has a threshold, and it is low.** A 14-bit accumulator
  is indistinguishable from float32, and so -- for this network, whose
  longest dot product has 784 terms -- is a bfloat16 one, even though
  section 3 measured it at a visibly larger error on a single product;
  stochastic rounding of the sums neither helps nor hurts. An E4M3
  accumulator is another matter: it trains to about 92-95 % for four
  epochs and then collapses to chance, a run that looks like it is working
  until it is not.
- **Master weights are where stochastic rounding matters.** The update
  :math:`\eta g` is small against the weight it is added to. With the
  weights held in E4M3 and rounded to nearest after every step, any update
  below half a spacing -- :math:`2^{-4}` of the weight's magnitude -- is lost
  entirely, and the network trains three to four points worse. The same
  master copy rounded *stochastically* keeps those updates in expectation
  and recovers most of the gap. This is the accumulator effect of
  :doc:`concepts`, in a network, and it is why every FP8 training recipe
  keeps its master weights in a wider format or rounds them stochastically.

Where to go from here
---------------------

- **Attention.** The two matrix products of an attention head are
  :class:`~mptorch.quant.QMatmul` modules over a ``QMatmulFormats``
  (:doc:`gemm` shows one), which puts the same recipe -- and the
  ``q @ k.mT`` shape, with no copy -- inside a transformer block.
- **Per-element formats.** A ``Palette`` lets the first rows of a product
  run in E5M2 and the rest in E4M3, or any assignment a ``prec_idx`` map
  describes.
- **Other formats.** Everything above is ``BinaryK``; ``SuperFP`` is a second
  family with a different trade-off, and any width you like is a
  constructor call away -- ``BinaryK(6, 3)``, ``BinaryK(4, 2)``, or a
  format nobody has built.
- **Other roundings.** Each format in a ``SplitMac`` has its own saturation
  and subnormal policy; each ``SplitMac`` its own rounding mode. The scripts
  in this tutorial take those as arguments and are meant to be edited.
