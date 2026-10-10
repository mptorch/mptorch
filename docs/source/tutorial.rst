Tutorial: training in 8-bit floating point
==========================================

Since the Hopper and Ada generations, NVIDIA GPUs multiply matrices in two
8-bit floating-point formats: **E4M3** (four exponent bits, three mantissa
bits) and **E5M2** (five and two). These are also the formats of the Open
Compute Project's FP8 specification, and AMD and Intel accelerators implement
the same pair. The standard recipe for training in them is E4M3 for the
forward pass, E5M2 for the gradients, 32-bit accumulation, and a scale per
tensor. It was worked out on hardware that already had the tensor cores.
MPTorch lets you take that recipe apart on a GPU without them (or on a CPU),
change any one ingredient, and see what it does to a training run.

This tutorial builds up from the formats to a trained network. Every code
block is a file under ``docs/snippets/tutorial/``, and every output is
recorded from a run. The training script of the last section took about
three minutes on a laptop GPU.

1. The formats
--------------

An FP8 number is a sign, an exponent and a mantissa, like any binary float
(:doc:`concepts` has the general form). With the OCP biases:

.. math::

   \text{E4M3:}\quad x = (-1)^s \Big(1 + \frac{m}{8}\Big) 2^{\,e-7}, \qquad
   \text{E5M2:}\quad x = (-1)^s \Big(1 + \frac{m}{4}\Big) 2^{\,e-15}.

E4M3 spends its bits on precision: eight values per binade, a largest finite
value of 448 and a smallest subnormal of :math:`2^{-9}`. E5M2 spends them on
range: four values per binade, up to 57344 and down to :math:`2^{-16}`, the
same exponent range as float16.

In MPTorch the nearest spellings are ``BinaryK(8, 4, bias=7)`` and
``BinaryK(8, 3, bias=15)``. The bias has to be given, because a ``BinaryK``
otherwise takes IEEE P3109's, which is one larger: ``BinaryK(8, 4)`` is
P3109's ``Binary8p4``, not OCP's E4M3. Only the first spelling is exact:

- E4M3 already spends its top exponent field on numbers, as P3109 does, so
  ``BinaryK(8, 4, bias=7)`` has exactly its finite values.
- E5M2 is laid out the IEEE 754 way, with that field kept for infinities
  and NaNs. No P3109 format keeps it, so no BinaryK format *is* E5M2, for
  the same reason that none is float16, bfloat16 or float32
  (:doc:`concepts`).

PyTorch's own ``torch.float8_e4m3fn`` and ``torch.float8_e5m2`` dtypes serve
as the reference. The run tabulates the ranges from both sides. It then
rounds *every one of the* :math:`2^{32}` *float32 values* with both and
counts where they disagree:

.. literalinclude:: ../snippets/tutorial/fp8_01_formats.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_01_formats.py

.. literalinclude:: ../snippets/tutorial/fp8_01_formats.out
   :language: text
   :caption: output

Up to each dtype's largest finite value, the two agree in value on every
input. The count compares with ``!=``, which cannot see the sign of a zero.
PyTorch keeps that sign. A BinaryK format has one unsigned zero, so it
returns :math:`+0.0` for :math:`-0.0` and for a negative input that rounds
to zero. Beyond the largest finite value the two differ, in a different way
for each format:

- For E4M3 only the overflow convention differs. PyTorch saturates to
  :math:`\pm 448`, infinite inputs included. A BinaryK format under its
  default ``SAT_NONE`` policy produces :math:`\pm\infty`. Under
  ``SAT_FINITE`` it clamps to 480, because the finite domain spends the top
  code on a number. ``SAT_FINITE_EXTENDED``, which is P3109's ``SatFinite``
  in the extended domain, is PyTorch's convention: it clamps to 448,
  infinities included.
- For E5M2 the value set differs. The first table's largest finite values,
  98304 against 57344, show it. ``BinaryK(8, 3, bias=15)`` spends the top
  exponent field, which E5M2 reserves, on three more numbers: 65536, 81920
  and 98304. So 61440 rounds to 65536 where PyTorch overflows. It is the
  nearest BinaryK to E5M2, not E5M2.

A model that keeps its values in range sees neither difference, and the sign
of a zero changes the value of no sum or product.

2. What rounding to FP8 costs
-----------------------------

Rounding a tensor to a format with a few mantissa bits replaces each value
by one within half a spacing of it. The spacing in :math:`[2^q, 2^{q+1})` is
:math:`2^{q-3}` for E4M3 and :math:`2^{q-2}` for E5M2. The relative error
per element is therefore bounded by :math:`2^{-4}` and :math:`2^{-3}`, and
its root mean square over a Gaussian tensor is near 2.7 % and 5.3 %. The
rounding *mode* decides the sign of the error: round-to-nearest is unbiased,
truncation is biased toward zero, rounding up is biased upward, and
stochastic rounding is unbiased with more variance.

.. literalinclude:: ../snippets/tutorial/fp8_02_rounding.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_02_rounding.py

.. literalinclude:: ../snippets/tutorial/fp8_02_rounding.out
   :language: text
   :caption: output

The second table shows why there are two formats. It covers nine decades of
magnitude:

- E4M3 keeps its 2.5 % accuracy only between about :math:`10^{-2}` and
  :math:`10^{2}`. Below that its subnormals run out of digits and then flush
  to zero (a relative error of 1.0 means the value *became* zero). Above 448
  it overflows.
- E5M2 is twice as coarse everywhere but keeps its 5 % accuracy from
  :math:`10^{-4}` to :math:`10^{4}`.

Weights and activations stay within a couple of decades of one. Gradients
shrink layer by layer and step by step, and do not. That is the argument for
E4M3 forward and E5M2 backward.

3. One matrix product, several arithmetics
------------------------------------------

An FP8 tensor core takes two 8-bit operands and multiplies them *exactly*:
the product of two 4-bit significands has 8 bits, which any format wider
than E4M3 holds. It adds the product to a running sum kept in a wider
format. On Hopper that accumulator is nominally float32, though it has been
measured to keep only about 14 mantissa bits for FP8 inputs. On other
hardware it may be float16 or bfloat16.

In MPTorch's vocabulary (:doc:`kernels/arithmetic`) this is a ``SplitMac``.
Its ``mul`` is any format wide enough to hold the product exactly, and its
``acc`` is the accumulator being studied:

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

The 16-bit formats are named after the dtypes whose field widths they have:
``fp16`` is P3109's ``Binary16p11`` and ``bf16`` its ``Binary16p8``. They
are not those dtypes. With P3109's bias, each stops one code short of the
dtype's largest value, which is its :math:`+\infty`, and reaches a binade
below the dtype's smallest value (:doc:`concepts`, "The IEEE 754 dtypes are
not P3109 formats"). Neither end matters in this product. The E4M3 operands
are multiples of :math:`2^{-9}`, so every product and sum is a multiple of
:math:`2^{-18}`, and the sums stay many binades below 65472. The rows
therefore measure the widths and nothing else.

Reading down the table:

- Rounding the operands to E4M3 costs 3.9 % relative error, and *nothing the
  accumulator does above float16's width changes that*. The float32 and
  14-bit sums give the same number, and the float16-width one is very
  slightly worse.
- A bfloat16-width accumulator (7 mantissa bits, for a sum of 1024 terms)
  adds visibly to the error.
- Stochastic rounding of that sum adds variance and removes no bias,
  because the terms have random signs.
- Rounding the *products* to E4M3 as well costs a further point.
- Accumulating in E4M3 no longer gives a usable matrix product.

This is the experiment to run when a new accelerator's accumulator width is
in question: change ``acc`` and read the row.

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

Everything the forward pass reads is in E4M3, every gradient is in E5M2, and
the products are exact with a wide sum. As a ``QAffineFormats`` this is one
factory call for the arithmetic and six quantizer assignments:

.. literalinclude:: ../snippets/tutorial/fp8_04_linear.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_04_linear.py

.. literalinclude:: ../snippets/tutorial/fp8_04_linear.out
   :language: text
   :caption: output

The output has three blocks:

- The first confirms that the layer is close to its float32 twin in all
  three directions.
- The second makes the range argument of section 2 concrete on a gradient of
  realistic size: two thirds of it vanishes in E4M3, and almost none in
  E5M2.
- The third shows the last ingredient of the recipe, **scaling**. A tensor
  is multiplied into the format's range before rounding and divided back
  afterwards. Its largest element then uses the top of the range, and its
  smallest elements are not lost at the bottom. Hardware FP8 keeps one such
  scale per tensor (Transformer Engine's "amax" scaling). Here it is eight
  lines of Python, because a quantizer slot takes any callable. With it,
  even E4M3 holds the gradient.

5. Training the network
-----------------------

The last script trains a 784-128-96-10 multilayer perceptron on MNIST. This
is the example of the original MPTorch tutorial, on the current interface.
It trains nine configurations from the same initial weights and the same
shuffles, with plain SGD, for five epochs each. Three things vary:

- The **signals**: float32, or E4M3 on everything the forward pass reads and
  E5M2 on every gradient. The logits are rounded by a ``Quantizer`` (E4M3
  forward, E5M2 backward), because there is no ``output_quant`` slot.
- The **matmuls**: float32 (no math hooks), or a ``binaryK_gemm_formats``
  core with exact products (``mul_K=16, mul_P=8``, wide enough to hold any
  FP8 product) and a running sum in one of four accumulators: 14 bits;
  bfloat16's widths (``Binary16p8``, which is not bfloat16 itself, as
  section 3 says); the same with stochastic rounding; or E4M3.
- The **master weights**: float32, or rounded to E4M3 after every optimizer
  step, with round-to-nearest or stochastic rounding. The update
  :math:`w \leftarrow Q(w - \eta\, g)`, with learning rate :math:`\eta` and
  gradient :math:`g`, is applied by hand after ``opt.step()``. A
  low-precision master copy is nothing more than that.

One configuration also scales the loss by 1024 (and the gradients back
down), the classic remedy for gradient underflow.

.. literalinclude:: ../snippets/tutorial/fp8_05_mlp_mnist.py
   :language: python
   :caption: docs/snippets/tutorial/fp8_05_mlp_mnist.py

.. literalinclude:: ../snippets/tutorial/fp8_05_mlp_mnist.out
   :language: text
   :caption: output

From epoch to epoch, the test accuracy of any one run moves by a few tenths
of a percent, so differences of that size between rows are noise. The table
shows three things:

- **FP8 signals cost almost nothing here.** Rounding every activation,
  weight and gradient of this network to eight bits, with float32 products
  and sums, trains to the same accuracy as float32. Loss scaling changes
  nothing on this problem. An MLP three layers deep has no gradient small
  enough to underflow E5M2, as section 4 already suggested.
- **The accumulator has a threshold, and it is low.** A 14-bit accumulator
  is indistinguishable from float32. So is a bfloat16-width one, for this
  network, whose longest dot product has 784 terms, even though section 3
  measured a visibly larger error for it on a single product. Stochastic
  rounding of the sums neither helps nor hurts. An E4M3 accumulator is
  different: it trains to about 92-95 % for four epochs and then collapses
  to chance. The run looks as if it is working until it stops working.
- **Master weights are where stochastic rounding matters.** The update
  :math:`\eta g` is small compared with the weight it is added to. With the
  weights held in E4M3 and rounded to nearest after every step, any update
  below half a spacing (:math:`2^{-4}` of the weight's magnitude) is lost
  entirely, and the network trains three to four points worse. The same
  master copy rounded *stochastically* keeps those updates in expectation
  and recovers most of the gap. This is the accumulator effect of
  :doc:`concepts`, in a network. It is why every FP8 training recipe keeps
  its master weights in a wider format or rounds them stochastically.

Where to go from here
---------------------

- **Attention.** The two matrix products of an attention head are
  :class:`~mptorch.quant.QMatmul` modules over a ``QMatmulFormats``
  (:doc:`kernels/calling` shows one). This puts the same recipe inside a
  transformer block, including the ``q @ k.mT`` shape with no copy.
- **Convolutions.** ``QConv2d`` takes the same formats, and
  :func:`~mptorch.quant.conv_formats` runs its three passes in the recipe's
  arithmetic (:doc:`layers`, "Convolutions").
- **Block formats.** The next generation of the recipe shares a scale per
  block of 32 or 16 elements: OCP's MXFP8 and MXFP4, and NVIDIA's NVFP4.
  :func:`~mptorch.quant.block_gemm_formats` trains the same network in them
  (:ref:`block-formats`, :doc:`kernels/block`).
- **Per-element formats.** A ``Palette`` lets the first rows of a product
  run in E5M2 and the rest in E4M3, or any assignment that a ``prec_idx``
  map describes.
- **Other formats.** Everything above uses ``BinaryK``. ``SuperFP`` is a
  second family with a different trade-off. Any other width is one
  constructor call away: ``BinaryK(6, 3)``, ``BinaryK(4, 2)``, or a format
  nobody has built.
- **Wider arithmetic.** Everything above runs in float32, whose binary32
  arithmetic carries every format used here. ``model.double()`` moves the
  same model into binary64, where formats of up to 53 bits of precision are
  simulated exactly. That gives a reference for measuring what an 8-bit
  accumulator really costs. ``carrier=torch.float64`` gives the float32
  model binary64's products and sums while it keeps its dtype
  (:doc:`layers`).
- **Other roundings.** Each format in a ``SplitMac`` has its own saturation
  and subnormal policy, and each ``SplitMac`` its own rounding mode. The
  scripts in this tutorial take those as arguments and are meant to be
  edited.
