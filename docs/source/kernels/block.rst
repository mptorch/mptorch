Block matrix products
=====================

A *block format* (:ref:`block-formats`) stores a tensor as narrow element
codes. The elements are grouped into *blocks* of consecutive elements along
one axis, and each block shares one *scale*. A product of two tensors in
block formats runs on the core routine of :doc:`index`. Only its tile load
is different: it decodes each element from its code and its block's scale.
This page states what that product computes:

- where it quantizes;
- why its blocks must run along the dimension it sums over;
- when it equals the dot product that OCP MX defines;
- where a per-tensor scale enters the sum.

Then come a rotation of the operands before they are quantized, and IEEE
P3109's operations on packed blocks. :doc:`calling` has the functions that run it, and :doc:`/layers` the layer
recipes built on it.

Quantization in the loop
------------------------

For :math:`C = AB` with :math:`A \in \mathbb{R}^{M \times K}` and
:math:`B \in \mathbb{R}^{K \times N}`, a block product rounds in two places.

**The operands, once per product.** Each operand is block-quantized along
:math:`K`, the dimension the product sums over. :math:`A` is split into
blocks of :math:`\beta` consecutive :math:`k` along each row, and :math:`B`
along each column. Each block has one scale. A 2D format instead shares one
scale over a *tile*: :math:`\rho` rows of :math:`A`, or :math:`\rho` columns
of :math:`B`, of :math:`\beta` elements each. The kernel reads each element
decoded:

.. math::

   \tilde a_{ik} = e^A_{ik}\, \sigma^A_{\lfloor i/\rho \rfloor,\, \lfloor k/\beta \rfloor},
   \qquad
   \tilde b_{kj} = e^B_{kj}\, \sigma^B_{\lfloor j/\rho \rfloor,\, \lfloor k/\beta \rfloor}.

Here :math:`e` is the element's value and :math:`\sigma` its block's scale.
The decoded value is the one that :func:`~mptorch.quant.block_unpack`
returns. It is computed in the *carrier*, the floating-point type the kernel
computes in. The carrier is binary64 when either operand was packed in
binary64 or the call names ``carrier=torch.float64``, and binary32
otherwise.

**The sum, at every step.** The loop is the core routine's, with one
difference: there is no multiply format. A block-format unit multiplies the
decoded element values as they are, so the product is the carrier's own.
The sum is rounded to the accumulate format:

.. math::

   s_k = Q_{\text{acc}}\big(s_{k-1} + \operatorname{fl}(\tilde a_{ik}\, \tilde b_{kj})\big)
   \quad \text{(split)}, \qquad
   s_k = Q_{\text{acc}}\big(\tilde a_{ik}\, \tilde b_{kj} + s_{k-1}\big)
   \quad \text{(fused)}.

Here :math:`\operatorname{fl}` is a product in the carrier.
:math:`Q_{\text{acc}}` rounds to a :class:`~mptorch.BinaryK` accumulate format
in the call's rounding mode. Without an accumulate format it is the carrier's
own rounding, so a split step is the carrier's addition and a fused step its
fused multiply-add. Every accumulate algorithm of :doc:`arithmetic` applies,
and :math:`c_{ij} = s_K`.

Why the blocks run along K
--------------------------

When :math:`\beta` consecutive terms of the sum share both operands' scales,
the scales can be factored out of that part of the sum. Take the block
:math:`\kappa` of :math:`K`, the set of its :math:`\beta` consecutive
:math:`k`. Let :math:`\sigma^A_{i\kappa}` and :math:`\sigma^B_{\kappa j}` be
the scales that row :math:`i` of :math:`A` and column :math:`j` of :math:`B`
have over it. Then

.. math::

   \sum_{k \in \kappa} \tilde a_{ik}\, \tilde b_{kj}
   \;=\; \sigma^A_{i\kappa}\, \sigma^B_{\kappa j}
   \sum_{k \in \kappa} e^A_{ik}\, e^B_{kj} .

OCP MX v1.0 defines the dot product of two blocks this way, and a longer dot
product as the sum of its blocks' dot products. A block-format unit
multiplies narrow elements and applies the scales once per block. This
factorization requires the blocks to run along the summed dimension.

MPTorch decodes each element on its own, so it could multiply operands
blocked along either dimension. But the products it simulates are the ones
hardware computes, and a training step sums over a different dimension in
each of its three products:

.. list-table::
   :header-rows: 1
   :widths: 22 18 22 38

   * - pass
     - product
     - sums over
     - packs
   * - forward
     - :math:`x W^\top`
     - input features
     - :math:`x` along its features, :math:`W` along its input features
   * - input gradient
     - :math:`G W`
     - output features
     - :math:`G` along its features, :math:`W` along its output features
   * - weight gradient
     - :math:`G^\top x`
     - the batch
     - :math:`G` and :math:`x` along the batch

Each tensor is therefore quantized twice, along two axes:

- the weight, for the forward pass and for the input gradient;
- the input, for the forward pass and for the weight gradient;
- the output gradient, for the input gradient and for the weight gradient.

With 1D blocks the two quantizations of a tensor differ, so the input
gradient multiplies a weight that the forward pass never saw. A square tile
(:math:`\rho = \beta`) holds the same elements whichever axis is packed, so
one packing of the weight serves both the forward pass and the input
gradient (:ref:`block-formats`, "2D tiles"). :doc:`/layers` has the recipe.

Exact products, and OCP's order of summation
--------------------------------------------

Under an E8M0 scale a decoded value is an element times a power of two,
:math:`\tilde a = e \cdot 2^X`, and the element has at most 8 significant
bits. A product of two decoded values is then a product of two significands
of at most 16 bits, times a power of two. It is **exact in binary32** as long
as it stays in binary32's normal range. In that range, rounding also commutes
with a power-of-two scale: :math:`Q(2^X y) = 2^X Q(y)`. Two things follow for
the MX formats:

- The split step rounds only the sum, because :math:`\operatorname{fl}(\tilde
  a \tilde b) = \tilde a \tilde b`. A fused step is therefore the same step.
- The block product is exactly OCP's two-level dot product, provided its
  terms are summed in OCP's order. That order is the ``BLOCK`` accumulate
  algorithm with ``block_size`` :math:`= \beta` and no ``outer`` format: the
  products of each block are summed on their own, and each block's sum is
  then added to the total,

  .. math::

     c_{ij} = \sum_\kappa \operatorname{fl}\Big( \sigma^A_{i\kappa} \sigma^B_{\kappa j}
     \, D_{ij\kappa} \Big),
     \qquad D_{ij\kappa} = \sum_{k \in \kappa} e^A_{ik}\, e^B_{kj}
     \ \text{(rounded as the accumulator rounds)},

  with the outer sum in binary32. OCP leaves the precision of a block's
  internal sum to the implementation, and the accumulate format is where you
  choose it. The default, ``NAIVE``, adds every product to one running total
  across the blocks. That is a different order, and in general it gives a
  different result.

The run after the next section checks both points on MXFP8 operands whose
magnitudes change along :math:`K`. It rebuilds OCP's sum from the codes.

Tensor scales: in the loop or after it
--------------------------------------

A *cast* scale is a scale format with a mantissa, such as NVFP4's E4M3 or a
superfp. It comes with one more scale for the whole tensor, :math:`S_t`: a
float32 value in binary32, a float64 value in binary64. The block product
can apply it in one of two places.

**In the loop** (the default). The decode includes the tensor scale. A
block's scale is the scale format's value times the tensor scale,
:math:`\sigma^A = \operatorname{fl}(s^A S^A_t)` for :math:`A` and likewise
for :math:`B`, so

.. math::

   \tilde a_{ik} = \operatorname{fl}\big(e^A_{ik} \cdot \operatorname{fl}(s^A S^A_t)\big).

This value can use the carrier's full precision (24 significant bits in
binary32). The decode and every product can round, and the accumulator sees
the operands' own magnitudes.

**After it**, with ``tensor_scale_epilogue=True``, as NVIDIA's NVFP4 GEMMs
do. The loop runs on the elements times their block scales alone. The
result is then multiplied once by the product of the two tensor scales, and
both multiplications are rounded to the carrier:

.. math::

   c_{ij} = \operatorname{fl}\big(\alpha\, s_K\big), \qquad
   \alpha = \operatorname{fl}\big(S^A_t S^B_t\big), \qquad
   s_k = Q_{\text{acc}}\big(s_{k-1} + \operatorname{fl}(e^A s^A \, e^B s^B)\big).

Here :math:`s_k` is the running sum, and :math:`s^A` and :math:`s^B` are the
scale formats' values for the two elements' blocks. With E2M1 elements and
E4M3 scales, an element times its block's scale, :math:`e^A s^A`, has at most
six significant bits, and a product of two at most twelve. The loop
therefore rounds only the sum.

In exact arithmetic the two options give the same product. They also give
the same bits when the tensor scales are powers of two and the sum is left in
binary32 (no accumulate format). Every value then differs by a power of two,
which commutes with each rounding in binary32's normal range. Otherwise they
differ in three ways:

- **Accuracy.** With a binary32 sum the epilogue rounds less. In the run its
  result is two to three times closer to the exact product.
- **Range.** In the epilogue the accumulator sums values that the tensor
  scales have not yet shrunk. Under NVFP4's default tensor scale,
  :math:`S_t = \max|x| / (6 \cdot 448)`, a tensor's largest element decodes
  to :math:`6 \cdot 448 = 2688` whatever the tensor's own magnitude, and a
  product of two such elements to about :math:`7 \cdot 10^6`. A narrow
  accumulate format can therefore overflow where the in-loop product does
  not, as the run's last lines show. NVIDIA's GEMMs accumulate in FP32.
- **The result's values.** The epilogue's result is :math:`\alpha s_K`
  rounded to the carrier. It is not a value of the accumulate format.

A format with a power-of-two scale, such as E8M0 in the MX formats, has no
tensor scale, so the option changes nothing for it.

.. literalinclude:: ../../snippets/kernels_block.py
   :language: python
   :caption: docs/snippets/kernels_block.py

.. literalinclude:: ../../snippets/kernels_block.out
   :language: text
   :caption: output

Rotating before block quantization
----------------------------------

A block's scale is set by its largest element. In a block that holds one
outlier, every other element is rounded on a grid fitted to the outlier, and
most of them lose their precision. A rotation spreads the outlier over its
block before the scale is chosen, which is the idea behind QuaRot and
SpinQuant. ``GemmFormats(rotation=Hadamard(size))`` rotates both operands of
a layer's forward product along :math:`K` before they are quantized:

.. math::

   y = Q(xR)\,Q(WR)^\top, \qquad R = D\,\operatorname{diag}(H_r, \dots, H_r),

with :math:`H_r` the orthonormal Walsh-Hadamard matrix of order :math:`r`
and :math:`D` a diagonal of :math:`\pm 1` (``signed=True``, the default).
Since :math:`RR^\top = I`, the product is :math:`xW^\top` in exact
arithmetic, so a pretrained layer keeps its function. For a matmul
:math:`ab`, the operands are :math:`aR` and :math:`R^\top b`.

- **The gradients.** The layers pass a gradient straight through their
  operand quantizers, so the gradient passes compute in the rotated basis,
  and each result is rotated back along :math:`K`:
  :math:`\partial x = (\partial y\, Q(WR))\,R^\top` and
  :math:`\partial W = (\partial y^\top Q(xR))\,R^\top`. These are the exact
  gradients of the rotated product, with its quantizers taken as the
  identity. The gradient passes pack their operands along :math:`N` and
  along the batch, which the rotation does not mix.
- **The block size.** :math:`r` is ``size``, or, for a :math:`K` that
  ``size`` does not divide, the largest power of two dividing both. A
  784-feature layer under ``Hadamard(128)`` rotates in blocks of 16, and an
  odd :math:`K` is not rotated.
- **The arithmetic.** The transform is a butterfly of additions in the
  carrier, :math:`\log_2 r` stages of :math:`(a + b, a - b)`, then one
  multiply by :math:`1/\sqrt r`. It is not a matmul by :math:`H_r`, which
  CUDA runs in TF32 under ``torch.set_float32_matmul_precision("high")``.
  The signs are an integer hash of the seed and the channel's index,
  computed on the operand's device, so no random generator is drawn from.
  Every device therefore gives the same rotated words.
  :func:`~mptorch.quant.hadamard_rotate` is the transform on its own.
- **Elsewhere in the library.**

  - ``presets.mx(rotation=Hadamard(128))`` rotates every linear product.
  - A frozen layer (:meth:`~mptorch.quant.QLinear.freeze_packed_`) stores
    the rotated weight's codes, and its checkpoint records the rotation.
  - :func:`~mptorch.quant.profile` lists each rotation as an element
    record.
  - A convolution takes no rotation, since its :math:`K` is gathered from
    its input channels and kernel taps.

On GPT-2 small (Hugging Face's pretrained model), every projection but the
tied ``lm_head`` was run through ``simulate`` in each format. The table
gives the perplexity over the first 32,768 tokens of WikiText-2's test split,
against 26.76 in float32 (``dev/experiments/hadamard_rotation.py
perplexity --library``, RTX 4060 laptop GPU, 2026-11-25). The 1× columns
rotate in blocks of the format's block size (32 for MX, 16 for NVFP4), and
the 4× columns in blocks four times that size. The signed rotation is shown
for the seeds 0, 1 and 2 at 1×, and for seed 0 at 4×.

==========  ========  ======  ======================  ======  ======
format      none      H, 1×   DH, 1×                  H, 4×   DH, 4×
==========  ========  ======  ======================  ======  ======
MXFP8 E4M3  27.83     27.29   26.94, 27.24, 26.94     26.99   27.01
MXFP4 E2M1  123.23    49.84   41.07, 41.67, 41.82     41.44   39.19
NVFP4       33.15     37.68   36.05, 35.89, 34.95     33.66   35.24
==========  ========  ======  ======================  ======  ======

**MXFP4 and MXFP8.** The rotation removes most of the loss: 87% of
MXFP4's increase over float32 (DH, 4×) and 55% to 83% of MXFP8's (DH, 1×,
by seed). The seed of the signs moves MXFP8's perplexity by up to 0.3.

**NVFP4.** The rotation makes it worse at every size. A rotation makes each
block's values close to Gaussian, and so brings a format's error to its error
on Gaussian data:

- On GPT-2's activations, MXFP4's power-of-two scale fits its blocks worse
  than it fits Gaussian ones. A rotation lowers its error to the Gaussian
  level.
- NVFP4's E4M3 scale over 16 elements fits a block's largest element
  closely, and on GPT-2's activations its error is already below the
  Gaussian level. A rotation raises it to that level.

**The cost.** A rotated MXFP8 Linear step, forward and backward, at GPT-2's
shapes takes 1.09 to 1.22 times the unrotated step's time on CUDA, and 1.20
to 1.22 times on the CPU (8 threads). That is the four rotations' elementwise
passes.

The run below compares the three formats on operands with three outlier
channels, then holds a rotated layer without quantization to
``nn.Linear``.

.. literalinclude:: ../../snippets/kernels_block_rotation.py
   :language: python
   :caption: docs/snippets/kernels_block_rotation.py

.. literalinclude:: ../../snippets/kernels_block_rotation.out
   :language: text
   :caption: output

IEEE P3109's block operations
-----------------------------

Section VIII of the P3109 paper (`arXiv:2606.04028
<https://arxiv.org/abs/2606.04028>`__) defines operations on a block given
as a scale code and its elements' codes. On a
:class:`~mptorch.quant.BlockPacked`, MPTorch has each of them, and defines
the two that the paper only names. Every one decodes its operands into the
carrier, computes there, and stores the result once.

**BlockProject** is :func:`~mptorch.quant.block_project`: a tensor stored
under scale codes the caller gives, laid out as
:attr:`~mptorch.quant.BlockPacked.scales` is, instead of the codes the
format's rule chooses. With :math:`S` the value a block's code decodes to
(times the tensor scale, for a cast scale), the first case that applies
gives each element :math:`X_i` the value

.. math::

   Z_i = \begin{cases}
     \text{NaN} & \text{if } S \text{ or } X_i \text{ is NaN,} \\
     0 & \text{if } S = 0 \text{, an infinite } X_i \text{ included,} \\
     X_i / S & \text{otherwise,}
   \end{cases}

and :math:`Z_i` is rounded once, as :func:`~mptorch.quant.block_pack` rounds
an element: in the call's rounding mode, saturating at the element format's
largest value. P3109's fourth case, an infinite :math:`S`, cannot arise,
since none of the scale formats here has an infinity. Projecting onto the
scales :func:`~mptorch.quant.block_pack` chose gives its codes word for word,
stochastic rounding included under the same generator state. The one
exception is a block that :func:`~mptorch.quant.block_pack` marked with the
scale format's NaN, because it held a NaN its element format has no code for.
There every element becomes code 0, where :func:`~mptorch.quant.block_pack`
keeps the other codes, and both decode to NaN.

A NaN element under a scale that is not NaN needs a NaN code, which E2M1,
E2M3 and E3M2 do not have (MXFP4, MXFP6, NVFP4). For those,
:func:`~mptorch.quant.block_project` raises. Finding out waits for the kernel
on a GPU, so a call in such a format synchronizes with the device.

**BlockMap** is :func:`~mptorch.quant.block_map`: a function of the decoded
values of one or more packed tensors, its result stored under the scale codes
given, or with ``scales=None`` under the ones the format's rule chooses
(:func:`~mptorch.quant.block_pack`), which is the variant P3109 lets choose its
scales. **BlockExp** is :func:`~mptorch.quant.block_exp`, the exponential, and
**ScaledAdd** is :func:`~mptorch.quant.block_scaled_add`, a sum. P3109 defines
ScaledAdd on single-element blocks; here the operands are any packed tensors
of one shape. A result is rounded twice: by the function's arithmetic in the
carrier, then to the format.

**ConvertToBlockMaxAbsFinite** is
:func:`~mptorch.quant.convert_to_block_max_abs_finite`: each block's scale is
its largest finite magnitude rounded to nearest in the scale format, so that
the block's largest element is stored as about :math:`\pm 1`. OCP's rule puts
it near the element format's largest value instead. A power-of-two scale is
the power of two nearest the magnitude, a tie going to the even exponent. A
cast scale is the magnitude over the tensor scale, which defaults to
:math:`\max|x| / \text{scale\_max}`, rounded by the scale format's cast.

**BlockReduceAdd and BlockDotProduct** are mptorch's definitions:
:func:`~mptorch.quant.block_reduce_add` sums each block's decoded values in
order, by :func:`~mptorch.quant.qsum`, and
:func:`~mptorch.quant.block_dot` multiplies each pair of blocks as a
:math:`[1, \beta] \cdot [\beta, 1]` product, by :func:`~mptorch.quant.qbmm`.
Each takes a mac, so every accumulate algorithm and rounding mode of
:doc:`arithmetic` applies, and returns one value per block in the carrier's
dtype. A block here is :math:`\beta` consecutive elements along the packed
axis, one row of a 2D tile.

:func:`~mptorch.quant.block_project` is an operation of its own on every
backend. It costs about what :func:`~mptorch.quant.block_pack` does, since it
skips the pass that finds each block's largest magnitude: 0.81 to 0.94 times
as long on a Ryzen CPU at one and sixteen threads, and 0.99 to 1.35 times on
an RTX 4060 laptop GPU, over :math:`4096^2` float32 elements in MXFP8, MXFP4
and NVFP4. The other operations are compositions of existing operations.

.. literalinclude:: ../../snippets/kernels_block_p3109.py
   :language: python
   :caption: docs/snippets/kernels_block_p3109.py

.. literalinclude:: ../../snippets/kernels_block_p3109.out
   :language: text
   :caption: output

How the kernel decodes
----------------------

Before its loop, each CUDA thread block builds two tables per operand from
the operand's format. A CPU task and an Apple GPU threadgroup do the same.
One table holds the value of each possible element code, and the other the
factor of each possible scale code, 256 entries each. The scale table includes the
tensor scale, unless the tensor scale is applied after the loop. A tile load
then does four things:

- it reads the element's code and its block's scale code from the packed
  arrays;
- it looks up both values in the tables;
- it multiplies them, in the carrier;
- it turns a zero result into :math:`+0`, so that no decoded value is
  :math:`-0`.

Because of the tables, the block product takes element codes of at most 8
bits. That covers every OCP and NVFP4 format, every superfp element up to a
byte, and block floating point from ``bfp(2)`` to ``bfp(8)``. The tables
decode a block floating point element's integer code as the binaryK layout
it is (:doc:`/concepts`, "Block floating point"). A wider element raises: its
operand can be unpacked and multiplied with a float product instead. The
elementwise block operations take codes of up to 16 bits. Compiling in an
arithmetic decode as a fallback for wider codes took the CUDA kernels from
30-38 registers to 48-80, so the kernels have none.

An operand can also be packed along its other trailing dimension, such as a
square-tiled weight read for the input gradient. The kernel then reads it
transposed, which costs one select in the load. Under stochastic rounding
each output element draws from its own random stream, as in every product
of the core routine.

How much the decode costs depends on the arithmetic each step does beside
it. The decode shows most where a step is cheapest, and little once a step
rounds more. At :math:`1024^3` on an RTX 4060 laptop GPU, the block product
was compared with the flat product on the decoded float32 operands
(``dev/benchmarks/timing/benchmark_block.py``):

- With a fused mac under round-to-nearest, it takes 1.29 to 1.30 times as
  long. This is the cheapest arithmetic, and so the one where the decode
  shows most.
- With a fused mac under stochastic rounding, it takes 1.07 to 1.09 times as
  long.
- With a split mac, it takes 0.40 to 0.41 times as long under
  round-to-nearest, and 0.54 to 0.55 times under stochastic rounding. The
  flat product also rounds every product to a multiply format, and the block
  product does not.

The kernels compute in binary32 or binary64 on the CPU and CUDA, and return
a float32 or a float64 result accordingly. An Apple GPU has no float64, so
it computes in binary32 only.
