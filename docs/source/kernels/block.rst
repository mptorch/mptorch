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

:doc:`calling` has the functions that run it, and :doc:`/layers` the layer
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
