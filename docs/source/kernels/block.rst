Block matrix products
=====================

A product of two tensors in block formats (:ref:`block-formats`) runs on the
core routine of :doc:`index` with a tile load that decodes each element from
its code and its block's scale. This page states what that product computes:
where it quantizes, why its blocks must run along the dimension it sums over,
when it equals the dot product OCP MX defines, and where a per-tensor scale
enters the sum. :doc:`calling` has the functions that run it, and
:doc:`/layers` the layer recipes built on it.

Quantization in the loop
------------------------

For :math:`C = AB` with :math:`A \in \mathbb{R}^{M \times K}` and
:math:`B \in \mathbb{R}^{K \times N}`, a block product rounds in two places.

**The operands, once per product.** Each operand is block-quantized along
:math:`K`, the dimension the product sums over: :math:`A` in blocks of
:math:`\beta` consecutive :math:`k` along each row, :math:`B` along each
column, a scale per block (per tile of :math:`\rho` rows or columns, for a 2D
format). The kernel reads each element decoded,

.. math::

   \tilde a_{ik} = e^A_{ik}\, \sigma^A_{\lfloor i/\rho \rfloor,\, \lfloor k/\beta \rfloor},
   \qquad
   \tilde b_{kj} = e^B_{kj}\, \sigma^B_{\lfloor j/\rho \rfloor,\, \lfloor k/\beta \rfloor},

an element value times its block's scale -- the value
:func:`~mptorch.quant.block_unpack` returns, computed in binary32.

**The sum, at every step.** The loop is the routine's, with the carrier's
own product in place of a multiply format -- the elements' values are what a
block-format unit multiplies -- and the sum rounded to the accumulate format:

.. math::

   s_k = Q_{\text{acc}}\big(s_{k-1} + \operatorname{fl}(\tilde a_{ik}\, \tilde b_{kj})\big)
   \quad \text{(split)}, \qquad
   s_k = Q_{\text{acc}}\big(\tilde a_{ik}\, \tilde b_{kj} + s_{k-1}\big)
   \quad \text{(fused)},

where :math:`\operatorname{fl}` is a binary32 product and
:math:`Q_{\text{acc}}` rounds to a :class:`~mptorch.BinaryK` accumulate format
in the call's rounding mode, or is binary32's own addition without one. Every
accumulate algorithm of :doc:`arithmetic` applies, and :math:`c_{ij} = s_K`.

Why the blocks run along K
--------------------------

Where :math:`\beta` consecutive terms of the sum share both operands'
scales, the scales come out of their part of the sum. For the block
:math:`\kappa` of :math:`K`,

.. math::

   \sum_{k \in \kappa} \tilde a_{ik}\, \tilde b_{kj}
   \;=\; \sigma^A_{i\kappa}\, \sigma^B_{\kappa j}
   \sum_{k \in \kappa} e^A_{ik}\, e^B_{kj} .

OCP MX v1.0 defines the dot product of two blocks this way, and a longer dot
product as the sum of its blocks' dot products: a block-format unit multiplies
narrow elements and applies the scales once per block. The factorization
needs the blocks to run along the summed dimension. MPTorch's decode is
random access, so it would multiply operands blocked either way, but the
products it simulates are the ones hardware computes, and a training step
sums over a different dimension in each of its three products:

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

So each tensor is quantized twice, along two axes: the weight for the forward
and the input gradient, the input for the forward and the weight gradient,
the output gradient for the two backward passes. With 1D blocks the two
quantizations of a tensor differ -- the input gradient multiplies a weight
the forward never saw. A square tile holds the same elements whichever axis
is packed, so one packing serves both passes (:ref:`block-formats`, "2D
tiles"); :doc:`/layers` has the recipe.

Exact products, and OCP's order of summation
--------------------------------------------

Under an E8M0 scale a decoded value is an element times a power of two,
:math:`\tilde a = e \cdot 2^X`, with at most 8 significant bits in the
element. A product of two is then a product of significands of at most 16
bits times a power of two, **exact in binary32** as long as it stays in
binary32's normal range, and rounding commutes with a power-of-two scale in
that range, :math:`Q(2^X y) = 2^X Q(y)`. Two things follow for the MX
formats:

- the split step rounds nothing but the sum: :math:`\operatorname{fl}(\tilde
  a \tilde b) = \tilde a \tilde b`, and a fused step is the same step;
- the block product is OCP's two-level dot product exactly, once its terms are
  summed in OCP's order. That order is the ``BLOCK`` accumulate algorithm with
  ``block_size`` :math:`= \beta` and no ``outer`` format: a block's products
  summed on their own, then added to the total,

  .. math::

     c_{ij} = \sum_\kappa \operatorname{fl}\Big( \sigma^A_{i\kappa} \sigma^B_{\kappa j}
     \, S_{ij\kappa} \Big),
     \qquad S_{ij\kappa} = \sum_{k \in \kappa} e^A_{ik}\, e^B_{kj}
     \ \text{(rounded as the accumulator rounds)},

  with the outer sum in binary32. OCP leaves the precision of a block's
  internal sum to the implementation; the accumulate format is where you
  choose it. The default, ``NAIVE``, adds every product to one running total
  across the blocks, which is another order and in general another result.

The run below checks both on MXFP8 operands whose magnitudes change along
:math:`K`, rebuilding OCP's sum from the codes, and then turns to the next
section's question.

Tensor scales: in the loop or after it
--------------------------------------

A cast scale (NVFP4's E4M3, or a superfp) comes with a float32 scale per
tensor, :math:`S_t`, and the block product can apply it in either of two
places.

**In the loop**, the default: the decode carries it,
:math:`\sigma = \operatorname{fl}(s\, S_t)`, so

.. math::

   \tilde a_{ik} = \operatorname{fl}\big(e^A_{ik} \cdot \operatorname{fl}(s^A S^A_t)\big),

a value of up to 24 significant bits. The decode and every product can round,
and the accumulator sees the operands' own magnitudes.

**After it**, with ``tensor_scale_epilogue=True``, as NVIDIA's NVFP4 GEMMs do:
the loop runs on the elements times their block scales alone, and the result
is multiplied once by the product of the two tensor scales,

.. math::

   c_{ij} = \operatorname{fl}\big(\alpha\, s_K\big), \qquad
   \alpha = \operatorname{fl}\big(S^A_t S^B_t\big), \qquad
   s_k = Q_{\text{acc}}\big(s_{k-1} + \operatorname{fl}(e^A s^A \, e^B s^B)\big).

With E2M1 elements and E4M3 scales, :math:`e\,s` has at most six significant
bits and a product of two at most twelve, so the loop rounds only the sum. The
two options are the same product in exact arithmetic, and with power-of-two
tensor scales and a binary32 sum the same bits, since every value then
differs by a power of two, which commutes with each rounding in binary32's
normal range. Otherwise they differ in three ways:

- **Accuracy.** With a binary32 sum, the epilogue rounds less: in the run it
  is two to three times closer to the exact product.
- **Range.** In the epilogue the accumulator sums values the tensor scales
  have not yet shrunk. Under NVFP4's default :math:`S_t = \max|x| / (6 \cdot
  448)` a tensor's largest element decodes to :math:`6 \cdot 448 = 2688`
  whatever the tensor's own magnitude, and a product of two to about
  :math:`7 \cdot 10^6`, so a narrow accumulate format can overflow where the
  in-loop product does not, as the run's last lines show. NVIDIA's GEMMs
  accumulate in FP32.
- **The result's values.** The epilogue's result is :math:`\alpha s_K` rounded
  to binary32, not a value of the accumulate format.

An E8M0 format has no tensor scale, and the option changes nothing there.

.. literalinclude:: ../../snippets/kernels_block.py
   :language: python
   :caption: docs/snippets/kernels_block.py

.. literalinclude:: ../../snippets/kernels_block.out
   :language: text
   :caption: output

How the kernel decodes
----------------------

Each thread block fills two tables per operand when it starts, from the
operand's format: the values of its 256 element codes and of its 256 scale
codes, the latter with the tensor scale folded in unless it is applied after
the loop. A tile load is then two table reads, one binary32 product and a
test that keeps a zero unsigned. The tables are why the block product takes
element codes of at most 8 bits -- every OCP and NVFP4 format, and every
superfp element up to a byte -- where the elementwise block operations take up
to 16: an arithmetic decode compiled in for wider codes took the kernels from
30-38 registers to 48-80. An operand packed along its other dimension is read
transposed, at the cost of a select in the load, and under stochastic
rounding each output element draws as in every product of the routine.

The decode costs little next to the arithmetic. At :math:`1024^3` on an RTX
4060 laptop GPU the block product takes 1.15 to 1.18 times the time of the
flat product on the decoded float32 operands with a fused mac under
round-to-nearest (the cheapest arithmetic, where the decode shows most), 1.05
to 1.08 times under stochastic rounding, and less than half with a split mac,
whose flat counterpart pays for a multiply rounding the block product does not
have (``dev/benchmarks/benchmark_block.py``). The kernels compute in binary32
on the CPU and CUDA; a float64 operand and an Apple GPU tensor raise.
