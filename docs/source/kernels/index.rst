Quantized numerical kernels
===========================

A quantizer rounds the *operands* of a computation. A matrix product has
another place where rounding can happen: inside, at every step of every dot
product. Consider a tensor core that takes 8-bit inputs and keeps a 32-bit
running sum, an accumulator that keeps only 14 bits, and a fused multiply-add
that rounds once per step. These are different arithmetics, and they give
different results on the same operands. MPTorch's kernels simulate them.

Every such computation runs on **one routine**: a batched matrix product in
which every dot product is computed in the arithmetic you choose. This holds
for a matrix product, the three products of a linear layer, the three passes
of a convolution (its forward product and its two gradients), and a product
of two tensors in a block format. This page
describes that routine. The pages under it cover each part and each
operation:

- :doc:`arithmetic`: what a step of a dot product rounds, and in what order
  the terms are summed;
- :doc:`block`: products of block-format tensors, and where their scales
  enter the sum;
- :doc:`convolutions`: a convolution and its two gradients as matrix products
  whose matrices are never built;
- :doc:`calling`: the functions, objects and modules that run them.

.. toctree::
   :maxdepth: 1

   arithmetic
   block
   convolutions
   calling

Matrix multiplication as the core routine
-----------------------------------------

The routine computes

.. math::

   C_\beta = A_\beta B_\beta, \qquad
   A_\beta \in \mathbb{R}^{M \times K},\;
   B_\beta \in \mathbb{R}^{K \times N}, \qquad
   \beta = 1, \dots, \text{batch},

and each element :math:`c_{\beta i j}` of the result is one dot product of
length :math:`K`, computed as a running sum in a fixed order:

.. math::

   s_0 = 0, \qquad
   s_k = \operatorname{step}\big(s_{k-1},\; a_{\beta i k},\; b_{\beta k j}\big)
   \quad (k = 1, \dots, K), \qquad
   c_{\beta i j} = s_K .

Only four things vary.

**How the operands are read.** The kernel reaches :math:`A` and :math:`B`
only through its tile loads:

- A dense operand is read through its strides. A transposed operand is then
  just a flag, and a broadcast one a stride of zero.
- A convolution's operands are gathered from the convolution's own tensors,
  with zeros for the padding (:doc:`convolutions`).
- A block-format operand is decoded from its codes and scales (:doc:`block`).

Whatever the load, it gives the step the value an explicit matrix would hold
at that position.

**What a step rounds.** A *mac* (multiply-accumulate) specifies it. A
``SplitMac`` rounds the product to one format and the sum to another. A
``FusedMac`` computes a fused multiply-add and rounds it once. Anything the
formats leave unrounded is computed in the *carrier*, the IEEE binary float
the kernel computes in: binary32 or binary64. By default the operands' dtype
chooses it (:doc:`/concepts`, "Carriers").

**In what order the terms are summed.** The *accumulate algorithm* sets the
order: one running sum (``NAIVE``), Kahan's compensated sum (``KAHAN``), or
a two-level sum over blocks (``BLOCK``) or pairwise trees (``TREE``)
(:doc:`arithmetic`). A rounded sum depends on its order, so the order is part
of the definition, and it is the same on every device.

**What each output element has of its own.** A *palette* is a list of
formats, and an integer map gives each output element one of them
(:doc:`arithmetic`). Under stochastic rounding each element draws from its
own random stream, keyed on its index :math:`(\beta, i, j)` into the result,
from one seed per call. A result therefore does not depend on how the work
is tiled or threaded, or on the device:

- The CPU and CUDA kernels return the same bits in every deterministic mode.
- An Apple GPU returns the CPU's bits, stochastic rounding included, except
  for what the GPU's own float unit does between the roundings
  (:ref:`apple-gpu`).

The kernel works on tiles of the operands. On a GPU, a group of threads
stages a :math:`16 \times 16` tile of each operand in shared memory (CUDA)
or threadgroup memory (an Apple GPU), 16 steps of K at a time. On the CPU,
each task packs a :math:`32 \times 32` tile of each operand, 32 steps of K
at a time. The step then runs over the staged tiles. The operations below differ
only in how they load those tiles. That is how one routine serves them all
without slowing any of them: each is bit for bit the routine applied to the
explicit matrices it describes. Each entry point runs the following matrix
products:

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - entry point
     - matrix products per call
     - operands
   * - ``qmatmul``, ``QMatmul``
     - :math:`AB` forward; :math:`G B^\top` and :math:`A^\top G` backward
     - dense; a transpose is a flag, and :math:`[\dots, M, K] \times [K, N]`
       is one product
   * - ``QLinear`` with a ``*_gemm_formats`` factory
     - :math:`x W^\top` forward; :math:`G W` (input gradient) and
       :math:`G^\top x` (weight gradient)
     - dense
   * - ``QConv1d``/``2d``/``3d`` with ``conv_formats``
     - one each for the forward pass and the weight gradient, and one per
       residue class of the stride for the input gradient
     - gathered from the input, the weight and the gradient
   * - ``block_matmul``, ``BlockMac``, ``block_gemm_formats``
     - as the rows above, each operand packed along the dimension its product
       sums over
     - decoded from codes and scales

The run below checks this with one mac and four entry points, each against
the routine on explicit matrices:

- a batched product, slice by slice;
- a linear layer's three products;
- a convolution, against the product of its unfolded input;
- a block product, against the product of its decoded operands.

.. literalinclude:: ../../snippets/kernels_core.py
   :language: python
   :caption: docs/snippets/kernels_core.py

.. literalinclude:: ../../snippets/kernels_core.out
   :language: text
   :caption: output
