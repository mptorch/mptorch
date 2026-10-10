Calling the kernels
===================

This page lists the entry points to the core routine (:doc:`index`), from the
value objects that describe an arithmetic to the functions that spell out
every argument of a kernel. The equations are in :doc:`arithmetic`,
:doc:`block` and :doc:`convolutions`. This page explains how to request them.

Saying which arithmetic
-----------------------

Four value types describe an arithmetic. All are frozen, so build them once
and keep them; the library memoizes their resolution.

:class:`~mptorch.quant.SplitMac` ``(mul, acc=None, *, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None)``
   ``mul`` and ``acc`` are each a format or a sequence of formats (a palette,
   :doc:`arithmetic`). ``acc`` can also be ``None``. Both must belong to the
   same family (BinaryK with BinaryK, SuperFP with SuperFP). ``carrier`` is
   ``None`` (the operands' own), ``torch.float32`` (binary32) or
   ``torch.float64`` (binary64).
:class:`~mptorch.quant.FusedMac` ``(fma=None, *, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None)``
   ``fma`` is a format, a palette, or ``None``.
:class:`~mptorch.quant.Palette` ``(formats)``
   Up to eight formats of one family, selected per output element. A plain
   list works anywhere a ``Palette`` does; the class exists to validate.
:class:`~mptorch.quant.BlockMac` ``(a, b=None, acc=None, *, fused=False, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None, tensor_scale_epilogue=False)``
   A product of two block-format operands. ``a`` and ``b`` are each a
   :class:`~mptorch.BlockFormat` or a :class:`~mptorch.quant.BlockQuant`
   (which specifies the rounding and a tensor scale). ``acc`` is a
   :class:`~mptorch.BinaryK` or ``None``. The multiply is binary32's own
   (:doc:`block`). ``tensor_scale_epilogue`` moves the operands' tensor
   scales out of the loop.

``accumulate_algorithm`` selects how partial products are added to the sum:
:attr:`~mptorch.AccumulateAlgorithm.NAIVE`, the sequential recurrence, or
``KAHAN``, ``BLOCK`` or ``TREE`` (:doc:`arithmetic`, "Summing differently").
``block_size`` applies to ``BLOCK`` and ``TREE``. ``outer`` is the one format
of the mac's family that they round the total to; ``None`` leaves the total
in the carrier's precision. A mac with an algorithm other than ``NAIVE``
holds one format per slot, not a palette.

qmatmul, qmm and qbmm
---------------------

:func:`~mptorch.quant.qmatmul` is ``torch.matmul`` in the arithmetic you
choose, and it is differentiable. Its ``formats`` argument accepts, from the
least to the most specific:

- ``None``: plain ``torch.matmul``, for a comparison against float32;
- a format (``BinaryK(8, 4)``): shorthand for ``SplitMac(f, f)``;
- a ``SplitMac`` or ``FusedMac``;
- a :class:`~mptorch.quant.BlockMac` (or a bare
  :class:`~mptorch.BlockFormat`), which packs both operands in block formats
  and multiplies them with the block product (`Block products`_, below);
- a :class:`~mptorch.quant.QMatmulFormats`, which also carries quantizers
  for the operands and for each gradient (next section).

It accepts everything ``torch.matmul`` accepts: matrices, batches of
matrices, broadcasting between leading dimensions, and 1D operands, which
are promoted and squeezed back in the same way. Two common shapes need no
copy:

- An operand that is the transpose of a contiguous tensor (``q @ k.mT``)
  sets the kernel's transpose flag and is not materialized.
- ``[..., M, K] @ [K, N]`` folds its batch into ``M``, giving one large
  product instead of a batch of small ones.

Both are bit-identical to the batched spelling, not merely close. Stochastic
rounding keys each output element's random stream on its global index, which
is the same either way.

The gradients are the usual ones. Each is computed as a GEMM in the
arithmetic of its own hook (by default, the forward's):

.. math::

   \frac{\partial L}{\partial A} = G\, B^{\top}, \qquad
   \frac{\partial L}{\partial B} = A^{\top} G, \qquad
   G = \frac{\partial L}{\partial C},

followed by a reduction over any dimensions that the forward pass broadcast.
:func:`~mptorch.quant.qmm` and :func:`~mptorch.quant.qbmm` are the same
function with the rank checks of ``torch.mm`` and ``torch.bmm``.

.. literalinclude:: ../../snippets/gemm_qmatmul.py
   :language: python
   :caption: docs/snippets/gemm_qmatmul.py

.. literalinclude:: ../../snippets/gemm_qmatmul.out
   :language: text
   :caption: output

QMatmulFormats and QMatmul
--------------------------

:class:`~mptorch.quant.QMatmulFormats` holds everything a quantized matmul
can vary. Its slots mirror the layer formats of :doc:`/layers`, renamed for
an operation whose two operands play the same role:

===================  ======================================================================
slot                 what it does
===================  ======================================================================
``a_quant``          quantizer applied to ``a`` before the product
``b_quant``          quantizer applied to ``b`` before the product
``agrad_quant``      quantizer applied to :math:`G` before ``a``'s gradient is computed
``bgrad_quant``      quantizer applied to :math:`G` before ``b``'s gradient is computed
``fwd_math``         the forward product, ``(q_a, q_b) -> q_a @ q_b``
``bwd_agrad_math``   ``a``'s gradient product, ``(q_grad, q_b) -> q_grad @ q_b^T``
``bwd_bgrad_math``   ``b``'s gradient product, ``(q_grad, q_a) -> q_a^T @ q_grad``
===================  ======================================================================

Every slot is optional, and ``None`` means "plain PyTorch". There is no
``output_quant``, on purpose: rounding the result is an elementwise step, so
apply a :class:`~mptorch.quant.Quantizer` after the product.

:func:`~mptorch.quant.matmul_formats` builds a ``QMatmulFormats`` with the
three math hooks set to the arithmetic of a ``SplitMac`` or ``FusedMac``. It
sets nothing else, so quantizers are added separately.
:class:`~mptorch.quant.QMatmul` is the module form: an ``nn.Module`` holding
a ``QMatmulFormats``, so that a quantizer with state is registered in the
model. A quantized attention head calls two of them:

.. literalinclude:: ../../snippets/qmatmul_module.py
   :language: python
   :caption: docs/snippets/qmatmul_module.py

.. literalinclude:: ../../snippets/qmatmul_module.out
   :language: text
   :caption: output

Palettes: a map per pass, held and reused
-----------------------------------------

A palette (:doc:`arithmetic`, "Palettes") picks a format per output element
through an integer map, ``prec_idx``, indexed by *output* element. The three
passes of a differentiable matmul have three output shapes: ``[M, N]``,
``[M, K]`` and ``[K, N]``. A palette that has to differentiate therefore
needs one map per pass, given to ``matmul_formats`` as ``prec_idx``,
``agrad_prec_idx`` and ``bgrad_prec_idx``. If a pass's map is missing, the
pass raises when it is reached and names the argument. It does not fall back
to something unquantized.

Keep the map and reuse it. The bounds check on its values is memoized
against the tensor you pass, and an ``int32``, contiguous map on the
operands' device is passed to the kernel unchanged. A map rebuilt on every
call is re-checked and re-packed on every call.

Block products
--------------

:func:`~mptorch.quant.block_matmul` multiplies two packed tensors, the
:class:`~mptorch.quant.BlockPacked` objects that
:func:`~mptorch.quant.block_pack` returns. ``a`` is logically
``[..., M, K]`` and ``b`` is ``[..., K, N]``. Each has its own format, so an
MXFP4 weight can multiply an MXFP8 activation. This is the block product of
:doc:`block`. It is bit for bit the flat binaryK product on the decoded
operands with a binary32 multiply:

.. literalinclude:: ../../snippets/block_gemm.py
   :language: python
   :caption: docs/snippets/block_gemm.py

.. literalinclude:: ../../snippets/block_gemm.out
   :language: text
   :caption: output

Each operand is read along K. An operand packed along its K is the natural
case: a weight ``[N, K]`` blocked along K is read through
:attr:`~mptorch.quant.BlockPacked.mT`. An operand packed along its other
dimension is read transposed.

:class:`~mptorch.quant.BlockMac` is the value spelling.
:func:`~mptorch.quant.qmatmul` and :class:`~mptorch.quant.QMatmul` accept
one, and a bare ``BlockFormat`` means ``BlockMac(fmt)``. The forward pass
packs ``a`` and ``b`` along their K. Each gradient pass packs what it
multiplies along the dimension it sums over, as :doc:`block` explains.
:func:`~mptorch.quant.block_matmul_formats` builds the ``QMatmulFormats``,
with one ``BlockMac`` per pass. An operand whose format has a square tile is
packed once, in the forward pass, and the gradient passes read that packing
transposed.

``tensor_scale_epilogue=True`` applies the operands' per-tensor scales to
the result instead of in the decode (:doc:`block`, "Tensor scales"). It is
accepted by ``block_matmul``, by a ``BlockMac`` and by the layer factory
:func:`~mptorch.quant.block_gemm_formats`.

Convolutions
------------

The convolutions have eight operators of their own, ``custom_conv_*``, one
per matrix-product operator family. They take the convolution's geometry in
place of the transpose flags. There are no Python functions for them. A
``QConv1d``/``2d``/``3d`` reaches them through
:func:`~mptorch.quant.conv_formats`, which fills the layer's three math
hooks from a mac, as :func:`~mptorch.quant.matmul_formats` does for a matmul
(:doc:`/layers`, "Convolutions"). :doc:`convolutions` describes what they
compute.

The schema tier
---------------

Below the value vocabulary there are eight functions, one per kernel. Each
spells out every argument of the kernel's schema and has no autograd:
``binaryK_matmul``, ``superfp_matmul``, their ``_fma`` variants, and the
``_mixed`` (palette) variants of all four. They are for scripts that need to
state exactly what the kernel receives. The vocabulary resolves to them: the
test suite asserts that a ``SplitMac`` and the equivalent flat call produce
the same resolved kernel arguments.

.. literalinclude:: ../../snippets/gemm_raw_ops.py
   :language: python
   :caption: docs/snippets/gemm_raw_ops.py

.. literalinclude:: ../../snippets/gemm_raw_ops.out
   :language: text
   :caption: output

The eight functions share their conventions:

- ``mul_*`` and ``acc_*`` give the two formats of a split
  multiply-accumulate. ``acc_*`` default to the ``mul_*`` values, and
  ``accumulate_quant=False`` leaves the sum unrounded.
- ``fma_*`` gives the one format of a fused step, and ``fma_quant=False``
  leaves it unrounded.
- ``trans_a`` and ``trans_b`` transpose the last two dimensions of an
  operand without copying it.
- ``*_prng_bits`` set the stochastic bits per format.
- ``carrier`` chooses the arithmetic, as on a ``SplitMac``.
- The ``_mixed`` variants take sequences for the width arguments, a scalar
  for anything the palette shares, and ``prec_idx`` as the third positional
  argument.

The four single-format functions also take ``accumulate_algorithm``, its
``block_size`` and the outer format, spelled ``outer_*`` in the same way as
the family's other formats. An algorithm other than ``NAIVE`` routes the
call to the kernel's ``*_accumulated`` twin, which is a separate operator. A
``NAIVE`` call is therefore exactly the call it was before the other three
algorithms existed.

Callers that keep one format across many calls should use the factories or
``qmatmul`` instead. These functions re-derive the format's defaults on
every call, which the resolved objects avoid. The convolutions' operators
have no functions at this tier (`Convolutions`_, above), and the block
product's function is :func:`~mptorch.quant.block_matmul` itself.

Performance notes
-----------------

- **Resolve once.** ``SplitMac``/``FusedMac``/``BinaryK``/``SuperFP`` are
  frozen and hashable, and ``qmatmul`` memoizes the resolution of the
  ``formats`` it is given. Constructing a new ``SplitMac`` per call still
  hits the cache. Constructing a ``QMatmulFormats`` per call does not.
  Keeping a ``QMatmul`` module, or the ``QMatmulFormats`` that a factory
  returned, costs nothing per call beyond the kernel.
- **One launch per batch.** A batched call is one kernel launch (one
  parallel region on the CPU) over the whole batch. On attention-shaped
  operands it is 2-4x faster than the loop of 2D calls it replaces
  (``dev/benchmarks/benchmark_qmatmul.py``).
- **Stochastic rounding is not free.** Each rounding draws from a
  per-element Philox stream, so a ``RoundMode.SR`` GEMM is slower than a
  deterministic one.
- **binary64 is slower on a GPU.** On an RTX 4060 a float64 GEMM took
  2.7-3.1x the time of the same product computed in binary32 (up to 6x for a
  stochastic split step). On the CPU the two are within a percent. A float64
  model whose formats binary32 can carry gets the binary32 speed back as a
  float32 model. A float32 model with ``carrier=torch.float64`` pays the
  binary64 time plus the widening and narrowing copies.
- **Block products and convolutions cost what their matrix products cost.**
  Decoding a block operand adds 5-18% to the flat product on the decoded
  operands (:doc:`block`, "How the kernel decodes"). A convolution's
  gathered loads add a few percent of address arithmetic
  (:doc:`convolutions`, "What it costs").
- **The kernels are simulators.** They are written for exactness and
  flexibility, not for speed, and are orders of magnitude slower than
  cuBLAS. The Linear layers fold their batch into one GEMM, which is the
  shape the kernels handle best.
