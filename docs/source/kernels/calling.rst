Calling the kernels
===================

The entry points to the core routine (:doc:`index`), from the value objects
that name an arithmetic to the functions that spell out a kernel's every
argument. The equations are :doc:`arithmetic`'s, :doc:`block`'s and
:doc:`convolutions`'; this page says how to ask for them.

Saying which arithmetic
-----------------------

Four value types name an arithmetic. All are frozen: build them once and
hold them, and the library memoizes their resolution.

:class:`~mptorch.quant.SplitMac` ``(mul, acc=None, *, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None)``
   ``mul`` and ``acc`` are each a format, a sequence of formats (a palette,
   :doc:`arithmetic`), or -- for ``acc`` -- ``None``. Both must belong to the same
   family (BinaryK with BinaryK, SuperFP with SuperFP). ``carrier`` is
   ``None`` (the operands' own), ``torch.float32`` (binary32) or
   ``torch.float64`` (binary64).
:class:`~mptorch.quant.FusedMac` ``(fma=None, *, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None)``
   ``fma`` is a format, a palette, or ``None``.
:class:`~mptorch.quant.Palette` ``(formats)``
   Up to eight formats of one family, selected per output element. A plain
   list works anywhere a ``Palette`` does; the class exists to validate.
:class:`~mptorch.quant.BlockMac` ``(a, b=None, acc=None, *, fused=False, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None, tensor_scale_epilogue=False)``
   A product of two block-format operands: ``a`` and ``b`` are each a
   :class:`~mptorch.BlockFormat` or a :class:`~mptorch.quant.BlockQuant`
   (which names the rounding and a tensor scale), ``acc`` a
   :class:`~mptorch.BinaryK` or ``None``; the multiply is binary32's own
   (:doc:`block`). ``tensor_scale_epilogue`` moves the operands' tensor scales
   out of the loop.

``accumulate_algorithm`` selects how partial products are folded into the
sum: :attr:`~mptorch.AccumulateAlgorithm.NAIVE`, the sequential recurrence,
or ``KAHAN``, ``BLOCK`` or ``TREE`` (:doc:`arithmetic`, "Summing
differently"). ``block_size`` is ``BLOCK``'s and ``TREE``'s, and ``outer`` is
the one format, of the mac's family, that they round the total to; ``None``
leaves it in the carrier's precision. A mac that names an algorithm other
than ``NAIVE`` holds one format per slot, not a palette.

qmatmul, qmm and qbmm
---------------------

:func:`~mptorch.quant.qmatmul` is ``torch.matmul`` in the arithmetic you
name, and it is differentiable. Its ``formats`` argument accepts, in
increasing order of specificity:

- ``None`` -- plain ``torch.matmul``, for an A/B against float32;
- a format (``BinaryK(8, 4)``) -- shorthand for ``SplitMac(f, f)``;
- a ``SplitMac`` or ``FusedMac``;
- a :class:`~mptorch.quant.BlockMac` (or a bare
  :class:`~mptorch.BlockFormat`), which packs both operands in block formats
  and multiplies them with the block product (`Block products`_, below);
- a :class:`~mptorch.quant.QMatmulFormats`, which also carries quantizers
  for the operands and for each gradient (next section).

It accepts everything ``torch.matmul`` accepts: matrices, batches of them,
broadcasting between leading dimensions, and 1D operands promoted and
squeezed back the same way. Two common shapes cost no copy: an operand that
is the transpose of a contiguous tensor (``q @ k.mT``) sets the kernel's
transpose flag instead of being materialized, and ``[..., M, K] @ [K, N]``
folds its batch into ``M`` for one large product rather than a batch of
small ones. Both are bit-identical to the batched spelling, not merely
close -- stochastic rounding keys each output element's random stream on its
global index, which is the same either way.

The gradients are the usual ones, each computed as a GEMM in the arithmetic
its own hook names (by default, the forward's):

.. math::

   \frac{\partial L}{\partial A} = G\, B^{\top}, \qquad
   \frac{\partial L}{\partial B} = A^{\top} G, \qquad
   G = \frac{\partial L}{\partial C},

followed by a reduction over any dimensions the forward broadcast.
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
can vary. Its slots mirror the layer formats of :doc:`/layers`, renamed for an
operation whose two operands have equal standing:

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

Every slot is optional and ``None`` means "plain PyTorch". There is
deliberately no ``output_quant``: rounding the result is an elementwise step,
which is a :class:`~mptorch.quant.Quantizer` you apply after the product.

:func:`~mptorch.quant.matmul_formats` builds one with the three math hooks
set to a ``SplitMac``/``FusedMac``'s arithmetic (and nothing else, so
quantizers are layered on separately), and :class:`~mptorch.quant.QMatmul`
is the module form: an ``nn.Module`` holding a ``QMatmulFormats``, so that a
quantizer with state is registered in the model. Two of them are what a
quantized attention head calls:

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
passes of a differentiable matmul have three output shapes -- ``[M, N]``,
``[M, K]`` and ``[K, N]`` -- so a palette that has to differentiate needs a
map per pass,
given to ``matmul_formats`` as ``prec_idx``, ``agrad_prec_idx`` and
``bgrad_prec_idx``. A pass whose map is missing raises when it is reached,
naming the argument, rather than falling back to something unquantized.

Hold the map and reuse it. The bounds check on its values is memoized
against the tensor you pass, and an ``int32``, contiguous map on the
operands' device is handed to the kernel untouched; a map rebuilt on every
call is re-checked and re-packed on every call.

Block products
--------------

:func:`~mptorch.quant.block_matmul` multiplies two packed tensors
(:func:`~mptorch.quant.block_pack`'s :class:`~mptorch.quant.BlockPacked`),
``a`` logically ``[..., M, K]`` and ``b`` ``[..., K, N]``, each in its own
format, so an MXFP4 weight multiplies an MXFP8 activation. It is the block
product of :doc:`block`, and so the flat binaryK product on the decoded
operands with a binary32 multiply, bit for bit:

.. literalinclude:: ../../snippets/block_gemm.py
   :language: python
   :caption: docs/snippets/block_gemm.py

.. literalinclude:: ../../snippets/block_gemm.out
   :language: text
   :caption: output

Each operand is read along K. An operand packed along its K is the natural
case (a weight ``[N, K]`` blocked along K, read through
:attr:`~mptorch.quant.BlockPacked.mT`); one packed along its other dimension
is read transposed. :class:`~mptorch.quant.BlockMac` is the value spelling,
and :func:`~mptorch.quant.qmatmul` and :class:`~mptorch.quant.QMatmul` take
one (a bare ``BlockFormat`` is ``BlockMac(fmt)``): the forward packs ``a`` and
``b`` along their K, and each gradient pass packs what it multiplies along the
dimension it sums over, as :doc:`block` explains.
:func:`~mptorch.quant.block_matmul_formats` builds the ``QMatmulFormats``, one
``BlockMac`` per pass. An operand whose format has a square tile is packed
once, in the forward, and the gradient passes read that packing transposed.
``tensor_scale_epilogue=True``, on ``block_matmul``, a ``BlockMac`` or the
layer factory :func:`~mptorch.quant.block_gemm_formats`, applies the
operands' per-tensor scales to the result instead of in the decode
(:doc:`block`, "Tensor scales").

Convolutions
------------

The convolutions have eight operators of their own, ``custom_conv_*``, one
per matrix-product operator family, which take the convolution's geometry in
place of the transpose flags. They have no functions of their own: a
``QConv1d``/``2d``/``3d`` reaches them through
:func:`~mptorch.quant.conv_formats`, which fills its three math hooks from a
mac as :func:`~mptorch.quant.matmul_formats` does a matmul's
(:doc:`/layers`, "Convolutions"), and :doc:`convolutions` has what they
compute.

The schema tier
---------------

Under the value vocabulary sit eight functions, one per kernel, with every
argument of the kernel's schema spelled out and no autograd:
``binaryK_matmul``, ``superfp_matmul``, their ``_fma`` variants, and the
``_mixed`` (palette) variants of all four. They exist for scripts that want
to say exactly what the kernel receives, and they are what the vocabulary
resolves to -- the test suite asserts that a ``SplitMac`` and the equivalent
flat call produce the same resolved kernel arguments.

.. literalinclude:: ../../snippets/gemm_raw_ops.py
   :language: python
   :caption: docs/snippets/gemm_raw_ops.py

.. literalinclude:: ../../snippets/gemm_raw_ops.out
   :language: text
   :caption: output

The conventions are uniform across the eight: ``mul_*`` and ``acc_*`` name
the two formats of a split multiply-accumulate (``acc_*`` default to the
``mul_*`` values, ``accumulate_quant=False`` leaves the sum unrounded);
``fma_*`` names the one format of a fused step (``fma_quant=False`` leaves it
unrounded); ``trans_a``/``trans_b`` transpose the last two dimensions of an
operand without copying it; ``*_prng_bits`` set the stochastic bits per
format; ``carrier`` chooses the arithmetic, as on a ``SplitMac``; the
``_mixed`` variants take sequences for the width arguments, a scalar for
anything shared by the palette, and ``prec_idx`` as the third positional
argument. The four single-format functions also take
``accumulate_algorithm``'s ``block_size`` and the outer format, spelled
``outer_*`` the way the family spells a format; naming an algorithm other
than ``NAIVE`` routes the call to the kernel's ``*_accumulated`` twin, a
separate operator, so that a ``NAIVE`` call is exactly the call it was before
the other three existed. Callers holding one format across many calls should use
the factories or ``qmatmul`` instead: these functions re-derive the format's
defaults on every call, which the resolved objects avoid. The convolutions'
operators have no functions at this tier (`Convolutions`_, above), and the
block product's is :func:`~mptorch.quant.block_matmul` itself.

Performance notes
-----------------

- **Resolve once.** ``SplitMac``/``FusedMac``/``BinaryK``/``SuperFP`` are
  frozen and hashable, and ``qmatmul`` memoizes the resolution of the
  ``formats`` it is given. Constructing a new ``SplitMac`` per call still
  hits the cache; constructing a ``QMatmulFormats`` per call does not.
  Holding a ``QMatmul`` module, or the ``QMatmulFormats`` a factory returned,
  pays nothing per call beyond the kernel.
- **One launch per batch.** A batched call is one kernel launch (one
  parallel region on CPU) over the whole batch, 2-4x faster than the loop of
  2D calls it replaces on attention-shaped operands
  (``dev/benchmarks/benchmark_qmatmul.py``).
- **Stochastic rounding is not free.** Each rounding draws from a
  per-element Philox stream; expect a ``RoundMode.SR`` GEMM to be slower
  than a deterministic one.
- **binary64 is slower on a GPU.** A float64 GEMM measured 2.7-3.1x the time
  of the same product computed in binary32 on an RTX 4060 (up to 6x for a
  stochastic split step); on the CPU the two are within a percent. A float64
  model whose formats binary32 can carry recovers the binary32 speed as a
  float32 model, and a float32 one naming ``carrier=torch.float64`` pays the
  binary64 time plus the widening and narrowing copies.
- **Block products and convolutions cost what their matrix products cost.**
  The decode of a block operand adds 5-18% to the flat product on the decoded
  operands (:doc:`block`, "How the kernel decodes"), and a convolution's
  gathered loads a few percent of address arithmetic (:doc:`convolutions`,
  "What it costs").
- **The kernels are simulators.** They are written for exactness and
  flexibility, not for speed, and are orders of magnitude slower than
  cuBLAS. The Linear layers fold their batch into one GEMM, which is the
  shape the kernels do best on.
