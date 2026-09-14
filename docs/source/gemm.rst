Quantized matrix multiplication
===============================

A quantizer rounds the *operands* of a computation. A matrix product has
another place rounding can happen: inside, at every step of every dot
product. A tensor core that takes 8-bit inputs and keeps a 32-bit running
sum, an accumulator that keeps only 14 bits, a fused multiply-add that rounds
once per step -- these are different arithmetics, they give different
results on the same operands, and simulating them is what this part of
MPTorch is for.

What is simulated
-----------------

For :math:`C = AB` with :math:`A \in \mathbb{R}^{M \times K}` and
:math:`B \in \mathbb{R}^{K \times N}`, each output element is a dot product
of length :math:`K`, computed as a running sum in order
:math:`k = 1, \dots, K`. Two arithmetics are implemented, named after the
kernel's own policies.

**SplitMac** -- the product and the sum are rounded separately, to formats
that may differ:

.. math::

   s_0 = 0, \qquad
   s_k = Q_{\text{acc}}\Big(s_{k-1} + Q_{\text{mul}}(a_{ik}\, b_{kj})\Big), \qquad
   c_{ij} = s_K .

With ``acc=None`` the accumulation is left unrounded and only the products
are: :math:`s_k = s_{k-1} + Q_{\text{mul}}(a_{ik} b_{kj})`.

**FusedMac** -- one fused multiply-add per step, rounded once, the way a
hardware FMA unit behaves:

.. math::

   s_k = Q_{\text{fma}}\big(a_{ik}\, b_{kj} + s_{k-1}\big).

The product is exact inside the FMA (the FMA the kernel uses keeps it so),
and with ``fma=None`` nothing is rounded at all: the dot product is a
sequence of FMAs.

In both, the operands :math:`a_{ik}` and :math:`b_{kj}` are read as they
are. Rounding *them* is a quantizer's job, and the layers apply one before
the GEMM; see :doc:`layers`. One rounding mode serves both roundings of a
``SplitMac`` -- it is a compile-time parameter of the kernel, and rounding
the product one way and the sum another would multiply the number of
kernels -- while saturation, subnormal handling and the number of stochastic
bits are properties of each format and can differ between the two.

Everything the formats do not round -- the product before its rounding, the
sum before its, the unrounded accumulator -- is computed in the operands'
dtype's carrier: float32 for float32, float16 and bfloat16 operands, and
float64 for float64 operands. So a float64 GEMM is not simply the float32 one
widened. Its products keep 53 bits where float32 keeps 24, so it can round
differently even when every operand is a float32 value, and under ``SR`` each
rounding draws twice the random words. A float32 and a float64 operand cannot
be mixed in one call.

The following run reproduces a ``SplitMac`` dot product by hand, step by
step, and gets the same bits.

.. literalinclude:: ../snippets/gemm_splitmac.py
   :language: python
   :caption: docs/snippets/gemm_splitmac.py

.. literalinclude:: ../snippets/gemm_splitmac.out
   :language: text
   :caption: output

The same for ``FusedMac``, using float64 to hold the exact product-and-add
before the single rounding:

.. literalinclude:: ../snippets/gemm_fusedmac.py
   :language: python
   :caption: docs/snippets/gemm_fusedmac.py

.. literalinclude:: ../snippets/gemm_fusedmac.out
   :language: text
   :caption: output

The accumulation format is usually the one that matters. Over a long dot
product the rounding of each partial sum compounds, and a format with three
mantissa bits cannot hold a sum of thousands of terms in any useful way:

.. literalinclude:: ../snippets/gemm_accumulation.py
   :language: python
   :caption: docs/snippets/gemm_accumulation.py

.. literalinclude:: ../snippets/gemm_accumulation.out
   :language: text
   :caption: output

Three things to read off this table. Rounding the products to E4M3 costs a
relative error of a few percent whatever the accumulator, because each
product carries the format's :math:`2^{-4}` relative error; a float32
accumulator (``acc=None``, or the same thing spelled as a 24-bit format) adds
nothing to it. A bfloat16 accumulator is noticeably worse and an E4M3 one is
useless. And stochastic rounding does *not* help here -- it is worse than
round-to-nearest -- because the terms have random signs: there is no
systematic bias to remove, only variance to add. Stochastic rounding earns
its keep when the increments are systematically below half a spacing, which
is the accumulator case shown in :doc:`concepts` and the weight-update case
in the :doc:`tutorial`.

Saying which arithmetic
-----------------------

Three value types name an arithmetic. All are frozen: build them once and
hold them, and the library memoizes their resolution.

:class:`~mptorch.quant.SplitMac` ``(mul, acc=None, *, rounding=RNE, accumulate_algorithm=NAIVE)``
   ``mul`` and ``acc`` are each a format, a sequence of formats (a palette,
   see below), or -- for ``acc`` -- ``None``. Both must belong to the same
   family (BinaryK with BinaryK, SuperFP with SuperFP).
:class:`~mptorch.quant.FusedMac` ``(fma=None, *, rounding=RNE, accumulate_algorithm=NAIVE)``
   ``fma`` is a format, a palette, or ``None``.
:class:`~mptorch.quant.Palette` ``(formats)``
   Up to eight formats of one family, selected per output element. A plain
   list works anywhere a ``Palette`` does; the class exists to validate.

``accumulate_algorithm`` selects how partial products are folded into the
sum. Only :attr:`~mptorch.AccumulateAlgorithm.NAIVE` -- the sequential
recurrence above -- is implemented; Kahan, blocked and tree summation are
declared for the future.

qmatmul, qmm and qbmm
---------------------

:func:`~mptorch.quant.qmatmul` is ``torch.matmul`` in the arithmetic you
name, and it is differentiable. Its ``formats`` argument accepts, in
increasing order of specificity:

- ``None`` -- plain ``torch.matmul``, for an A/B against float32;
- a format (``BinaryK(8, 4)``) -- shorthand for ``SplitMac(f, f)``;
- a ``SplitMac`` or ``FusedMac``;
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

.. literalinclude:: ../snippets/gemm_qmatmul.py
   :language: python
   :caption: docs/snippets/gemm_qmatmul.py

.. literalinclude:: ../snippets/gemm_qmatmul.out
   :language: text
   :caption: output

QMatmulFormats and QMatmul
--------------------------

:class:`~mptorch.quant.QMatmulFormats` holds everything a quantized matmul
can vary. Its slots mirror the layer formats of :doc:`layers`, renamed for an
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

.. literalinclude:: ../snippets/qmatmul_module.py
   :language: python
   :caption: docs/snippets/qmatmul_module.py

.. literalinclude:: ../snippets/qmatmul_module.out
   :language: text
   :caption: output

Palettes: a format per output element
-------------------------------------

A ``Palette`` in any format slot selects the *spatially-varying* kernel: each
output element's dot product runs in the palette entry that an integer map,
``prec_idx``, assigns to it. The map has the output's shape ``[M, N]``, or
``[M, 1]`` for one format per row, or ``[1, N]`` for one per column,
optionally with a leading batch dimension (of the batch size, or 1 to share
the map). Values must lie in ``[0, len(palette))``. A palette's entries must
agree on everything but their widths -- sign, stochastic bits, saturation and
subnormal policy -- because the kernel tabulates only the widths.

.. literalinclude:: ../snippets/gemm_palette.py
   :language: python
   :caption: docs/snippets/gemm_palette.py

.. literalinclude:: ../snippets/gemm_palette.out
   :language: text
   :caption: output

A map is indexed by *output* element, and the three passes of a
differentiable matmul have three output shapes -- ``[M, N]``, ``[M, K]`` and
``[K, N]`` -- so a palette that has to differentiate needs a map per pass,
given to ``matmul_formats`` as ``prec_idx``, ``agrad_prec_idx`` and
``bgrad_prec_idx``. A pass whose map is missing raises when it is reached,
naming the argument, rather than falling back to something unquantized.

Hold the map and reuse it. The bounds check on its values is memoized
against the tensor you pass, and an ``int32``, contiguous map on the
operands' device is handed to the kernel untouched; a map rebuilt on every
call is re-checked and re-packed on every call.

The schema tier
---------------

Under the value vocabulary sit eight functions, one per kernel, with every
argument of the kernel's schema spelled out and no autograd:
``binaryK_matmul``, ``superfp_matmul``, their ``_fma`` variants, and the
``_mixed`` (palette) variants of all four. They exist for scripts that want
to say exactly what the kernel receives, and they are what the vocabulary
resolves to -- the test suite asserts that a ``SplitMac`` and the equivalent
flat call produce the same resolved kernel arguments.

.. literalinclude:: ../snippets/gemm_raw_ops.py
   :language: python
   :caption: docs/snippets/gemm_raw_ops.py

.. literalinclude:: ../snippets/gemm_raw_ops.out
   :language: text
   :caption: output

The conventions are uniform across the eight: ``mul_*`` and ``acc_*`` name
the two formats of a split multiply-accumulate (``acc_*`` default to the
``mul_*`` values, ``accumulate_quant=False`` leaves the sum unrounded);
``fma_*`` names the one format of a fused step (``fma_quant=False`` leaves it
unrounded); ``trans_a``/``trans_b`` transpose the last two dimensions of an
operand without copying it; ``*_prng_bits`` set the stochastic bits per
format; the ``_mixed`` variants take sequences for the width arguments, a
scalar for anything shared by the palette, and ``prec_idx`` as the third
positional argument. Callers holding one format across many calls should use
the factories or ``qmatmul`` instead: these functions re-derive the format's
defaults on every call, which the resolved objects avoid.

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
- **The kernels are simulators.** They are written for exactness and
  flexibility, not for speed, and are orders of magnitude slower than
  cuBLAS. The Linear layers fold their batch into one GEMM, which is the
  shape the kernels do best on.
