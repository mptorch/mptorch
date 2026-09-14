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
*carrier*: binary32 for float32, float16 and bfloat16 operands, binary64 for
float64 ones (`float64 and the carrier`_, below).

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

float64 and the carrier
-----------------------

A float64 GEMM is computed in binary64 end to end: every product, every sum
and every rounding. That makes it a different arithmetic from the float32
one, not the same one widened, in three ways.

- **Products of narrow operands are rounded once.** In binary32 a product is
  rounded to 24 bits before the multiply format rounds it again; in binary64
  the product of two operands of at most 26 significant bits -- two float32
  values, or two values a quantizer has rounded -- is exact, and only the
  format rounds it. So a float64 GEMM can round differently even when every
  operand is a float32 value, as the run's first lines show, and agrees with
  the float32 GEMM exactly when every product and sum is exact in both --
  which operands already rounded to a narrow format usually give (the FP8
  recipe in :doc:`layers` agrees to the bit).
- **Wider formats.** Each call holds the formats to its carrier's bounds
  (:doc:`concepts`), so formats past binary32 -- a 30-bit multiply format
  into a 40-bit accumulator, say -- are refused on float32 operands and
  computed exactly on float64 ones.
- **Wider random draws.** Under ``SR`` each rounding draws two words of its
  element's stream, and ``P - 1 + prng_bits`` may reach 52.

``carrier="binary32"`` on a ``SplitMac`` / ``FusedMac``, or on the flat
function, computes float64 operands the float32 way instead: both operands
are narrowed, the float32 kernel runs, and the result is widened -- bit for
bit what the float32 call returns, random draws included. A float32 and a
float64 operand cannot be mixed in one call, with or without it.

.. literalinclude:: ../snippets/gemm_float64.py
   :language: python
   :caption: docs/snippets/gemm_float64.py

.. literalinclude:: ../snippets/gemm_float64.out
   :language: text
   :caption: output

The table shows the other side of the choice. A float64 GEMM in binary32's
own widest format, 24 bits, is barely nearer to float64's product than the
same GEMM with ``carrier="binary32"``: past what the formats keep, the carrier
adds little, and on a GPU it costs (see `Performance notes`_).

Saying which arithmetic
-----------------------

Three value types name an arithmetic. All are frozen: build them once and
hold them, and the library memoizes their resolution.

:class:`~mptorch.quant.SplitMac` ``(mul, acc=None, *, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None)``
   ``mul`` and ``acc`` are each a format, a sequence of formats (a palette,
   see below), or -- for ``acc`` -- ``None``. Both must belong to the same
   family (BinaryK with BinaryK, SuperFP with SuperFP). ``carrier`` is
   ``None`` (the operands' own), ``"binary32"`` or ``"binary64"``.
:class:`~mptorch.quant.FusedMac` ``(fma=None, *, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None)``
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

A format for every element
~~~~~~~~~~~~~~~~~~~~~~~~~~

A dense ``[M, N]`` map is the general case: every output element's dot
product runs in the palette entry its own map entry names, independently of
its row, its column and its neighbours. How the map is chosen is up to the
caller -- a sensitivity analysis, a hardware assignment, a search. The run
below makes one per element: for each of the 20 elements of a product it takes
the narrowest of eight BinaryK formats, from 3 to 24 bits of precision, whose
result stays within 0.1% of the exact dot product there. It then runs the
whole product as a single palette GEMM and checks every element against the
single-format GEMM in that element's format -- equal, to the bit, because
an element's dot product reads only its own row and column of the operands
and its own palette entry.

The same map picks a *pair* when the multiply and the accumulate palettes
differ: entry ``i`` of a ``SplitMac`` palette pairs ``mul[i]`` with ``acc[i]``,
so one element can multiply in 4 bits into a 24-bit sum while its neighbour
does the opposite. A ``FusedMac`` takes a palette the same way, a batched
product takes a ``[B, M, N]`` map with a format per element of every sample,
and the fields a palette shares -- sign, stochastic bits, saturation and
subnormals -- are shared by every element; a palette whose entries disagree
on one is refused, naming the entry.

.. literalinclude:: ../snippets/gemm_palette_elements.py
   :language: python
   :caption: docs/snippets/gemm_palette_elements.py

.. literalinclude:: ../snippets/gemm_palette_elements.out
   :language: text
   :caption: output

A map per pass, held and reused
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
format; ``carrier`` chooses the arithmetic, as on a ``SplitMac``; the
``_mixed`` variants take sequences for the width arguments, a scalar for
anything shared by the palette, and ``prec_idx`` as the third positional
argument. Callers holding one format across many calls should use
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
- **binary64 is slower on a GPU.** A float64 GEMM measured 2.7-3.1x the time
  of the same product computed in binary32 on an RTX 4060 (up to 6x for a
  stochastic split step); on the CPU the two are within a percent. A float64
  model whose formats binary32 can carry recovers the binary32 speed with
  ``carrier="binary32"``, less the narrowing and widening copies.
- **The kernels are simulators.** They are written for exactness and
  flexibility, not for speed, and are orders of magnitude slower than
  cuBLAS. The Linear layers fold their batch into one GEMM, which is the
  shape the kernels do best on.
