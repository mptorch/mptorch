Calling the kernels
===================

This page lists the entry points to the core routine (:doc:`index`). They
range from value objects that describe an arithmetic to functions that spell
out every argument of a kernel. The equations are in :doc:`arithmetic`,
:doc:`block` and :doc:`convolutions`. This page explains how to request them.

Saying which arithmetic
-----------------------

Four value types describe an arithmetic. Three of them, ``SplitMac``,
``FusedMac`` and ``BlockMac``, each describe a *mac* (multiply-accumulate):
what one step of a dot product rounds, and how the steps are summed. The
fourth, ``Palette``, is a set of formats that a slot of a mac can hold. All
four are frozen dataclasses, so build them once and keep them. The library
memoizes what it derives from them.

:class:`~mptorch.quant.SplitMac` ``(mul, acc=None, *, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None, carrier_rounding=RNE, overflow=SATURATE)``
   Rounds each product to ``mul`` and each partial sum to ``acc``. Each is a
   format or a *palette*, a sequence of formats from which each output
   element selects one (:doc:`arithmetic`). ``acc`` can also be ``None``,
   which leaves the sum unrounded. The two may be of different families (a
   BinaryK multiply with a SuperFP accumulate, or the reverse) when each
   holds one format. Two :class:`~mptorch.FixedPoint` formats make a
   fixed-point mac, whose sums are exact (:doc:`arithmetic`, "Fixed-point
   macs").
:class:`~mptorch.quant.FusedMac` ``(fma, *, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None, carrier_rounding=RNE, overflow=SATURATE)``
   Rounds each fused multiply-add to ``fma``: a format, a palette, or
   ``None``. ``fma`` has no default.
:class:`~mptorch.quant.Palette` ``(formats)``
   Up to eight formats of one family, selected per output element. A plain
   list works anywhere a ``Palette`` does. The class exists to validate the
   entries.
:class:`~mptorch.quant.BlockMac` ``(a, b=None, acc=None, *, fused=False, rounding=RNE, accumulate_algorithm=NAIVE, carrier=None, block_size=None, outer=None, tensor_scale_epilogue=False)``
   A product of two block-format operands. ``a`` and ``b`` are each a
   :class:`~mptorch.BlockFormat` or a
   :class:`~mptorch.quant.BlockQuant` (which also gives the rounding mode and
   a tensor scale). ``b`` defaults to ``a``. ``acc`` is a
   :class:`~mptorch.BinaryK` or ``None``. Only the carrier rounds each
   product (:doc:`block`). ``tensor_scale_epilogue`` applies the operands'
   tensor scales to the result instead of inside the loop.

The keywords the macs share:

- ``rounding`` is one :class:`~mptorch.RoundMode` for every rounding
  of the mac.
- ``carrier`` is the float the kernel computes in: ``None`` for the operands'
  own, ``torch.float32`` for binary32, or ``torch.float64`` for binary64.
- ``accumulate_algorithm`` selects how the products are added to the sum:
  :attr:`~mptorch.AccumulateAlgorithm.NAIVE`, the sequential
  recurrence, or ``KAHAN``, ``BLOCK`` or ``TREE`` (:doc:`arithmetic`,
  "Summing differently: KAHAN, BLOCK and TREE"). A mac with an algorithm
  other than ``NAIVE`` holds one format per slot, not a palette.
- ``block_size`` applies to ``BLOCK`` and ``TREE``.
- ``outer`` is the one format that ``BLOCK`` and ``TREE`` round the total
  to. It is of the accumulate format's family (the multiply format's when
  ``acc`` is ``None``). ``None`` leaves the total in the carrier's precision.
- ``carrier_rounding`` is ``RoundMode.RNE`` (the carrier's ordinary
  arithmetic) or ``RoundMode.RO``, which rounds the carrier's steps to odd
  (:doc:`arithmetic`, "What the carrier rounds"). ``BlockMac`` does not take
  it.
- ``overflow`` is :class:`~mptorch.Overflow`'s ``SATURATE`` (the default) or
  ``WRAP``, which wraps a fixed-point sum around its range instead of
  saturating. It needs a ``FixedPoint`` accumulate (or fused) format. A
  fixed-point mac takes only ``NAIVE``, one format per slot and
  ``carrier_rounding=RoundMode.RNE``.

qmatmul, qmm and qbmm
---------------------

:func:`~mptorch.quant.qmatmul` is ``torch.matmul`` in the arithmetic
you choose, and it is differentiable. Its ``formats`` argument accepts, from
the least to the most specific:

- ``None``: plain ``torch.matmul``, for a comparison against float32;
- a ``BinaryK`` or ``SuperFP`` format ``f`` (such as ``BinaryK(8, 4)``):
  shorthand for ``SplitMac(f, f)``;
- a ``SplitMac`` or ``FusedMac``;
- a :class:`~mptorch.quant.BlockMac` (or a bare
  :class:`~mptorch.BlockFormat`), which packs both operands in block
  formats and multiplies them with the block product (`Block products`_,
  below);
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

The gradients are the usual ones:

.. math::

   \frac{\partial L}{\partial A} = G\, B^{\top}, \qquad
   \frac{\partial L}{\partial B} = A^{\top} G, \qquad
   G = \frac{\partial L}{\partial C}.

Each is a GEMM in the arithmetic of its own hook, which is the forward's by
default. A gradient is then summed over any dimensions that the forward pass
broadcast.

:func:`~mptorch.quant.qmm` and :func:`~mptorch.quant.qbmm`
are the same function with the rank checks of ``torch.mm`` and
``torch.bmm``. Like ``torch.bmm``, ``qbmm`` does not broadcast: the two
batch sizes must be equal.

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

Two flags, ``recompute_a`` and ``recompute_b``, choose what the forward pass
keeps for backward. By default it keeps a quantized copy of each operand.
With the flag set it keeps the raw operand and quantizes it again in
backward, as the layers' ``recompute_*`` flags do (:doc:`/layers`, "What a
layer keeps for backward").

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

A palette (:doc:`arithmetic`, "Palettes: a format per output element") picks
a format per output element through an integer map, ``prec_idx``, indexed by
*output* element. A differentiable matmul runs three *passes*, one matrix
product each: the forward product and the two gradients. They have three
output shapes: ``[M, N]``, ``[M, K]`` and ``[K, N]``. A palette that has to
differentiate therefore needs one map per pass, given to ``matmul_formats``
as ``prec_idx``, ``agrad_prec_idx`` and ``bgrad_prec_idx``. If a pass's map
is missing, the pass raises when it is reached, and the error names the
argument. It does not fall back to something unquantized.

Keep the map and reuse it:

- The kernel reads an ``int32``, contiguous map on the operands' device. A
  map that is already one is passed unchanged. Any other map is converted on
  every call.
- On a GPU, checking the map's bounds means copying them to the host, which
  waits for the device. That check is memoized against the tensor you pass,
  so a map that comes back unchanged is checked once. A map rebuilt on every
  call is checked on every call.

Block products
--------------

:func:`~mptorch.quant.block_matmul` multiplies two packed tensors, the
:class:`~mptorch.quant.BlockPacked` objects that
:func:`~mptorch.quant.block_pack` returns. ``a`` is logically
``[..., M, K]`` and ``b`` is ``[..., K, N]``. Each has its own format, so an
MXFP4 weight can multiply an MXFP8 activation. This is the block product of
:doc:`block`. In binary32 it is bit for bit the flat binaryK product on the
decoded operands, with a multiply format that rounds nothing (binary32's own
layout as a ``BinaryK``):

.. literalinclude:: ../../snippets/block_gemm.py
   :language: python
   :caption: docs/snippets/block_gemm.py

.. literalinclude:: ../../snippets/block_gemm.out
   :language: text
   :caption: output

The kernel reads each operand along K. An operand packed along its K is the
natural case. An operand packed along its other dimension is read
transposed. A weight ``[N, K]`` blocked along K, for example, enters
``x @ w.T`` through :attr:`~mptorch.quant.BlockPacked.mT`, without a copy.

:class:`~mptorch.quant.BlockMac` is the value spelling.
:func:`~mptorch.quant.qmatmul` and :class:`~mptorch.quant.QMatmul`
accept one, and a bare ``BlockFormat`` means ``BlockMac(fmt)``. The forward pass
packs ``a`` and ``b`` along their K. Each gradient pass packs what it
multiplies along the dimension it sums over, as :doc:`block` explains.
:func:`~mptorch.quant.block_matmul_formats` builds the ``QMatmulFormats``,
with one ``BlockMac`` per pass. An operand whose format has a square tile is
packed once, in the forward pass, and the gradient passes read that packing
transposed.

``tensor_scale_epilogue=True`` applies the operands' per-tensor scales to
the result instead of in the decode (:doc:`block`, "Tensor scales: in the
loop or after it"). It is accepted by ``block_matmul``, by a ``BlockMac`` and
by the layer factory :func:`~mptorch.quant.block_gemm_formats`.

Convolutions
------------

The convolutions have operators of their own, ``custom_conv_*``. There are
twelve: one for each kind of mac of the float formats (BinaryK or SuperFP,
split or fused, one format per slot or a palette, and the two cross-family
split macs), and the fixed-point split and fused macs. Each of the ten float
ones also has a twin that rounds the carrier's steps to odd
(``custom_conv_*_odd``). They take the convolution's geometry in place of
the transpose flags. There are no Python functions for them. A
``QConv1d``/``2d``/``3d`` reaches them through
:func:`~mptorch.quant.conv_formats`, which fills the layer's three math
hooks from a mac, as :func:`~mptorch.quant.matmul_formats` does for a matmul
(:doc:`/layers`, "Convolutions"). :doc:`convolutions` describes what they
compute.

The schema tier
---------------

Below the value vocabulary there are twelve flat functions. Each spells out
every argument of one GEMM operator's schema, and none has autograd:

- ``binaryK_matmul`` and ``superfp_matmul``, and their ``_fma`` variants;
- the ``_mixed`` (palette) variants of those four;
- the two cross-family split macs, ``binaryK_superfp_matmul`` (BinaryK
  products, SuperFP sums) and ``superfp_binaryK_matmul``;
- the fixed-point macs, ``fixedpoint_matmul`` and ``fixedpoint_matmul_fma``.

They are for scripts that need to state exactly what the kernel receives.
The vocabulary resolves to them: the test suite asserts that a ``SplitMac``
and the equivalent flat call produce the same resolved kernel arguments.

.. literalinclude:: ../../snippets/gemm_raw_ops.py
   :language: python
   :caption: docs/snippets/gemm_raw_ops.py

.. literalinclude:: ../../snippets/gemm_raw_ops.out
   :language: text
   :caption: output

The ten float functions share their conventions:

- ``mul_*`` and ``acc_*`` give the two formats of a split
  multiply-accumulate. The ``acc_*`` widths and sign default to the
  ``mul_*`` values. (A BinaryK ``acc_bias`` defaults instead to P3109's bias
  for the accumulate widths, and ``acc_prng_bits`` to 0.)
  ``accumulate_quant=False`` leaves the sum unrounded. In the two
  cross-family functions the ``acc_*`` widths are the other family's, so
  they are required, and there is no ``accumulate_quant``.
- ``fma_*`` gives the one format of a fused step, and ``fma_quant=False``
  leaves it unrounded.
- ``trans_a`` and ``trans_b`` transpose the last two dimensions of an
  operand without copying it.
- ``*_prng_bits`` set the stochastic bits per format.
- ``rounding_mode`` is the one rounding mode of the call.
  ``saturation_mode`` (and, for a BinaryK format, ``subnormals_mode``)
  belongs to the multiply or fused format, although it has no ``mul_`` or
  ``fma_`` prefix. The accumulate and outer formats have ``acc_*`` and
  ``outer_*`` versions of it.
- ``carrier`` chooses the arithmetic, and ``carrier_rounding`` how the
  carrier rounds, as on a ``SplitMac``.
- The ``_mixed`` variants take ``prec_idx`` as the third positional
  argument. Their width arguments take sequences, one entry per palette
  format, and the arguments that the palette shares take a scalar.

The six functions without a palette also take ``accumulate_algorithm``, its
``block_size`` and the outer format, spelled ``outer_*`` in the same way as
the accumulate format. A call may therefore run one of four operators:

- the default operator, under ``NAIVE``;
- its ``*_accumulated`` twin, under ``KAHAN``, ``BLOCK`` or ``TREE``;
- its ``*_odd`` twin, under ``NAIVE`` with ``carrier_rounding=RoundMode.RO``;
- its ``*_accumulated_odd`` twin, under the other three algorithms with
  ``carrier_rounding=RoundMode.RO``.

The twins are separate operators, so a ``NAIVE`` call passes none of the
arguments that only the other algorithms use. The ``_mixed`` functions run
their default operator, or its ``*_odd`` twin.

The two fixed-point functions spell their formats as
:class:`~mptorch.FixedPoint` does: ``mul_wl``, ``mul_fl`` and the required
``acc_wl``, ``acc_fl`` (or ``fma_wl``, ``fma_fl``), each with ``*_is_signed``,
``*_symmetric`` and ``*_prng_bits``. They take ``rounding_mode``,
``overflow`` and ``carrier``, and no algorithm or carrier rounding. A
result whose accumulate format is wider than the operands' dtype is
returned in float64 (:doc:`arithmetic`, "Fixed-point macs").

Callers that keep one format across many calls should use the factories or
``qmatmul`` instead. These functions work out the format's defaults again on
every call, which the resolved objects avoid. The convolutions' operators
have no functions at this tier (`Convolutions`_, above), and the block
product's function is :func:`~mptorch.quant.block_matmul` itself.

torch.compile and torch.export
------------------------------

Every operator has a fake (Meta) kernel. It runs the CPU operator's checks
and allocates a result of the shape, dtype and layout the CPU allocates, but
computes nothing and draws no random state. The meta device,
``FakeTensorMode``, :func:`torch.export.export` and :func:`torch.compile`
run it in the operator's place.

A fake kernel accepts symbolic sizes, so a batch, a matrix extent or a
channel count can stay dynamic in a compiled graph or an exported program.
An extent passed to an operator as an integer cannot stay dynamic, and is
specialized to its value instead. Two extents are passed that way: the
spatial size of a convolution's output, and the length of a block format's
packed axis.

.. literalinclude:: ../../snippets/compile_export.py
   :language: python
   :caption: docs/snippets/compile_export.py

.. literalinclude:: ../../snippets/compile_export.out
   :language: text
   :caption: output

What to expect:

- **An exported layer is its forward.** Export inlines the layer's autograd
  function, so the program holds the GEMM operator and no gradient. Run it
  under ``torch.no_grad()``. In grad mode the operator receives a parameter
  that requires a gradient, and refuses it, as it refuses any such tensor:
  an operator's gradient is its layer's to compute.
- **The flat functions and the layers compile into one graph,**
  ``fullgraph=True`` included. Every result an mptorch operator rounds is
  eager's, word for word. What Inductor compiles around the operators is
  its own, as in any model. A layer's bias gradient, for example, is a sum
  over the batch, and can differ from eager's in the last bits, as a plain
  ``nn.Linear``'s does.
- **Recomputed operands are kept under stochastic rounding.** A layer can
  keep a raw operand and quantize it again in backward (``recompute_input``,
  ``recompute_weight``). Under stochastic rounding that replay sets the
  random generators' state, which a graph cannot do, so a compiled layer
  keeps the quantized copy instead. The values are the same, and a compiled
  graph decides what it saves anyway.
- **Format warnings and refusals.** A ``FormatRangeWarning`` is issued while
  the graph is traced, naming the line that called the compiled code. It is
  not issued again on the compiled calls after it. A format the call refuses
  raises its ``ValueError`` when the graph is traced. Under
  ``fullgraph=True``, a refusal inside a layer or ``qmatmul`` (whose
  autograd function Dynamo traces) becomes Dynamo's own error instead, with
  the ``ValueError``'s message in it. A flat function's refusal stays a
  ``ValueError``.
- **Dynamo's notice about the format checks.** Dynamo says, once per format
  check, that it traces through the check's ``functools.lru_cache``. The
  checks are pure, so the notice is safe to filter, as the snippet does.
- **Stochastic rounding.** Every operator that takes a rounding mode is
  tagged ``nondeterministic_seeded``, so a compiled graph never merges two
  identical stochastic calls into one draw. Inductor may still reorder
  independent random operators, and it removes one whose result is unused,
  as it does PyTorch's own random operators. The compiled draws are then
  valid but not eager's. ``torch._inductor.config.fallback_random = True``
  keeps them in program order, and the result is eager's, bit for bit.
- **The checks that need values run when the operator does.** The fake
  kernel cannot read tensor values or addresses. So three checks wait for
  the real operator: a palette's ``prec_idx`` bounds, CUDA's alignment check
  on an in-place quantizer's view, and the refusals of the operators that
  MPS does not run.

Performance notes
-----------------

- **Resolve once.** ``SplitMac``, ``FusedMac``, ``BinaryK`` and ``SuperFP``
  are frozen and hashable, and ``qmatmul`` memoizes the resolution of the
  ``formats`` it is given. Constructing a new ``SplitMac`` per call still
  hits the cache. Constructing a ``QMatmulFormats`` per call does not.
  Keeping a ``QMatmul`` module, or the ``QMatmulFormats`` that a factory
  returned, costs nothing per call beyond the kernel.
- **One launch per batch.** A batched call is one kernel launch (one
  parallel region on the CPU) over the whole batch. On attention-shaped
  operands on an RTX 4060 laptop GPU, it is 1.9-2.6x faster than the loop of
  2D calls it replaces (``dev/benchmarks/timing/benchmark_qmatmul.py``).
- **Stochastic rounding is not free.** Each rounding draws from a
  per-element Philox stream, so a GEMM under a stochastic mode is slower
  than a deterministic one.
- **binary64 is slower.** On an RTX 4060 a float64 GEMM took about 2.5x the
  time of the same product computed in binary32. The ratio ranged from 1.7x
  to 5.3x, the most for a binaryK split step under round-to-nearest-even. On
  a CPU, the carriers' times depend on the rounding mode:

  - Under the directed modes the two are within about 12%.
  - Under round-to-nearest-even binary32 is faster wherever it has a fast
    path that binary64 lacks: 11-14x for binaryK, which runs in vector code
    (next item), and 1.4x for a superfp split step, whose binary32 cast
    rounds with float arithmetic.
  - Under stochastic rounding binary64 takes 1.4-1.7x, since each of its
    roundings draws two random words.

  A float64 model whose formats binary32 can carry gets the binary32 speed
  back as a float32 model. A float32 model with ``carrier=torch.float64``
  pays the binary64 time plus the widening and narrowing copies.
- **On the CPU, binaryK under round-to-nearest-even is the fast case.** The
  CPU runs a GEMM's tiles in vector code (AVX-512 or AVX2 on x86-64, NEON on
  arm64) when all of the following hold:

  - the mac is a ``SplitMac`` with a BinaryK multiply format and a BinaryK
    or ``None`` accumulate format, or a ``FusedMac`` with a BinaryK format,
    with one format per slot;
  - the carrier is binary32, the rounding mode ``RoundMode.RNE`` and the
    accumulate algorithm ``NAIVE``;
  - every format is signed, keeps its subnormals, does not saturate with
    ``SAT_PROPAGATE``, and has a precision from 2 to 23 bits and a range
    well inside binary32's (in practice at most seven exponent bits).

  The convolutions over such a mac run in vector code too. The results are
  the scalar loop's, word for word. On a Ryzen 9 7940HS (AVX-512) the vector
  code is 6-10x faster than the scalar loop on one thread (a split step goes
  from 3.4 to 0.41 ns), and 8-14x faster on sixteen. On an M3 Pro (NEON) it
  is about 4x faster at every thread count (0.65 ns a split step on one
  thread).

  The odd carrier step (``carrier_rounding=RoundMode.RO``) has vector tiles
  for the same cases. With AVX-512 they run at 1.04-1.2x the default tile's
  time, through the instruction set's directed roundings. AVX2 and NEON get
  the same roundings from the processor's rounding mode, at 1.1-1.25x and
  1.3-1.4x.

  Every other mode, format and accumulate algorithm runs the scalar loop,
  palettes included. So does every call under the environment variable
  ``MPTORCH_CPU_SIMD=0``, which is read when mptorch loads.
- **Block products and convolutions cost what their matrix products cost.**
  How much the decode of a block operand adds depends on the arithmetic
  beside it (:doc:`block`, "How the kernel decodes"). At 1024\ :sup:`3` on
  an RTX 4060 laptop GPU, against the flat product on the decoded operands,
  the block product took 1.29-1.30x the time with a fused mac under
  round-to-nearest, and 1.07-1.09x under stochastic rounding. With a split
  mac it took 0.40-0.55x the time, since the flat product also rounds every
  product to a multiply format and the block product does not. A
  convolution's gathered loads add a few percent of address arithmetic
  (:doc:`convolutions`, "What it costs").
- **The kernels are simulators.** They are written for exactness and
  flexibility, not for speed, and are orders of magnitude slower than
  cuBLAS. The Linear layers fold their batch into one GEMM, which is the
  shape the kernels handle best.
