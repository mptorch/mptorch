Quantizers
==========

A quantizer rounds every element of a tensor to a format. It is the simplest
thing MPTorch does, and everything else is built on it. A layer applies
quantizers to its weights and activations, a straight-through estimator wraps
one, and a quantization-aware-training observer sits in front of one. An
elementwise quantizer rounds each element independently:

.. math::

   \tilde{x}_i = Q_{F,\circ}(x_i) \qquad \text{for every element } i,

where :math:`F` is the format and :math:`\circ` the rounding mode. A block
quantizer first chooses a scale for each block of elements from the block's
values, and then rounds each element (`Block quantization and packed
storage`_).

Elementwise quantizers have three spellings, described below from the most
explicit to the most convenient:

- the functions, which take the format's parameters one by one;
- :class:`~mptorch.quant.Quant`, which takes a format object and a rounding
  mode;
- :class:`~mptorch.quant.Quantizer`, a module that also says what happens to
  the gradient.

The functions
-------------

:func:`~mptorch.quant.binaryK_quantize` and
:func:`~mptorch.quant.superfp_quantize` take a tensor and the format's
parameters spelled out, and return a new tensor of the same shape and dtype.

.. literalinclude:: ../snippets/quantizers_functions.py
   :language: python
   :caption: docs/snippets/quantizers_functions.py

.. literalinclude:: ../snippets/quantizers_functions.out
   :language: text
   :caption: output

Some things to know about them:

- ``bias`` is optional for ``binaryK_quantize``, because P3109 defines a
  default: :math:`2^{K-P-1}`, or :math:`2^{K-P}` for an unsigned format. It
  is required for ``superfp_quantize``, because a superfp's bias is part of
  the format's design and no rule derives it (:doc:`concepts`, "SuperFP:
  precision at the top, range below").
- Every floating-point dtype that PyTorch trains in is accepted: float32,
  float64, float16 and bfloat16. The rounding runs in a *carrier*, the binary
  floating-point type its arithmetic uses. A float64 input is rounded
  directly in binary64. The others are rounded on their float32 value, in
  binary32 unless the call names binary64. `float64 and the carrier`_ below
  shows what that changes.
- Every call checks the format first, and raises ``ValueError`` when:

  - the parameters describe no format: a negative width, a binaryK ``P``
    below 1 or above ``K`` (``K + 1`` for an unsigned format), a superfp
    ``normal_binades`` outside ``1 .. 2**exp_bits - 1``, or a bias that puts
    ``1 - bias`` or ``2**exp_bits - 1 - bias`` outside :math:`-2^{30}` to
    :math:`2^{30}`;
  - the carrier cannot hold the format (`float64 and the carrier`_).

- The result is stored back in the input's dtype, whichever carrier rounded
  it, so a float16 or bfloat16 result is rounded once more. The input is
  already a value of that dtype, so only an edge of the format's range can
  miss the dtype's grid, and the call warns when one can. The run shows the
  top of E5M2's layout meeting float16's. :doc:`concepts` gives the rule, and
  the stricter one that a GEMM's result is held to.
- The result is a new tensor, and the input is never modified. The in-place
  twins, ``binaryK_quantize_`` and ``superfp_quantize_``, are the exception
  and have `a section of their own <In place_>`_.
- NaN passes through with its payload. An infinity passes through too,
  except under P3109's ``SatFinite`` in either domain
  (``SaturationMode.SAT_FINITE`` and ``SAT_FINITE_EXTENDED``), which clamps
  it to the largest finite value like any other overflow.
- ``prng_bits`` matters only under the stochastic rounding modes
  (``RoundMode.SR``, ``SR_B`` and ``SR_C``). The random bits are drawn in the
  carrier, below the format's mantissa. The format's mantissa bits and the
  random bits therefore share the carrier's mantissa: 23 bits in binary32
  and 52 in binary64, as the run shows.
- Every function here also takes ``carrier``, described below.
- CPU and CUDA give bit-identical results under every deterministic mode.
  Under stochastic rounding, CUDA draws from its own default generator, so
  ``torch.manual_seed`` reproduces a CUDA run on CUDA. An Apple GPU
  (``"mps"``) draws from the CPU's generator, and its quantizers match the
  CPU's bit for bit in every mode, stochastic rounding included. It has no
  float64 (:ref:`apple-gpu`).

:func:`~mptorch.quant.fixedpoint_quantize` is the function for a
:class:`~mptorch.FixedPoint` format (:ref:`fixed-point`). It takes ``wl``,
``fl``, ``is_signed`` and ``symmetric`` in place of a float's fields. It
takes no ``saturation_mode``, because a fixed-point result always saturates.
The points above apply to it too:

- the dtypes, the carrier and the storage check;
- ``prng_bits``, which shares the carrier's precision (24 bits or 53) with
  the format's magnitude bits;
- the backends.

Under stochastic rounding it is exact: the random bits are added to the
input's word directly below the step (:ref:`fixed-point`).

.. literalinclude:: ../snippets/quantizers_fixedpoint.py
   :language: python
   :caption: docs/snippets/quantizers_fixedpoint.py

.. literalinclude:: ../snippets/quantizers_fixedpoint.out
   :language: text
   :caption: output

The end of the run shows the storage check on a narrower result, and the two
backends side by side:

- A float16 or bfloat16 result is held to the elementwise storage rule, as a
  float format's is: only an end of the range can miss the dtype's grid.
  ``FixedPoint(16, 14)``'s largest value needs 15 bits, so a float16 result
  warns, and an input that saturates to it is stored as 2.0.
  ``FixedPoint(8, 6)``'s largest value fits bfloat16's 8 bits, so that call
  is silent. :ref:`fixed-point` gives the rule.
- CPU and CUDA return the same words in every deterministic mode. This run
  was recorded on a machine with an NVIDIA GPU. Without one, the snippet
  skips that line.

float64 and the carrier
-----------------------

A quantizer rounds in its carrier (:doc:`concepts`, "Carriers"). By default
the carrier is binary64 for a float64 tensor, and binary32 for the other
three dtypes.
Rounding a float64 tensor in binary64 has three consequences:

- **Wider formats.** The format is held to binary64's bounds, so up to 53
  bits of precision and ten exponent bits are simulated exactly. The same
  call on a float32 tensor raises, because binary32 has only 24 bits to round
  in.
- **Rounded once.** A value that float32 cannot hold is rounded on its own
  bits. ``1.0625 + 2**-30`` is above the tie between ``1.0`` and ``1.125`` on
  Binary8p4's grid, and rounds up. Narrowed to float32 it *is* the tie, and
  round-to-nearest-even takes it down.
- **Wider random draws.** Stochastic rounding takes its random bits below the
  format's mantissa in the carrier. ``P - 1 + prng_bits`` may therefore reach
  52 instead of 23, and each element draws two 32-bit words of its random
  stream instead of one.

``carrier=torch.float64`` rounds a float32, float16 or bfloat16 tensor in
binary64. The tensor is widened, rounded under binary64's bounds, and the
result is narrowed back to the tensor's dtype. This is bit for bit the
float64 call on the widened tensor, stored with one rounding.

For an elementwise quantizer, this gives the other dtypes the wider formats
and the wider random draws, but not the single rounding. The tensor's values
are already values of its dtype, and a format that binary32 can hold rounds
them to the same answer in either carrier, as the run checks. The result is
held to what its dtype can store (:doc:`concepts`). For float32, that is 24
bits of precision and binary32's range at the edges.

``carrier=torch.float32`` names binary32. A float64 tensor refuses it,
because the carrier is never narrower than the tensor.
:class:`~mptorch.quant.Quant` takes the same keyword.

.. literalinclude:: ../snippets/quantizers_float64.py
   :language: python
   :caption: docs/snippets/quantizers_float64.py

.. literalinclude:: ../snippets/quantizers_float64.out
   :language: text
   :caption: output

Quant: a format and a rounding mode, as a function
--------------------------------------------------

:class:`~mptorch.quant.Quant` binds a format object and a rounding mode into
a callable. The ``*_quant`` slots of a layer's formats are meant to hold
one. Calling it makes the function call above, with the arguments filled in
from the format, and the run checks that the two give the same bits. It
takes ``carrier`` and ``inplace`` as keywords. Its format may also be a
:class:`~mptorch.FixedPoint` or a block format (`Block quantization and
packed storage`_).

.. literalinclude:: ../snippets/quantizers_quant.py
   :language: python
   :caption: docs/snippets/quantizers_quant.py

.. literalinclude:: ../snippets/quantizers_quant.out
   :language: text
   :caption: output

Formats and Quants are frozen dataclasses, so they compare and hash by value.
This lets a layer resolve its format once, and lets the library memoize on
it. Build them once and keep them, instead of constructing one per call.

In place
--------

:func:`~mptorch.quant.binaryK_quantize_`,
:func:`~mptorch.quant.superfp_quantize_` and
:func:`~mptorch.quant.fixedpoint_quantize_` take the arguments of the
functions above, write the result over the tensor, and return the tensor.
They run the same kernels with the output pointer set to the input's. The
result is therefore the out-of-place one bit for bit, including under
stochastic rounding with the same seed. The only difference is that no
result is allocated. They are meant for a tensor the caller owns, such as a
weight rounded once when a model is loaded, or an activation that nothing
else will read. ``Quant(fmt, rounding, inplace=True)`` is the same operation
as a value.

The out-of-place functions handle some inputs by making a copy. The in-place
ones refuse those inputs, because the kernel would round the copy and leave
the caller's tensor unchanged:

- a tensor that is not contiguous;
- on CUDA, a contiguous view that does not start on a 16-byte boundary
  (``x[1:]``), because the kernel moves 16 bytes at a time and cannot address
  such a view;
- ``carrier=torch.float64`` on a float32, float16 or bfloat16 tensor, which
  rounds a float64 copy and narrows the result. A float64 tensor does round
  in place in binary64, since that is its own carrier.

Each error names the out-of-place operation, which accepts all three.

The in-place functions are not differentiable, like their out-of-place
twins, and raise the same way on a tensor that requires grad. On a tensor
that does not, the write bumps the tensor's version counter, like any
in-place op of PyTorch's own. Autograd therefore notices if a tensor it had
saved is rounded. A layer's ``*_quant`` slots should never hold an in-place
``Quant``: a layer's input belongs to the graph of whatever produced it, and
autograd may have saved it. On an ``"mps"`` tensor the in-place functions
run the out-of-place kernel with the tensor as both its input and its
output.

.. literalinclude:: ../snippets/quantizers_inplace.py
   :language: python
   :caption: docs/snippets/quantizers_inplace.py

.. literalinclude:: ../snippets/quantizers_inplace.out
   :language: text
   :caption: output

The saving is mostly in memory, because the kernel reads and writes the
same bytes either way. On CUDA the two take the same time at every size
measured. On the CPU the in-place call was 2-15% faster on a 16 MiB float32
tensor, whose result the allocator recycles. A result large enough to be
mapped fresh from the system on every call pays for its pages when they are
first written, which can cost more than the rounding: on a 32 MiB float64
tensor and one thread, the out-of-place call took nearly three times as long
(``dev/benchmarks/timing/cpu_quantize_elementwise.py`` compares the two).

Block quantization and packed storage
-------------------------------------

A block format (:ref:`block-formats`) stores a tensor as narrow element
codes, and gives each *block* of consecutive elements along one axis, the
*packed* axis, a shared *scale*. A tensor is quantized one block at a time.
The block's scale is chosen from the block's values by the format's
``scale_rounding`` rule. Each element is then divided by the scale and
rounded to the element format, in the call's rounding mode.

Four functions do this. The three that round take an ``axis`` argument,
which names the packed axis:

- :func:`~mptorch.quant.block_pack` rounds a tensor and returns its codes, as
  a :class:`~mptorch.quant.BlockPacked`.
- :func:`~mptorch.quant.block_unpack` decodes a packed tensor back to its
  shape: each element is its value times its block's scale.
- :func:`~mptorch.quant.block_quantize` does both at once, with no packed
  intermediate, and returns the decoded values in the input's dtype, like an
  elementwise quantizer. It is the *fake quantizer* that a layer's
  ``*_quant`` slot takes.
- :func:`~mptorch.quant.block_quantize_` writes ``block_quantize``'s result
  over its input.

:class:`~mptorch.quant.BlockQuant` is the value spelling of
``block_quantize``, and its ``pack`` method calls ``block_pack``.
:class:`~mptorch.quant.Quant` with a block format is another spelling, along
the last axis.

A :class:`~mptorch.quant.BlockPacked` holds ``data``, the element codes, and
``scales``, the scale codes, both ``uint8``. Each row of ``data`` is a row of
the tensor with its packed axis moved last. Each block in it is stored as a
little-endian bit stream: the :math:`w`-bit code of the block's element
:math:`i` occupies bits :math:`[i\,w, (i+1)\,w)`. Two 4-bit codes therefore
share a byte, the first in the low nibble (the order of
``torch.float4_e2m1fn_x2`` and CUTLASS), and four 6-bit codes fill three
bytes.

A format with a per-tensor scale (NVFP4, or any other format whose scale has
a mantissa) also stores one scale for the whole tensor. By default it is
computed from the tensor, as its largest magnitude over the largest element
value times the format's ``top_scale``. ``top_scale`` is the scale format's
largest value, except under a Four Over Six rule (:doc:`concepts`, "Four Over
Six"). The ``tensor_scale`` argument of ``block_pack``, ``block_quantize``
and ``BlockQuant`` gives a static scale instead.

.. literalinclude:: ../snippets/block_formats.py
   :language: python
   :caption: docs/snippets/block_formats.py

.. literalinclude:: ../snippets/block_formats.out
   :language: text
   :caption: output

A strided input is packed through its strides, so packing a weight along its
first axis copies nothing. Only a tensor of rank 4 or more may need a copy,
to merge its leading dimensions. Under stochastic rounding an element's
random bits are keyed on its index in the packed orientation.
``block_quantize`` and ``block_pack`` therefore draw the same bits from the
same generator state, whatever the thread count.

The block kernels round in the carrier: binary64 for a float64 tensor, or
for a narrower one with ``carrier=torch.float64``, and binary32 otherwise.
A narrower tensor is widened, and a decoded result is narrowed back with a
single rounding. A :class:`~mptorch.quant.BlockPacked` remembers the carrier
it was packed in, and :func:`~mptorch.quant.block_unpack` decodes in that
carrier or a wider one. A format that only binary64 holds, such as one whose
elements have nine exponent bits, is refused in binary32 with an error that
says so. A result narrower than its carrier is held to what its dtype can
store (:doc:`concepts`, "What a narrower tensor can store").

``block_quantize_`` refuses what would make it write a copy. Its rules are
looser than the elementwise ones of `In place`_:

- A tensor whose elements overlap or leave gaps in memory is refused (a
  strided slice such as ``x[:, ::2]``). A permuted contiguous tensor, such as
  a transposed weight, is accepted.
- A tensor of rank 4 or more is refused if its leading dimensions cannot be
  merged as a view once the packed axis is moved last.
- ``carrier=torch.float64`` on a narrower tensor is refused, as for the
  elementwise in-place functions.

:doc:`kernels/block` covers multiplying packed block-format tensors.

.. literalinclude:: ../snippets/block_float64.py
   :language: python
   :caption: docs/snippets/block_float64.py

.. literalinclude:: ../snippets/block_float64.out
   :language: text
   :caption: output

Quantizer: a straight-through estimator
---------------------------------------

Rounding is a step function. Its derivative is zero almost everywhere and
undefined at the steps, so autograd cannot derive a gradient "through" a
quantizer. That gradient is a modelling choice, and the standard choice is
the **straight-through estimator** (STE): in the backward pass, pretend that
the quantizer was the identity. :class:`~mptorch.quant.Quantizer` implements
it, with one refinement: the gradient that passes through can itself be
rounded, to a format of its own:

.. math::

   \text{forward:}\quad y = Q_{F_{\text{fwd}}}(x), \qquad
   \text{backward:}\quad \frac{\partial L}{\partial x} := Q_{F_{\text{bwd}}}\Big(\frac{\partial L}{\partial y}\Big).

Each direction takes one of three things:

- ``None``, which leaves that direction unchanged;
- a format object, which is wrapped in a ``Quant`` that rounds to nearest,
  ties to even;
- any callable ``Tensor -> Tensor`` of your own, such as a ``BlockQuant``.

The two directions are separate on purpose: a quantization-aware forward
pass usually wants a narrow format, and the gradient a wider one.

.. literalinclude:: ../snippets/quantizers_quantizer.py
   :language: python
   :caption: docs/snippets/quantizers_quantizer.py

.. literalinclude:: ../snippets/quantizers_quantizer.out
   :language: text
   :caption: output

A ``Quantizer`` is an ``nn.Module``, so it can sit anywhere in a model:
between two layers, or in a ``*_quant`` slot of a layer's formats. It is
registered with its parent, so any state it carries travels with the model's
``state_dict``.

The rule for gradients
~~~~~~~~~~~~~~~~~~~~~~

The run above also shows what the raw functions do with a tensor that
requires a gradient: they raise. None of MPTorch's operators is
differentiable in autograd's sense. A quantizer has a zero derivative almost
everywhere, and a GEMM that simulates its own arithmetic has no derivative
the framework could infer. Returning a tensor anyway would silently drop its
gradient: ``.grad`` would be ``None`` after ``.backward()``, and the model
would not train. So each raw operator raises, and names the entry point that
does carry a gradient:

- ``Quantizer`` for the quantizers, the block ones included (with a
  ``BlockQuant``);
- ``qmatmul`` or ``QMatmul`` for the matrix products, and for the block
  product ``qmatmul`` with a ``BlockMac``;
- ``QConv1d``, ``QConv2d`` or ``QConv3d`` with ``conv_formats`` for the
  convolution passes.

The check is *gradient mode and* ``requires_grad``, so the layers are
unaffected. A ``torch.autograd.Function``'s forward runs with gradient mode
off, so ``QLinear`` and the other layers can call the raw operators on
tensors that require grad. Under ``torch.no_grad()``, or on a detached
tensor, the raw functions simply run.
