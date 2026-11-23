Layers
======

The layers combine the two kinds of rounding. A *quantizer* rounds each
*signal*, that is, each tensor that flows through the layer: its input,
weight and bias, and the gradients of the backward pass. An *arithmetic*
says how each matrix product rounds its products and partial sums. Each
layer is a subclass of its ``torch.nn`` counterpart (``QLinear`` of
``nn.Linear``, ``QConv2d`` of ``nn.Conv2d``), and its ``forward`` calls a
custom ``torch.autograd.Function`` that applies the rounding in both passes.
A model built from these layers trains with the ordinary optimizer,
scheduler and loss.

What a quantized layer computes
-------------------------------

Take a linear layer, :math:`y = xW^{\top} + b`. With
:math:`G = \partial L / \partial y` arriving from above, the three gradients
are

.. math::

   \frac{\partial L}{\partial x} = G\,W, \qquad
   \frac{\partial L}{\partial W} = G^{\top} x, \qquad
   \frac{\partial L}{\partial b} = \sum_{\text{batch}} G .

A :class:`~mptorch.quant.QLinear` computes the same four quantities, with a
quantizer in front of each signal and a separate arithmetic for each
product:

.. math::

   \begin{aligned}
   y &= \big[\, Q_{\text{in}}(x)\; Q_{\text{w}}(W)^{\top} \,\big]_{\text{fwd}} + Q_{\text{b}}(b) \\[4pt]
   \frac{\partial L}{\partial x} &= \big[\, Q_{\text{igrad}}(G)\; Q_{\text{w}}(W) \,\big]_{\text{igrad}} \\[4pt]
   \frac{\partial L}{\partial W} &= \big[\, Q_{\text{wgrad}}(G)^{\top}\; Q_{\text{in}}(x) \,\big]_{\text{wgrad}} \\[4pt]
   \frac{\partial L}{\partial b} &= \sum_{\text{batch}} Q_{\text{bgrad}}(G)
   \end{aligned}

Here :math:`[\cdot]_{\text{fwd}}`, :math:`[\cdot]_{\text{igrad}}` and
:math:`[\cdot]_{\text{wgrad}}` are the three matrix products, each in the
arithmetic of its hook. Two points are worth noting:

- The backward pass uses the same quantized weight and input as the forward
  pass, :math:`Q_{\text{w}}(W)` and :math:`Q_{\text{in}}(x)`, not
  :math:`W` and :math:`x`. The layer either keeps them from the forward
  pass, or keeps :math:`W` and :math:`x` and quantizes them again, with the
  same result to the bit (see `What a layer keeps for backward`_).
- The incoming gradient is quantized *separately* for the two products it
  feeds, because the input gradient and the weight gradient often need
  different formats.

QAffineFormats
--------------

:class:`~mptorch.quant.QAffineFormats` is the configuration object that a
layer takes. It has nine optional slots. Each defaults to ``None``, which
means plain PyTorch for that step.

====================  ==============================================================
slot                  role
====================  ==============================================================
``input_quant``       :math:`Q_{\text{in}}`, applied to the layer's input
``weight_quant``      :math:`Q_{\text{w}}`, applied to the weight
``bias_quant``        :math:`Q_{\text{b}}`, applied to the bias
``igrad_quant``       :math:`Q_{\text{igrad}}`, applied to :math:`G` on the input-gradient path
``wgrad_quant``       :math:`Q_{\text{wgrad}}`, applied to :math:`G` on the weight-gradient path
``bgrad_quant``       :math:`Q_{\text{bgrad}}`, applied to :math:`G` on the bias-gradient path
``fwd_math``          the forward product; default ``F.linear`` / ``F.conv*d``
``bwd_igrad_math``    the input-gradient product; default a ``matmul`` / ``conv*d_input``
``bwd_wgrad_math``    the weight-gradient product; default a ``matmul`` / ``conv*d_weight``
====================  ==============================================================

A quantizer slot takes any callable ``Tensor -> Tensor``: a
:class:`~mptorch.quant.Quant`, a :class:`~mptorch.quant.Quantizer`, or a
function of your own, such as a scaled quantizer or an observer.
``QAffineFormats`` is itself an ``nn.Module``. A quantizer that is a module
is therefore registered in the model, and its state appears in the model's
``state_dict``. Two more fields, ``recompute_input`` and
``recompute_weight``, choose what the layer keeps for its backward pass (see
`What a layer keeps for backward`_).

The run below builds a layer with a quantizer on every signal: Binary8p4 on
the input, the weight and the bias, and Binary8p3 on the gradients. It then
recomputes the four equations by hand on the same tensors. The results agree
to the bit.

.. literalinclude:: ../snippets/layers_qlinear.py
   :language: python
   :caption: docs/snippets/layers_qlinear.py

.. literalinclude:: ../snippets/layers_qlinear.out
   :language: text
   :caption: output

Custom arithmetic in the matrix products
----------------------------------------

With the ``*_math`` hooks unset, the three products are PyTorch's own. They
compute in the layer's dtype, however narrowly the operands were rounded. To
simulate the arithmetic *inside* the products, a factory sets the three
hooks to MPTorch's matrix-product kernels (:doc:`kernels/index`), in the
arithmetic you describe:

- :func:`~mptorch.quant.binaryK_gemm_formats`: a ``SplitMac`` (a
  multiply-accumulate step whose product and sum are rounded separately) in
  BinaryK formats. The products are rounded to the format of the ``mul_*``
  arguments, and the running sums to the format of the ``acc_*`` arguments.
  ``accumulate_quant=False`` leaves the sums unrounded, in the carrier.
- :func:`~mptorch.quant.binaryK_gemm_formats_fma`: a ``FusedMac``, one fused
  multiply-add per step, rounded to the BinaryK format of the ``fma_*``
  arguments.
- :func:`~mptorch.quant.superfp_gemm_formats` and
  :func:`~mptorch.quant.superfp_gemm_formats_fma`: the same two for SuperFP
  formats.
- :func:`~mptorch.quant.binaryK_superfp_gemm_formats` and
  :func:`~mptorch.quant.superfp_binaryK_gemm_formats`: a ``SplitMac`` whose
  products and sums are in different families. The first rounds the
  products to a BinaryK format and the sums to a SuperFP one, and the second
  does the reverse.

Each factory also takes:

- ``rounding_mode``, one rounding mode for every format of the product;
- ``accumulate_algorithm``, the order in which the terms are summed, with
  its ``block_size`` and ``outer_*`` format (:doc:`kernels/arithmetic`);
- ``carrier`` (see `float64 models and the carrier`_);
- ``carrier_rounding``, which can make the carrier round each step to odd
  (:doc:`kernels/arithmetic`, "What the carrier rounds").

Each factory returns a ``QAffineFormats`` with *only* the math hooks set.
You assign the operand quantizers afterwards, so the two choices stay
independent. The hooks follow the layer's call contract:
``fwd_math(q_input, q_weight, q_bias)``,
``bwd_igrad_math(q_grad, q_weight)`` and ``bwd_wgrad_math(q_grad, q_input)``.
They handle the transposes. They also fold any leading batch dimensions of
the input into the rows of one matrix product, and restore them in the
result. The formats are resolved once, when the factory runs, not on every
forward pass.

.. literalinclude:: ../snippets/layers_gemm_formats.py
   :language: python
   :caption: docs/snippets/layers_gemm_formats.py

.. literalinclude:: ../snippets/layers_gemm_formats.out
   :language: text
   :caption: output

The last three lines run the same weights and input through the three
other factories, without operand quantizers. Each line gives the cosine
similarity of that layer's output to the output of the split BinaryK layer
above. The three arithmetics are a fused Binary8p4 step with stochastic
rounding, then the split and the fused SuperFP arithmetic. This SuperFP
format keeps a full mantissa only in its top two binades, from 128 to 512.
Below 128 its values are powers of two. This layer's products and sums are
far smaller than 128, so their error is large.

Convolutions
------------

:class:`~mptorch.quant.QConv1d`, :class:`~mptorch.quant.QConv2d` and
:class:`~mptorch.quant.QConv3d` take the same ``QAffineFormats`` and apply
its quantizers to the same signals. The equations are the convolution
versions of the linear ones: :math:`G^{\top} x` becomes the weight-gradient
convolution, and :math:`GW` the input-gradient (transposed) convolution.
The math hooks receive the convolution's geometry as keyword arguments:
``stride``, ``padding``, ``dilation``, ``groups`` and ``nd`` (the number of
spatial dimensions). The input-gradient hook also receives ``input_size``,
and the weight-gradient hook ``weight_size``. With the hooks unset, the
convolution is PyTorch's own, on the quantized operands.

.. literalinclude:: ../snippets/layers_conv.py
   :language: python
   :caption: docs/snippets/layers_conv.py

.. literalinclude:: ../snippets/layers_conv.out
   :language: text
   :caption: output

Custom arithmetic in the convolutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~mptorch.quant.conv_formats` sets the three hooks to MPTorch's
convolution kernels, in the arithmetic of a *mac*, the value that describes
one multiply-accumulate step. A mac is a ``SplitMac``, a ``FusedMac``, or a
single format ``fmt``, which means ``SplitMac(fmt, fmt)``. The hooks run the
convolution's three passes: the forward pass, the input gradient and the
weight gradient. ``conv_formats`` is the convolution counterpart of
:func:`~mptorch.quant.matmul_formats`.

Each pass gives, bit for bit, the matrix product of that mac over the
operands that ``F.unfold`` would build, zeros included. But nothing is
unfolded. The kernel reads each operand element directly from the
convolution's tensors, so a pass allocates only its result. By comparison,
the unfolded input of a 3 x 3 convolution with stride 1 is nine times the
size of the input.

Every format, rounding mode and accumulate algorithm of
:doc:`kernels/arithmetic` applies, with any stride, padding (``"same"``
included), dilation and ``groups``. A palette mac, which picks a format for
each output element, takes one map per pass: ``prec_idx`` for the forward
pass, ``igrad_prec_idx`` for the input gradient and ``wgrad_prec_idx`` for
the weight gradient. Each map is shaped like its pass's result with the
spatial dimensions flattened, so ``[Cout, 1]`` gives one format per output
channel. As with the matmul factories, you assign the operand quantizers
afterwards.

:doc:`kernels/convolutions` describes what the three passes compute: which
dot product each one is, in which order its terms are summed, how the input
gradient of a strided convolution skips the zeros that a transposed
convolution would insert, and what each pass costs.

.. literalinclude:: ../snippets/layers_conv_gemm.py
   :language: python
   :caption: docs/snippets/layers_conv_gemm.py

.. literalinclude:: ../snippets/layers_conv_gemm.out
   :language: text
   :caption: output

Block formats
-------------

:func:`~mptorch.quant.block_gemm_formats` builds a ``QAffineFormats`` for a
:class:`~mptorch.quant.QLinear` whose three matrix products are block
products: products of tensors stored in block formats. Each hook *packs*
(encodes in a block format) the two tensors it multiplies, along the
dimension that the product sums over. A block's scale can be factored out
of the sum only when the block runs along that dimension
(:doc:`kernels/block`, "Why the blocks run along K"). Each tensor is packed
in the format of its role: the input's, the weight's, or the output
gradient's, which defaults to the input's.

There are two recipes for the weight:

* **1D weight blocks.** Leave ``weight_quant`` unset. The forward pass packs
  the weight along its input features, and the input gradient packs it along
  its output features. Both start from the float weight, as MX training
  recipes do.
* **Square weight tiles**, such as NVFP4 with 16 x 16 tiles,
  ``w16 = dataclasses.replace(NVFP4, block_rows=16)``. Set
  ``weight_quant = BlockQuant(w16, axis=1).pack``. The weight is packed
  once, and the input gradient reads that packing transposed, so the forward
  and backward passes see the same quantized weight.

  The packing is cached, keyed on the weight, its version counter, its
  device and its dtype. It is therefore redone once per optimizer step,
  which writes the weight in place, and after the layer is moved or
  converted (``.to()``, ``.half()``).

  A weight packed in a 1D or non-square format cannot serve the input
  gradient: it raises there, with a message that names the square-tile
  spelling. In inference there is no backward pass, so a 1D packed weight
  is fine.

  The tile may use any scale rule, including a Four Over Six rule, which
  tries two scales for each tile and keeps the one with the smaller error
  (``dataclasses.replace(w16, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MSE)``).
  Each tile then makes that choice once, for both passes.

.. literalinclude:: ../snippets/block_layers.py
   :language: python
   :caption: docs/snippets/block_layers.py

.. literalinclude:: ../snippets/block_layers.out
   :language: text
   :caption: output

Packing the weight once does not save memory by itself. The layer's
parameter stays the float weight, and the packing is a second copy beside
it, held by the ``BlockQuant``: 0.504 bytes per element for NVFP4's 16 x 16
tiles and 1.03 for MXFP8, next to float32's 4. In training, the float weight
that a 1D layer keeps for its backward pass is the parameter itself, while
the 16 x 16 packing is an extra allocation, kept alive from the forward pass
to the input gradient. A ``BlockQuant`` caches the packing of one weight
only, so give each layer its own. What packing once buys is time and
consistency: the weight is quantized once, and every product reads the same
codes. Each product is computed in the carrier and stored in the layer's
dtype. To drop the float weight, freeze the layer (next section).

``tensor_scale_epilogue=True`` applies the operands' per-tensor scales to
each product's result, instead of to each element as it is decoded. That is
what NVIDIA's NVFP4 matrix-multiplication kernels do (:doc:`kernels/block`,
"Tensor scales: in the loop or after it").

Freezing a layer to its codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For inference, or to train other layers around a fixed one,
:meth:`~mptorch.quant.QLinear.freeze_packed_` replaces a layer's float weight
by its block codes, in place. :func:`~mptorch.quant.freeze_packed_` does the
same for every block-format ``QLinear`` of a model, one layer at a time, and
returns how many layers it froze.

A frozen layer becomes a :class:`~mptorch.quant.QPackedLinear`, which is
still a ``QLinear``:

- its ``weight`` is ``None``;
- its codes and scales are two ``uint8`` buffers, ``weight_codes`` and
  ``weight_scales``. They follow ``.to()``, and stay ``uint8`` under
  ``.half()``.

On a 4096 x 4096 weight the buffers take 8.06 MiB for NVFP4's 16 x 16 tiles,
9.00 MiB for its 1 x 16 blocks and 16.5 MiB for MXFP8. Before the freeze the
layer held 64 MiB of float32, and the codes besides.

The stored codes are the ones the layer's forward pass multiplies, so the
frozen layer returns what it returned before, word for word. They depend on
``weight_quant``:

- unset: the weight packed in the weight format of ``block_gemm_formats``;
- ``BlockQuant(fmt, axis=1).pack``: that quantizer's packing, which is the
  cached one from the last forward pass if the weight has not changed since;
- any other quantizer: its result, packed in the weight format.

.. literalinclude:: ../snippets/block_freeze.py
   :language: python
   :caption: docs/snippets/block_freeze.py

.. literalinclude:: ../snippets/block_freeze.out
   :language: text
   :caption: output

* **Gradients.** A square-tiled weight still serves the input gradient, so
  the layers before a frozen one (closer to the input) can train. A 1D
  weight raises there: a 1D format quantizes the weight differently along
  the other axis, and there is no float weight left to pack again. A frozen
  layer has no weight gradient, and its bias stays a float parameter.
* **Memory.** The float weight is freed once nothing else refers to it. So
  freeze before you build an optimizer or any copy of the parameters, which
  would keep it alive. A model that does not fit on its device in float can
  be frozen on the CPU and then moved. On CUDA the freed memory returns to
  PyTorch's caching allocator, and ``torch.cuda.empty_cache()`` hands it back
  to the driver.
* **Stochastic rounding.** The frozen codes are a single draw. With the
  packed-once recipe they are the packing that the last forward pass used,
  random draws included. A layer that packs in each hook draws again in
  every forward pass, but once it is frozen it reuses the codes drawn at the
  freeze.
* **Checkpoints.** The format is not stored in the checkpoint: like every
  hook of the layer's formats, it is part of the code that builds the model.
  The ``state_dict`` holds the codes, the bias and an ``_extra_state`` that
  records the packing's format, shape, tensor scale, dtype and carrier. To
  load a checkpoint, build the model as it was built, freeze it with
  ``pack=False`` (zeroed codes, nothing packed), and then load. A model
  built on the meta device allocates nothing until
  ``load_state_dict(..., assign=True)``, so its float weights never exist.
  A checkpoint frozen in another format or shape raises an error that names
  both. A float checkpoint does not load into a frozen model, nor a frozen
  checkpoint into a float model.
* **Code that reads** ``layer.weight``, such as a model summary or an
  initializer, finds ``None``, and ``reset_parameters`` raises.
  :meth:`~mptorch.quant.QPackedLinear.unfreeze_packed_` turns the layer back
  into a ``QLinear`` whose weight is the decoded codes: the quantized weight,
  not the original one.

Activations between layers
--------------------------

A layer's ``input_quant`` rounds what enters the layer. In the backward
pass, the layer treats that rounding as the identity: the gradient that
reaches the input is what ``bwd_igrad_math`` computes. To round a tensor
*between* layers, such as an activation, the logits, or the output of a
block that is not a layer, use a :class:`~mptorch.quant.Quantizer`. It is a
module, and it also says what its gradient is rounded to (see
:doc:`quantizers`). The last lines of the convolution run above show one in
a ``Sequential``.

What a layer keeps for backward
-------------------------------

The backward pass needs :math:`Q_{\text{in}}(x)` again for the weight
gradient, and :math:`Q_{\text{w}}(W)` for the input gradient. For each, a
layer can do one of two things:

- keep a quantized copy from the forward pass, a tensor the size of the
  operand;
- keep the raw operand, and quantize it again in the backward pass.

The second way costs one more quantization per operand per step, and saves
the copy. The flags ``recompute_input`` and ``recompute_weight`` of
:class:`~mptorch.quant.QAffineFormats` choose between the two, as do
``recompute_a`` and ``recompute_b`` of
:class:`~mptorch.quant.QMatmulFormats` for a
:class:`~mptorch.quant.QMatmul`. ``True`` quantizes again, and ``False``
keeps the copy. Each flag defaults to ``None``, which means the layer's own
default:

====================  ======================  ======================
layer                 input (``a``)           weight (``b``)
====================  ======================  ======================
``QConv1d/2d/3d``     quantized again         quantized again
``QLinear``           copy kept               quantized again
``QMatmul``           copy kept               copy kept
====================  ======================  ======================

The gradients are the same either way, to the bit. Under stochastic rounding
the second quantization sees the random generators in the state the first
one saw, and then leaves them as it found them.

Quantizing again saves memory only where the raw operand stays alive anyway.
A weight always does, since it is the layer's parameter. An input does when
the operation that produced it keeps its output for its own backward pass.
ReLU, sigmoid, tanh and softmax do, and so does the data batch, which the
training loop holds. GELU, SiLU, ReLU6, LayerNorm, BatchNorm and max pooling
keep their *input* instead, and an addition keeps nothing. After those, the
raw input would be alive only for this layer, so quantizing it again holds
the same memory and adds work.

That is why the defaults differ:

- A convolution's input is usually a ReLU output or the data.
- A Linear layer's input is usually the output of a LayerNorm, or of a GELU
  or SiLU.
- A matmul's operands are activations of any origin.

Set ``recompute_input=True`` for a ReLU MLP, ``recompute_input=False`` for a
convolution after a GELU, and ``recompute_a=True`` for the product of
attention's softmax probabilities with :math:`V`.

The cost is one elementwise pass per recomputed operand. Measured on an RTX
4060, it was within 1% of the training step when the products run on
MPTorch's kernels, and up to 4.5% with PyTorch's own products. The largest
cost, 13%, was for the weights of 4096-wide Linear layers at 256 tokens with
PyTorch's products, which is also the case where recomputing saved the most
memory (from 148 to 20 MiB for three layers).

Four cases keep the copy, even where the table says that the operand is
quantized again:

- **A quantizer of your own.** Under a ``None`` flag, only these quantizers
  are run twice: a :class:`~mptorch.quant.Quant` that is not in place, a
  :class:`~mptorch.quant.ScaledQuant`, a
  :class:`~mptorch.quant.BlockQuant` called as a function (which returns
  the decoded values, not the packing), and a
  :class:`~mptorch.quant.Quantizer` over any of them. A callable with state, such
  as an observer that records statistics for quantization-aware training,
  would update that state a second time in the backward pass, and could
  round differently there. Setting the flag to ``True`` runs any callable
  again. That is your promise that it depends only on its input and the
  random generators.
- **A result that is not a tensor.** A packed weight (``BlockQuant.pack``)
  is compact already, and the ``BlockQuant``'s cache holds it anyway.
- **Under** ``torch.compile``, **a quantizer that may draw random bits**: a
  ``Quant`` or ``BlockQuant`` with a stochastic rounding mode, or a callable
  of your own whose flag is ``True``. Quantizing again would need to read
  and set the random generators' state, which ``torch.compile`` cannot
  trace. The copy holds the same values.
- **A flag set to** ``False``.

An operand that is kept raw is checked by autograd, as ``nn.Conv2d`` checks
its own saved input. Writing to it in place between the forward and backward
passes raises "modified by an inplace operation". With the flag set to
``False``, the layer keeps its own copy and the write does no harm.

The run below counts what four convolution layers keep for the backward
pass, in units of one activation, beyond what the parameters and the input
already hold: four for ``nn.Conv2d``, eight with the copies kept, and four
with the defaults. It then shows what a Linear layer keeps, by default and
with a quantizer of its own.

.. literalinclude:: ../snippets/layers_recompute.py
   :language: python
   :caption: docs/snippets/layers_recompute.py

.. literalinclude:: ../snippets/layers_recompute.out
   :language: text
   :caption: output

float64 models and the carrier
------------------------------

A layer computes in the *carrier* of its tensors, the IEEE 754 format in
which each operation is computed before its result is rounded
(:doc:`concepts`). By default, a float32, float16 or bfloat16 layer rounds
and multiplies in binary32, and a float64 layer in binary64. This applies
to every quantizer and every product. The layer's dtype decides the carrier,
unless a quantizer or a factory names one with ``carrier=``. So
``model.double()`` moves a whole model into binary64, and ``model.float()``
moves it back. On every call, each hook checks its formats against the
carrier it meets.

A float64 model can use formats that a float32 one cannot: up to 53 bits of
precision and ten exponent bits. That is what you need to study arithmetic
*wider* than binary32, or to compute an exact reference.

A float32 model can also get binary64's arithmetic without changing its
dtype. Give ``carrier=torch.float64`` to the factory and to each
:class:`~mptorch.quant.Quant`. Each hook then widens its operands, computes
in binary64 and narrows its result back. The model's signals stay float32,
so their formats must still fit what float32 can store.

The other direction is not a carrier choice. A float64 model computes the
float32 way by becoming a float32 model. On a GPU that is also the faster
arithmetic for the matrix products (:doc:`kernels/calling`, "Performance
notes").

The convolutions of :func:`~mptorch.quant.conv_formats` and the block
products follow the same rule. A float64 model computes them in binary64,
and its block operands are packed, decoded and multiplied there. A float32
model does so too when its mac (or :func:`~mptorch.quant.block_gemm_formats`)
names ``carrier=torch.float64``.

The run has three parts:

1. A float64 layer and a float64 convolution in an arithmetic that only
   binary64 can hold: 30-bit products summed in a 40-bit accumulator. The
   layer is compared with a float64 ``nn.Linear``, and the convolution with a
   float64 ``nn.Conv2d``.
2. A float32 layer given the same arithmetic. It raises in both carriers:
   binary32 cannot round to 30 bits of precision, and a float32 result
   cannot store a 40-bit sum. With a 24-bit accumulator instead, binary64
   gives it the 30-bit products.
3. A float32 layer in two more arithmetics, once in each carrier:

   - an FP8 recipe. Its operands are rounded to 8 bits before any product,
     so both carriers compute every intermediate exactly and agree to the
     bit;
   - 22-bit products summed in a 23-bit accumulator, on unquantized
     operands. Binary32 keeps 24 bits of each product and sum before the
     formats round them, and binary64 keeps 53, so the two carriers differ.

.. literalinclude:: ../snippets/layers_float64.py
   :language: python
   :caption: docs/snippets/layers_float64.py

.. literalinclude:: ../snippets/layers_float64.out
   :language: text
   :caption: output

Putting a model together
------------------------

A mixed-precision model is a set of decisions per layer:

0. Choose the **carrier**. By default it is the model's dtype: float32 for
   the formats that binary32 can carry, and float64 for wider formats or an
   exact reference. Use ``carrier=torch.float64`` where a float32 (or
   half-precision) model should compute in binary64 and keep its dtype.
1. Choose the format of each **signal**: a ``Quant`` (or a ``Quantizer``, or
   a :class:`~mptorch.quant.ScaledQuant`) for ``input_quant``, ``weight_quant``,
   ``bias_quant`` and the three gradient paths.
2. Choose the **arithmetic** of each matrix product and convolution: a
   ``*_gemm_formats`` factory, :func:`~mptorch.quant.conv_formats` or
   :func:`~mptorch.quant.block_gemm_formats` for the ``*_math`` hooks
   (:doc:`kernels/index`). Or leave them as PyTorch's own.
3. Share one ``QAffineFormats`` across layers that should behave the same,
   or give each layer its own.
4. Round anything else with a ``Quantizer``.

The :doc:`tutorial` goes through these steps for the 8-bit formats, training
a network with each combination and comparing them.

Converting a model
------------------

A model that already exists need not be rebuilt from mptorch's layers. A
*recipe* says what each part of it computes in, and
:func:`~mptorch.quant.convert` converts a copy of the model, or
:func:`~mptorch.quant.convert_` the model itself, in place.

- :class:`~mptorch.quant.GemmFormats` holds one layer's formats: the
  operand and gradient quantizers, a rounding of the output, and one mac
  for each pass (the forward product, the input gradient and the weight
  gradient), any of them ``None`` for PyTorch's own arithmetic.
  ``GemmFormats.uniform(mac, quant)`` fills them all.
- :class:`~mptorch.quant.Recipe` gives formats to the *kinds* ``linear``,
  ``conv`` and ``matmul`` (``gemm`` names the three), and
  ``override(where, ...)`` changes them for the modules ``where`` chooses: a
  glob over their names (``"encoder.*"``), a type, or a predicate. The last
  rule that matches wins, and ``None`` keeps a module PyTorch's.
- ``mptorch.quant.presets`` holds three recipes: ``uniform(fmt)``,
  ``fp8()`` (E4M3 operands and E5M2 gradients, each scaled per tensor) and
  ``mx()`` (OCP MX block formats).

``recipe.plan(model)`` shows what a conversion would do. Each ``nn.Linear``
and ``nn.Conv1d/2d/3d`` that the recipe gives formats to becomes a
:class:`~mptorch.quant.QLinear` or a ``QConv*d``: the same object with its
class changed, so its parameters, buffers, hooks and ``state_dict`` keys stay
as they were, and an optimizer built before the conversion still trains it.
Only those exact classes convert. A subclass, a parametrized module among
them, is listed and kept, and ``strict=True`` raises instead.
:func:`~mptorch.quant.unconvert_` turns the layers back, and
:func:`~mptorch.quant.register` adds a class of your own.
:func:`~mptorch.quant.profile` runs one forward pass and counts each layer's
multiply-accumulates by the arithmetic it ran in, priced by a cost per MAC.

.. literalinclude:: ../snippets/layers_convert.py
   :language: python
   :caption: docs/snippets/layers_convert.py

.. literalinclude:: ../snippets/layers_convert.out
   :language: text
   :caption: output

A converted model is saved and loaded through its ``state_dict``, which is
the float model's: the layers' formats hold closures, which the recipe
rebuilds. Only modules convert, so a product that a model's code computes
with a function, such as a ``torch.matmul`` in an attention block, keeps
PyTorch's arithmetic, and ``profile`` does not count it.
