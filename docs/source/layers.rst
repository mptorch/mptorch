Layers
======

The layers combine the two kinds of rounding: a quantizer for each signal,
and an arithmetic for each matrix product. Each layer is a thin subclass of
its ``torch.nn`` counterpart, and its ``forward`` delegates to a custom
``torch.autograd.Function``. A model built from them trains with the
ordinary optimizer, scheduler and loss.

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

- The backward pass reuses the quantized weight and input from the forward
  pass. The layer saves :math:`Q_{\text{w}}(W)` and :math:`Q_{\text{in}}(x)`,
  not :math:`W` and :math:`x`.
- The incoming gradient is quantized *separately* for the two paths it
  feeds, because the input gradient and the weight gradient often need
  different formats.

QAffineFormats
--------------

:class:`~mptorch.quant.QAffineFormats` is the configuration object that a
layer takes. It has nine optional slots, and each defaults to "plain
PyTorch".

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
function of your own (a scaled quantizer, an observer). ``QAffineFormats``
is itself an ``nn.Module``, so a quantizer that is a module is registered in
the model's ``state_dict``. Two more fields, ``recompute_input`` and
``recompute_weight``, choose what the layer keeps for its backward pass
(below, "What a layer keeps for backward").

The run below builds a layer with a quantizer on every signal: Binary8p4 in
the forward direction and Binary8p3 on the gradients. It then recomputes the
four equations by hand on the same tensors. The results agree to the bit.

.. literalinclude:: ../snippets/layers_qlinear.py
   :language: python
   :caption: docs/snippets/layers_qlinear.py

.. literalinclude:: ../snippets/layers_qlinear.out
   :language: text
   :caption: output

Custom arithmetic in the matrix products
----------------------------------------

With the ``*_math`` hooks unset, the three products are PyTorch's own, in the
layer's dtype, however narrowly the operands were rounded. To simulate the
arithmetic *inside* the products, a factory fills the three hooks with
products of the core routine (:doc:`kernels/index`):

- :func:`~mptorch.quant.binaryK_gemm_formats`: a ``SplitMac`` in BinaryK
  formats. It takes ``mul_*`` and ``acc_*`` arguments, and
  ``accumulate_quant=False`` for an unrounded sum.
- :func:`~mptorch.quant.binaryK_gemm_formats_fma`: a ``FusedMac`` in a
  BinaryK format. It takes ``fma_*`` arguments.
- :func:`~mptorch.quant.superfp_gemm_formats` and
  :func:`~mptorch.quant.superfp_gemm_formats_fma`: the same for SuperFP.
- :func:`~mptorch.quant.binaryK_superfp_gemm_formats` and
  :func:`~mptorch.quant.superfp_binaryK_gemm_formats`: a ``SplitMac`` whose
  products are of one family and whose sums are of the other.

Each returns a ``QAffineFormats`` with *only* the math hooks set. Operand
quantizers are assigned afterwards, so the two choices stay independent. The
hooks follow the layer's call contract:
``fwd_math(q_input, q_weight, q_bias)``,
``bwd_igrad_math(q_grad, q_weight)`` and ``bwd_wgrad_math(q_grad, q_input)``.
They handle the transposes, and they fold any leading batch dimensions of
the input into one GEMM. The format is resolved once, when the factory runs,
not on every forward pass. Each factory also takes ``carrier`` (`float64
models and the carrier`_).

.. literalinclude:: ../snippets/layers_gemm_formats.py
   :language: python
   :caption: docs/snippets/layers_gemm_formats.py

.. literalinclude:: ../snippets/layers_gemm_formats.out
   :language: text
   :caption: output

The last three lines show the same weights under three other arithmetics,
measured against the split BinaryK layer: a fused Binary8p4 step with
stochastic rounding, and the two SuperFP cores. The SuperFP format here has
two binades of mantissa, which fits these activations poorly, and the error
shows it.

Convolutions
------------

:class:`~mptorch.quant.QConv1d`, :class:`~mptorch.quant.QConv2d` and
:class:`~mptorch.quant.QConv3d` take the same ``QAffineFormats`` and apply
its quantizers to the same signals. The equations are the convolution
versions of the linear ones: :math:`G^{\top} x` becomes the weight-gradient
convolution, and :math:`GW` the input-gradient (transposed) convolution.
Their math hooks receive the convolution's geometry as keyword arguments
(``stride``, ``padding``, ``dilation``, ``groups``, ``nd``). With the hooks
unset, the convolution is PyTorch's own on the quantized operands.

.. literalinclude:: ../snippets/layers_conv.py
   :language: python
   :caption: docs/snippets/layers_conv.py

.. literalinclude:: ../snippets/layers_conv.out
   :language: text
   :caption: output

Custom arithmetic in the convolutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~mptorch.quant.conv_formats` fills the three hooks with the
convolution's passes: the forward pass, the input gradient and the weight
gradient. Each runs as products of the core routine in the arithmetic of a
mac, as :func:`~mptorch.quant.matmul_formats` does for a matmul.

Each pass is bit for bit the product of that mac over the operands that
``F.unfold`` would build, zeros included. But nothing is unfolded. The kernel
reads each operand element from the convolution's tensors when it loads its
tile, so a pass allocates only its result. By comparison, the unfolded input
of a 3x3 convolution is nine times the size of the input.

Every format, rounding mode and accumulate algorithm of
:doc:`kernels/arithmetic` applies, with any stride, padding (``"same"``
included), dilation and ``groups``. A palette mac takes one ``prec_idx`` map
per pass, each shaped like that pass's result with the spatial dimensions
flattened (``[Cout, 1]`` is one format per output channel). As with the
matmul factories, the operand quantizers are assigned afterwards.

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

:func:`~mptorch.quant.block_gemm_formats` builds a ``QAffineFormats`` whose
three matrix products are block products, for
:class:`~mptorch.quant.QLinear`. Each hook packs what it multiplies along the
dimension it sums over, because a block's scales can be factored out of the
sum only when the blocks run along that dimension (:doc:`kernels/block`,
"Why the blocks run along K"). It uses the format of the tensor's role: the
input's, the weight's, or the output gradient's (which defaults to the
input's). There are two recipes for the weight:

* **1D weight blocks.** Leave ``weight_quant`` unset. The forward pass packs
  the weight along its input features, and the input gradient packs it along
  its output features. Both start from the float weight, as MX training
  recipes do.
* **Square weight tiles** (``dataclasses.replace(NVFP4, block_rows=16)``). Set
  ``weight_quant = BlockQuant(w16, axis=1).pack``. The weight is packed once,
  and the input gradient reads that packing transposed, so the forward and
  backward passes see the same quantized weight. The packing is memoized on
  the weight, its version counter, its device and its dtype, so it is redone
  once per optimizer step (which writes the weight in place) and after the
  layer is moved or converted (``.to()``, ``.half()``). A weight packed in a
  1D or non-square format cannot serve the input gradient. It raises there,
  with a message that names the square-tile spelling. For inference there is
  no backward pass, and a 1D packed weight is fine. The tile may carry any
  scale rule, a Four Over Six one included
  (``dataclasses.replace(w16, scale_rounding=ScaleRounding.FOUR_OVER_SIX_MSE)``):
  each tile then makes its choice between two scales once, for both passes.

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
that a 1D layer saves for its backward pass is the parameter itself, and the
16 x 16 packing is an extra allocation, kept alive from the forward pass to
the input gradient. A ``BlockQuant`` remembers one weight, so give each layer
its own. What packing once buys is time and consistency: the weight is
quantized once, and every product reads the same codes. Each GEMM's result is
float32, stored in the input's dtype. To drop the float weight, freeze the
layer (next section).

Freezing a layer to its codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For inference, or to train other layers around a fixed one,
:meth:`~mptorch.quant.QLinear.freeze_packed_` replaces a layer's float weight
by its block codes, in place, and :func:`~mptorch.quant.freeze_packed_` does
so for every block-format ``QLinear`` of a model, one layer at a time. The
layer becomes a :class:`~mptorch.quant.QPackedLinear`. It is still a
``QLinear``, its ``weight`` is ``None``, and its codes and scales are two
``uint8`` buffers, ``weight_codes`` and ``weight_scales``, which follow
``.to()`` and stay ``uint8`` under ``.half()``. On a 4096 x 4096 weight that is
8.06 MiB for NVFP4's 16 x 16 tiles, 9.00 for its 1 x 16 blocks and 16.5 for
MXFP8, where the layer held 64 MiB of float32 and the codes besides. The
codes are what the layer's forward multiplies (its ``weight_quant`` packing,
or the weight packed in the format the layer's formats name), so the frozen
layer returns what it returned before, word for word.

.. literalinclude:: ../snippets/block_freeze.py
   :language: python
   :caption: docs/snippets/block_freeze.py

.. literalinclude:: ../snippets/block_freeze.out
   :language: text
   :caption: output

* **Gradients.** A square-tiled weight still serves the input gradient, so
  layers below a frozen one can train. A 1D weight raises there, because its
  packing reads differently along the other axis and there is no float
  weight left to pack again. A frozen layer has no weight gradient, and its
  bias stays a float parameter.
* **Memory.** The float weight is freed once nothing else refers to it, so
  freeze before building an optimizer or any copy of the parameters, which
  would keep it alive. A model that does not fit on its device as float can
  be frozen on the CPU and then moved. On CUDA the freed memory returns to
  torch's caching allocator; ``torch.cuda.empty_cache()`` hands it back to the
  driver.
* **Stochastic rounding.** A freeze draws once. With the packed-once recipe it
  keeps the packing the last forward used, draws included. A layer that packs
  in each hook draws again in every forward.
* **Checkpoints.** The format is code, as every hook of the layer's formats
  is. The ``state_dict`` holds the codes, the bias and an ``_extra_state``
  that records the packing's format, shape, tensor scale, dtype and carrier.
  To load, build the model as it was built, freeze it with ``pack=False``
  (zeroed codes, nothing packed), and load. A model built on the meta device
  allocates nothing until ``load_state_dict(..., assign=True)``, so its float
  weights never exist. A checkpoint frozen in another format or shape
  raises, naming both, and a float checkpoint does not load into a frozen
  model, nor the reverse.
* **Code that reads** ``layer.weight`` (a model summary, an initializer) meets
  ``None``, and ``reset_parameters`` raises.
  :meth:`~mptorch.quant.QPackedLinear.unfreeze_packed_` turns the layer back
  into a ``QLinear`` whose weight is the decoded codes: the quantized weight,
  not the original.

``tensor_scale_epilogue=True`` applies a format's per-tensor scales to each
product's result instead of in the decode, as NVIDIA's NVFP4 GEMMs do
(:doc:`kernels/block`, "Tensor scales").

Activations between layers
--------------------------

A layer's ``input_quant`` rounds what enters the layer. The gradient of that
rounding is handled by the layer itself: it flows to the input through the
layer's ``bwd_igrad_math``. To round a tensor *between* layers (an
activation, the logits, the output of a block that is not a layer), use a
:class:`~mptorch.quant.Quantizer`. It is a module, and it also specifies
what its gradient is rounded to (see :doc:`quantizers`). The last lines of
the convolution run above show one in a ``Sequential``.

What a layer keeps for backward
-------------------------------

The backward pass needs :math:`Q_{\text{in}}(X)` again for the weight
gradient and :math:`Q_{\text{w}}(W)` for the input gradient. A layer can keep
each as a quantized copy, a tensor the size of the operand, or keep the raw
operand and quantize it again in the backward pass. The second way costs one
more quantization per operand per step and saves the copy. The flags
``recompute_input`` and ``recompute_weight`` of
:class:`~mptorch.quant.QAffineFormats` choose (``recompute_a`` and
``recompute_b`` of :class:`~mptorch.quant.QMatmulFormats` for a
:class:`~mptorch.quant.QMatmul`). Each defaults to ``None``, which means the
layer's own default:

====================  ======================  ======================
layer                 input (``a``)           weight (``b``)
====================  ======================  ======================
``QConv1d/2d/3d``     quantized again         quantized again
``QLinear``           copy kept               quantized again
``QMatmul``           copy kept               copy kept
====================  ======================  ======================

The gradients are the same either way, to the bit. Under stochastic rounding
the second quantization sees the random generators as the first did, and
leaves them as it found them.

Quantizing again saves memory only where the raw operand stays alive anyway.
A weight always does, since it is the layer's parameter. An input does when
the op that made it keeps its output for its own backward pass: ReLU,
sigmoid, tanh and softmax do, and so does the data batch, which the training
loop holds. GELU, SiLU, ReLU6, LayerNorm, BatchNorm and max pooling keep
their *input* instead, and an addition keeps nothing. After those, the raw
input would be alive only for this layer, so quantizing it again holds the
same memory and adds the work. That is why the defaults differ. A
convolution's input is usually a ReLU output or the data. A Linear layer's
input is usually the output of a LayerNorm or of a GELU or SiLU. A matmul's
operands are activations of any origin. Set ``recompute_input=True`` for a
ReLU MLP, ``recompute_input=False`` for a convolution after a GELU, and
``recompute_a=True`` for the product of attention's softmax probabilities
with :math:`V`.

The cost is one elementwise pass per recomputed operand. Measured on an RTX
4060, it was within 0.6% of the step when the products run on mptorch's
kernels. With PyTorch's own products it was a few percent, and 12% for the
weights of 4096-wide Linear layers at 256 tokens, the case where recomputing
also saved the most memory (from 148 to 20 MiB for three layers).

Three cases keep the copy whatever the default:

- **A quantizer of your own.** Only a :class:`~mptorch.quant.Quant` (not in
  place), a :class:`~mptorch.quant.BlockQuant` used as a fake quantizer, and
  a :class:`~mptorch.quant.Quantizer` over one are run twice by default. A
  callable with state, such as a QAT observer, would update that state a
  second time in the backward pass, and could round differently there.
  Setting the flag to ``True`` runs any callable again. That is your promise
  that it depends only on its input and the random generators.
- **A result that is not a tensor.** A packed weight (``BlockQuant.pack``)
  is compact already, and its memo holds it anyway.
- **A flag set to** ``False``.

An operand that is kept raw is checked by autograd as ``nn.Conv2d`` checks
its own saved input. Writing to it in place between the forward and backward
passes raises "modified by an inplace operation". With the flag set to
``False`` the layer keeps its own copy and the write does no harm.

The run below counts what four convolution layers keep for the backward
pass, in activations, beyond what the parameters and the input already hold:
four for ``nn.Conv2d``, eight with the copies kept, and four with the
defaults. Then it shows what a Linear layer keeps, by default and with a
quantizer of its own.

.. literalinclude:: ../snippets/layers_recompute.py
   :language: python
   :caption: docs/snippets/layers_recompute.py

.. literalinclude:: ../snippets/layers_recompute.out
   :language: text
   :caption: output

float64 models and the carrier
------------------------------

A layer computes in its tensors' *carrier* (:doc:`concepts`). By default a
float32, float16 or bfloat16 layer rounds and multiplies in binary32, and a
float64 layer in binary64. This applies to every quantizer and every
product. The formats do not determine the carrier unless they name one; the
dtype does. ``model.double()`` therefore moves a whole model into binary64,
and ``model.float()`` moves it back. Each hook checks its formats against
the carrier it meets on every call.

A float64 model can use formats that a float32 one cannot: up to 53 bits of
precision and ten exponent bits. That is needed to study arithmetic *wider*
than binary32, or to have an exact reference.

A float32 model can get binary64's arithmetic without changing dtype. Give
``carrier=torch.float64`` to the factory and to each
:class:`~mptorch.quant.Quant`. Each hook then widens its operands, computes
in binary64 and narrows its result back. The model's signals stay float32
and are held to what float32 can store.

The other direction is not a carrier choice. A float64 model computes the
float32 way by becoming a float32 model. On a GPU that is also the faster
arithmetic for the matrix products (:doc:`kernels/calling`, "Performance
notes").

The convolutions of :func:`~mptorch.quant.conv_formats` and the block
products follow the same rule. A float64 model computes them in binary64:
its block operands are packed, decoded and multiplied there. A float32 model
does so too when its mac (or :func:`~mptorch.quant.block_gemm_formats`) names
``carrier=torch.float64``.

The run compares one float64 reference with a float64 layer in a 30-bit
arithmetic that only binary64 can hold, and a float64 convolution in the same
arithmetic with ``nn.Conv2d``'s float64 one. It then compares the reference
with a float32 layer in each carrier, in three arithmetics:

- a 30-bit multiply format under a 24-bit accumulator, which only binary64
  can give a float32 layer;
- an FP8 recipe. Its operands are rounded to 8 bits before any product, so
  both carriers compute every intermediate exactly and agree to the bit;
- a 22-bit arithmetic on unquantized operands. Binary32 keeps 24 bits of
  each product and sum before the formats' rounding, and binary64 keeps 53.

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
   the formats binary32 carries, and float64 for wider formats or an exact
   reference. Use ``carrier=torch.float64`` where a float32 (or
   half-precision) model should compute in binary64 and keep its dtype.
1. Choose the format of each **signal**: a ``Quant`` (or a ``Quantizer``, or
   a scaled quantizer of your own) for ``input_quant``, ``weight_quant``,
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
