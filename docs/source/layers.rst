Layers
======

The layers are where the two kinds of rounding meet: a quantizer for each
signal, and an arithmetic for each matrix product. Every one of them is a
thin subclass of its ``torch.nn`` counterpart whose ``forward`` delegates to a
custom ``torch.autograd.Function``, so a model built from them trains with
the ordinary optimizer, scheduler and loss.

What a quantized layer computes
-------------------------------

Take a linear layer, :math:`y = xW^{\top} + b`. With
:math:`G = \partial L / \partial y` arriving from above, the three gradients
are

.. math::

   \frac{\partial L}{\partial x} = G\,W, \qquad
   \frac{\partial L}{\partial W} = G^{\top} x, \qquad
   \frac{\partial L}{\partial b} = \sum_{\text{batch}} G .

A :class:`~mptorch.quant.QLinear` computes the same four products with a
quantizer in front of each signal and an arithmetic of its own for each
product:

.. math::

   \begin{aligned}
   y &= \big[\, Q_{\text{in}}(x)\; Q_{\text{w}}(W)^{\top} \,\big]_{\text{fwd}} + Q_{\text{b}}(b) \\[4pt]
   \frac{\partial L}{\partial x} &= \big[\, Q_{\text{igrad}}(G)\; Q_{\text{w}}(W) \,\big]_{\text{igrad}} \\[4pt]
   \frac{\partial L}{\partial W} &= \big[\, Q_{\text{wgrad}}(G)^{\top}\; Q_{\text{in}}(x) \,\big]_{\text{wgrad}} \\[4pt]
   \frac{\partial L}{\partial b} &= \sum_{\text{batch}} Q_{\text{bgrad}}(G)
   \end{aligned}

where :math:`[\cdot]_{\text{fwd}}`, :math:`[\cdot]_{\text{igrad}}` and
:math:`[\cdot]_{\text{wgrad}}` are the three matrix products, each in
whatever arithmetic its hook names. Note that the quantized weight and input
from the forward pass are the ones the backward reuses -- the layer saves
:math:`Q_{\text{w}}(W)` and :math:`Q_{\text{in}}(x)`, not :math:`W` and
:math:`x` -- and that the incoming gradient is quantized *separately* for the
two paths it feeds, since the input gradient and the weight gradient
commonly want different formats.

QAffineFormats
--------------

:class:`~mptorch.quant.QAffineFormats` is the configuration object a layer
takes: nine optional slots, every one of which defaults to "plain PyTorch".

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
:class:`~mptorch.quant.Quant`, a :class:`~mptorch.quant.Quantizer`, a
function of your own (a scaled quantizer, an observer). ``QAffineFormats``
is itself an ``nn.Module``, so a quantizer that is a module is registered in
the model's ``state_dict``.

The run below builds a layer with a quantizer on every signal -- E4M3 in the
forward direction, E5M2 on the gradients -- and then recomputes the four
equations by hand on the same tensors. They agree to the bit.

.. literalinclude:: ../snippets/layers_qlinear.py
   :language: python
   :caption: docs/snippets/layers_qlinear.py

.. literalinclude:: ../snippets/layers_qlinear.out
   :language: text
   :caption: output

Custom arithmetic in the matrix products
----------------------------------------

Leaving the ``*_math`` hooks unset means the three products are PyTorch's
own, in the layer's dtype, however narrow the operands were rounded. To
simulate the arithmetic *inside* them, a factory fills the three hooks with a
GEMM of :doc:`gemm`'s kind:

- :func:`~mptorch.quant.binaryK_gemm_formats` -- ``SplitMac`` in BinaryK
  formats; ``mul_*`` and ``acc_*`` arguments, ``accumulate_quant=False``
  for an unrounded sum.
- :func:`~mptorch.quant.binaryK_gemm_formats_fma` -- ``FusedMac`` in a
  BinaryK format; ``fma_*`` arguments.
- :func:`~mptorch.quant.superfp_gemm_formats`,
  :func:`~mptorch.quant.superfp_gemm_formats_fma` -- the same for SuperFP.

Each returns a ``QAffineFormats`` with *only* the math hooks set; operand
quantizers are assigned afterwards, so the two concerns stay composable. The
hooks follow the layer's call contract --
``fwd_math(q_input, q_weight, q_bias)``,
``bwd_igrad_math(q_grad, q_weight)``, ``bwd_wgrad_math(q_grad, q_input)`` --
and take care of the transposes and of folding any leading batch dimensions
of the input into one GEMM. The format is resolved once, when the factory
runs, not on every forward pass; each factory also takes ``carrier``
(`float64 models and the carrier`_).

.. literalinclude:: ../snippets/layers_gemm_formats.py
   :language: python
   :caption: docs/snippets/layers_gemm_formats.py

.. literalinclude:: ../snippets/layers_gemm_formats.out
   :language: text
   :caption: output

The last three lines show the same weights under three other arithmetics,
measured against the split BinaryK layer: a fused E4M3 step with stochastic
rounding, and the two SuperFP cores, whose format (two binades of mantissa)
is a poor fit for these activations and says so.

Convolutions
------------

:class:`~mptorch.quant.QConv1d`, :class:`~mptorch.quant.QConv2d` and
:class:`~mptorch.quant.QConv3d` take the same ``QAffineFormats`` and apply
its quantizers to the same signals; the equations are the convolution
analogues of the linear ones, with :math:`G^{\top} x` becoming the
weight-gradient convolution and :math:`GW` the input-gradient (transposed)
convolution. Their math hooks receive the convolution's geometry as keyword
arguments (``stride``, ``padding``, ``dilation``, ``groups``, ``nd``). No
custom-arithmetic convolution kernel ships, so the hooks are yours to write
if you need one; with them unset the convolution is PyTorch's own on the
quantized operands.

.. literalinclude:: ../snippets/layers_conv.py
   :language: python
   :caption: docs/snippets/layers_conv.py

.. literalinclude:: ../snippets/layers_conv.out
   :language: text
   :caption: output

Activations between layers
--------------------------

A layer's ``input_quant`` rounds what enters it, but the gradient of that
rounding is the layer's own business (it flows to the input through the
layer's ``bwd_igrad_math``). To round a tensor *between* layers -- an
activation, the logits, the output of a block that is not a layer -- use a
:class:`~mptorch.quant.Quantizer`, which is a module and also says what its
gradient is rounded to (see :doc:`quantizers`). The last lines of the
convolution run above show one in a ``Sequential``.

float64 models and the carrier
------------------------------

A layer computes in its tensors' *carrier* (:doc:`concepts`): a float32 --
or float16, or bfloat16 -- layer rounds and multiplies in binary32, and a
float64 layer in binary64, every quantizer and every product. Nothing in the
formats says which; the dtype does, so ``model.double()`` moves a whole model
into binary64 and ``model.float()`` back, and each hook checks its formats
against the carrier it meets on every call.

A float64 model therefore reaches formats a float32 one cannot -- up to 53
bits of precision and ten exponent bits -- which is what a study of
arithmetic *wider* than binary32 needs, or one whose reference has to be
exact. To run a float64 model in binary32 instead, as it would have run
before float64 had a carrier of its own, give ``carrier="binary32"`` to the
factory and to each :class:`~mptorch.quant.Quant`. On a GPU that is also the
faster arithmetic for the matrix products (:doc:`gemm`, "Performance notes").

The run compares the carriers on one float64 layer: a 30-bit arithmetic only
binary64 can hold; an FP8 recipe, whose operands are rounded to 8 bits before
any product, so that both carriers compute every intermediate exactly and
agree to the bit; and a 22-bit arithmetic on unquantized operands, where
binary32 keeps 24 bits of each operand, product and sum ahead of the
formats' rounding and binary64 keeps 53.

.. literalinclude:: ../snippets/layers_float64.py
   :language: python
   :caption: docs/snippets/layers_float64.py

.. literalinclude:: ../snippets/layers_float64.out
   :language: text
   :caption: output

Putting a model together
------------------------

A mixed-precision model is a set of decisions per layer, and this is where
they are written down:

0. Choose the **carrier** -- the model's dtype: float32 for the formats
   binary32 carries, float64 for wider ones or an exact reference, with
   ``carrier="binary32"`` where a float64 model should compute the float32
   way.
1. Choose the format of each **signal** -- a ``Quant`` (or ``Quantizer``, or
   a scaled quantizer of your own) for ``input_quant``, ``weight_quant``,
   ``bias_quant``, and for the three gradient paths.
2. Choose the **arithmetic** of each matrix product -- a factory from
   :doc:`gemm` for the ``*_math`` hooks, or leave them PyTorch's own.
3. Share one ``QAffineFormats`` across layers that should behave the same,
   or give each layer its own.
4. Round anything else with a ``Quantizer``.

The :doc:`tutorial` walks through exactly this for the 8-bit formats,
training a network with each combination and comparing them.
