Elementwise quantization
========================

An elementwise quantizer rounds every element of a tensor to a format,
independently. It is the simplest thing MPTorch does and the building block
of everything else: what a layer applies to its weights and activations, what
a straight-through estimator wraps, what a quantization-aware-training
observer sits in front of.

.. math::

   \tilde{x}_i = Q_{F,\circ}(x_i) \qquad \text{for every element } i.

There are three spellings, from the most explicit to the most convenient.

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

- ``bias`` is optional for ``binaryK_quantize`` (it defaults to IEEE P3109's,
  :math:`2^{K-P-1}`, or :math:`2^{K-P}` unsigned) and required for
  ``superfp_quantize``, for the reasons given in :doc:`concepts`.
- Any floating dtype PyTorch trains in is accepted: float32, float64,
  float16 and bfloat16. A float64 input is rounded in binary64, directly, and
  the format is held to binary64's bounds -- up to 53 bits of precision and
  ten exponent bits; the others are rounded on their float32 value, in
  binary32, and held to its bounds (:doc:`concepts`, "What the carrier can
  hold"). ``carrier="binary32"`` rounds a float64 input the float32 way, and
  the run shows the two disagreeing on a value float32 cannot hold.
- The result is stored back in the input's dtype, which rounds a float16 or
  bfloat16 result once more. The input is already a value of that dtype, so
  only an edge of the format's range can miss, and the call warns when one
  can -- the run shows E5M2's top meeting float16's. :doc:`concepts` gives
  the rule, and the stricter one a GEMM's result is held to.
- The result is a fresh tensor; the input is never modified.
- NaN passes through with its payload. An infinity passes through too,
  except under ``SaturationMode.SAT_FINITE``, which clamps it to the largest
  finite value like any other overflow.
- ``prng_bits`` only matters under ``RoundMode.SR``. The random bits are
  drawn in the carrier, below the format's mantissa, so the two share its
  mantissa bits: 23 in binary32 and 52 in binary64, as the run shows.
- CPU and CUDA produce bit-identical results under every deterministic mode.
  Under ``SR`` each device draws from its own default generator, so
  ``torch.manual_seed`` reproduces a run *on the same device*.

Quant: a format and a rounding mode, as a function
--------------------------------------------------

:class:`~mptorch.quant.Quant` binds a format object and a rounding mode into
a callable. It is what the ``*_quant`` slots of a layer's formats are meant
to hold, and it is exactly the function call above with the arguments filled
in from the format -- the run checks the two are equal to the bit.

.. literalinclude:: ../snippets/quantizers_quant.py
   :language: python
   :caption: docs/snippets/quantizers_quant.py

.. literalinclude:: ../snippets/quantizers_quant.out
   :language: text
   :caption: output

Formats and Quants are frozen dataclasses: they compare and hash by value,
which is what lets a layer resolve its format once and the library memoize
on it. Build them once and hold them, rather than constructing one per call.

Quantizer: a straight-through estimator
---------------------------------------

Rounding is a step function. Its derivative is zero almost everywhere and
undefined at the steps, so a gradient "through" a quantizer is not something
autograd can derive -- it is a modelling choice, and the standard choice is
the **straight-through estimator** (STE): pretend, in the backward pass, that
the quantizer was the identity. :class:`~mptorch.quant.Quantizer` is that
choice, with one refinement -- the gradient that passes through can itself be
rounded, to a format of its own:

.. math::

   \text{forward:}\quad y = Q_{F_{\text{fwd}}}(x), \qquad
   \text{backward:}\quad \frac{\partial L}{\partial x} := Q_{F_{\text{bwd}}}\Big(\frac{\partial L}{\partial y}\Big).

Either format may be ``None`` (that direction is left alone), a format object
(wrapped in a ``Quant`` with round-to-nearest), or any callable
``Tensor -> Tensor`` of your own. The asymmetry is the point: a
quantization-aware forward pass usually wants a narrow format, and the
gradient a wider one.

.. literalinclude:: ../snippets/quantizers_quantizer.py
   :language: python
   :caption: docs/snippets/quantizers_quantizer.py

.. literalinclude:: ../snippets/quantizers_quantizer.out
   :language: text
   :caption: output

Being an ``nn.Module``, a ``Quantizer`` can sit anywhere in a model -- between
two layers, in a ``*_quant`` slot of a layer's formats -- and is registered
with its parent, so any state it carries travels with the model's
``state_dict``.

The rule for gradients
~~~~~~~~~~~~~~~~~~~~~~

The run above also shows what the raw functions do when handed a tensor that
requires a gradient: they raise. None of MPTorch's operators is
differentiable in autograd's sense -- a quantizer has a zero derivative, and a
GEMM that simulates its own arithmetic has no derivative the framework could
infer -- so instead of returning a tensor whose gradient would silently be
dropped (leaving ``.grad`` as ``None`` after ``.backward()``, and a model that
silently does not train), each raw operator names the entry point that does
carry a gradient: ``Quantizer`` for the quantizers, ``qmatmul`` /
``QMatmul`` for the matrix products.

The check is *gradient mode and* ``requires_grad``, which is why the layers
are unaffected: a ``torch.autograd.Function``'s forward runs with gradient
mode off, so ``QLinear`` and friends call the raw operators on tensors that
require grad without tripping it. Under ``torch.no_grad()``, or on a detached
tensor, the raw functions simply run.
