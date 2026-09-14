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
  the others on their float32 value, in binary32 unless the call names
  binary64 -- `float64 and the carrier`_ below shows what that changes.
- The result is stored back in the input's dtype, whichever carrier rounded
  it, which rounds a float16 or bfloat16 result once more. The input is already a value of that dtype, so
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
- Every function here also takes ``carrier``, below.
- CPU and CUDA produce bit-identical results under every deterministic mode.
  Under ``SR`` each device draws from its own default generator, so
  ``torch.manual_seed`` reproduces a run *on the same device*.

float64 and the carrier
-----------------------

The arithmetic a quantizer rounds in is its *carrier* (:doc:`concepts`): by
default binary64 for a float64 tensor and binary32 for the other three dtypes.
For a float64 tensor binary64 has three consequences, and
``carrier=torch.float64`` gives the other dtypes the ones their values can
use.

- **Wider formats.** The format is held to binary64's bounds, so up to 53
  bits of precision and ten exponent bits are simulated exactly -- where the
  same call on a float32 tensor raises, because binary32 has only 24 bits to
  round in.
- **Rounded once.** A value float32 cannot hold is rounded on its own bits.
  ``1.0625 + 2**-30`` is above the tie between ``1.0`` and ``1.125`` on
  E4M3's grid, and rounds up; narrowed to float32 it *is* the tie, and
  round-to-nearest-even takes it down.
- **Wider random draws.** Stochastic rounding takes its random bits below the
  format's mantissa in the carrier, so ``P - 1 + prng_bits`` may reach 52
  rather than 23, and each element draws two words of its random stream.

``carrier=torch.float64`` rounds a float32, float16 or bfloat16 tensor in
binary64: the tensor is widened, rounded under binary64's bounds, and the
result narrowed back to the tensor's dtype -- bit for bit the float64 call on
the widened tensor, stored with one rounding. For an elementwise quantizer
that buys the wider formats and the wider random draws, but not the rounding
once: the tensor's values are its dtype's already, and a format binary32
carries rounds them to the same answer in either carrier, as the run checks.
The result is held to what its dtype can store (:doc:`concepts`), which for
float32 is 24 bits of precision and binary32's range at the edges.
``carrier=torch.float32`` names binary32, which a float64 tensor refuses --
the carrier is never narrower than the tensor -- and
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
