Attention
=========

An attention block's core is two matrix products with a softmax between
them. :func:`~mptorch.quant.qattention` computes it from the pieces the other
pages describe, each with its own formats and its own gradient: the products
on the core routine of :doc:`index`, the scale as a pointwise product
(:doc:`pointwise`), the softmax of :doc:`reductions`. It takes
``F.scaled_dot_product_attention``'s arguments, and its formats are an
:class:`~mptorch.quant.AttentionFormats`. :doc:`/layers` shows
``nn.MultiheadAttention``'s counterpart and how a recipe reaches a model's
attention, however the model spells it.

The steps
---------

For queries :math:`q`, keys :math:`k` and values :math:`v`,
``qattention(q, k, v, formats)`` computes, one step after another,

.. math::

   s = q\,k^\top, \qquad s' = s \cdot \sigma + m, \qquad
   p = \operatorname{softmax}(s'), \qquad o = p\,v,

with :math:`\sigma` the scale (:math:`1/\sqrt{E}` by default, :math:`E` the
head dimension) and :math:`m` the mask. Each part of ``formats`` rounds one
of them:

- ``qk``, a :class:`~mptorch.quant.GemmFormats`, is the product
  :math:`q\,k^\top`: its ``input_quant`` rounds :math:`q`, its
  ``weight_quant`` :math:`k`, its ``fwd_mac`` multiplies, and its
  ``output_quant`` rounds :math:`s`;
- ``scores``, a :class:`~mptorch.quant.PointwiseFormats`, rounds
  :math:`s \cdot \sigma`, computed in its carrier (round to odd included);
- ``softmax``, a :class:`~mptorch.quant.SoftmaxFormats`, is the softmax, each
  of its steps rounded as on :doc:`reductions`;
- ``pv``, a :class:`~mptorch.quant.GemmFormats`, is the product
  :math:`p\,v`: its ``input_quant`` rounds the probabilities, as FP8
  attention rounds them before the second product, its ``weight_quant``
  :math:`v`, and its ``output_quant`` the result.

A part left ``None`` is PyTorch's own step, and with every part ``None``
``qattention`` is ``F.scaled_dot_product_attention`` itself. Each step's
result is stored in the operands' dtype, as an eager attention stores it.
:meth:`AttentionFormats.uniform <mptorch.quant.AttentionFormats.uniform>`
gives both products ``GemmFormats.uniform(mac, quant)`` and the softmax
``SoftmaxFormats.uniform(mac, quant)``; under a ``BlockMac`` the products
multiply block formats (MXFP8 attention) and the softmax is PyTorch's.

.. literalinclude:: ../../snippets/attention_qattention.py
   :language: python
   :caption: docs/snippets/attention_qattention.py

.. literalinclude:: ../../snippets/attention_qattention.out
   :language: text
   :caption: output

Where the steps differ from PyTorch's kernels
---------------------------------------------

The steps are those of transformers' eager attention (``eager_attention_forward``),
which is the arithmetic a recipe can give an eager model, rather than those
of PyTorch's fused kernels, which have none to give:

- **The scale** multiplies the product, :math:`(q\,k^\top) \cdot \sigma`.
  PyTorch's math kernel scales :math:`q` and :math:`k` by
  :math:`\sqrt{\sigma}` each before the product, and
  ``nn.MultiheadAttention``'s ``need_weights=True`` path scales :math:`q` by
  :math:`\sigma`.
- **The softmax** of float16 and bfloat16 scores computes in float32 and
  stores its result in their dtype, as Llama's eager attention does
  (``softmax(..., dtype=torch.float32).to(q.dtype)``).
- **A boolean mask** (``True`` attends), and ``is_causal``'s (the lower
  triangle, aligned to the top left), are added as :math:`0` and
  :math:`-\infty` in the scores' dtype; a float mask is added as it is,
  converted to that dtype.
- **A query the mask leaves without a key** attends to nothing: its
  probabilities, its output and its gradients are zero, as in PyTorch's
  kernels (``nn.MultiheadAttention``'s ``need_weights=True`` path gives NaN
  there). Its scores are replaced before the softmax as well as after it,
  since a row of :math:`-\infty` would carry NaN through the softmax's
  gradient into every gradient of :math:`q` and :math:`k`.
- **Dropout** is PyTorch's, after the softmax and before the probabilities
  are rounded, whenever ``dropout_p`` is positive.

Grouped-query attention
-----------------------

With ``enable_gqa=True`` the keys and values have fewer heads than the
queries, :math:`H_{kv}` dividing :math:`H`, and query head :math:`h` reads
key head :math:`h / G` (:math:`G = H / H_{kv}`, rounded down). The key and
value heads are repeated over their groups, as transformers' ``repeat_kv``
and PyTorch's ``repeat_interleave`` repeat them, so ``enable_gqa=True`` is
the same call on repeated heads, word for word: a key head's gradient is the
sum of its copies'. This is what makes a model that passes ``enable_gqa`` to
SDPA, one that calls ``repeat_kv`` first (transformers does, whenever there
is a mask) and its eager twin compute the same words.

The gradients
-------------

The backward is each step's own: the two products' gradients are the
matrix products of :doc:`arithmetic` in the ``igrad_mac`` and ``wgrad_mac``
of ``qk`` and ``pv``, with their gradient quantizers; the softmax's is its
closed form or PyTorch's gradient (:doc:`reductions`); the scale's and the
mask's are PyTorch's. Stochastic rounding draws in the order an eager
attention's calls would.

What attention costs
--------------------

``dev/benchmarks/timing/attention_ops.py`` times a causal attention over
``[batch, heads, length, 64]`` against SDPA's math kernel (the unfused
product, softmax and product), with both products in ``SplitMac(BinaryK(8,
4), BinaryK(14, 8))`` and every operand, probability and softmax step
rounded to ``BinaryK(8, 4)`` under RNE (an RTX 4060 laptop GPU and a Ryzen 9
7940HS at 4.7 GHz, 2026-11-24):

=================  ======  ===============  ===============
batch x heads x L  CUDA    one CPU thread   sixteen threads
=================  ======  ===============  ===============
8 x 8 x 256        7.7x
4 x 8 x 256                7.5x             4.6x
2 x 8 x 1024               9.0x             5.3x
4 x 16 x 1024      8.0x
1 x 16 x 4096      7.5x
=================  ======  ===============  ===============

Forward and backward together cost 8.6-10x on CUDA. On CUDA the two
products take three quarters of the time and the softmax the rest: an
attention costs what its products would cost as simulated layers.
Stochastic rounding costs about three times RNE's time on CUDA, and much
more on the CPU, whose vector tiles take RNE only.
