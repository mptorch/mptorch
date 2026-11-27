Attention
=========

An attention block's core is two matrix products with a softmax between
them. :func:`~mptorch.quant.qattention` computes it from the pieces the other
pages describe, each with its own formats and its own gradient: the products
on the core routine of :doc:`index`, the scale as a pointwise product
(:doc:`pointwise`), the softmax of :doc:`reductions`. It takes
``F.scaled_dot_product_attention``'s arguments, and its formats are an
:class:`~mptorch.quant.AttentionFormats`. :doc:`/layers` shows
``nn.MultiheadAttention``'s counterpart, and :doc:`/instrument` how a recipe
reaches a model's attention, however the model spells it.

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
more on the CPU, whose vector tiles take the deterministic modes only.

Flash attention
---------------

``AttentionFormats(..., flash=FlashFormats(block_q, block_kv, ...))`` makes
:func:`~mptorch.quant.qattention` compute FlashAttention-2's order
(Algorithms 1 and 2 of arXiv:2307.08691) in place of the steps above. The
two orders sum differently:

- **The standard path** forms the whole row of scores, takes its maximum
  :math:`m`, and divides each :math:`e_j = \exp(s_j - m)` by their sum
  before the product with :math:`v`: :math:`o = \sum_j (e_j / \ell)\, v_j`.
- **Flash** takes the keys ``block_kv`` at a time and keeps, for each
  query, a running maximum :math:`m`, sum :math:`\ell` and output
  :math:`O`. For each tile of scores :math:`S`, with :math:`m' =
  \max(m, \max_j S_j)`, :math:`\tilde P = \exp(S - m')` and
  :math:`\alpha = \exp(m - m')`, it updates :math:`\ell \leftarrow
  \alpha\ell + \sum_j \tilde P_j` and :math:`O \leftarrow \alpha O +
  \tilde P V`. It divides once, :math:`o = O / \ell`, and saves
  :math:`L = m + \log \ell` for the backward. The scores are never held
  whole.

The parts keep their roles, tile by tile. The three steps the standard path
has no counterpart for are ``FlashFormats``' own slots:

==========================  ==================================================
step                        rounded by
==========================  ==================================================
q and k tiles               ``qk.input_quant``, ``qk.weight_quant``
:math:`S = q_i k_j^\top`    ``qk``'s mac, then ``qk.output_quant``
:math:`S \cdot` scale       ``scores``
the mask                    added in the carrier; the maximum is exact
:math:`\tilde P`            ``softmax.sub`` and ``softmax.exp``
a tile's row sums           ``softmax.acc``
:math:`\alpha`              ``flash.rescale``
the updates of ℓ and O      ``flash.combine``, each computed in the carrier
                            and rounded once
:math:`\tilde P V`          ``pv.input_quant`` on :math:`\tilde P`,
                            ``pv.weight_quant`` on the tile of ``v``, ``pv``'s
                            mac
:math:`O / \ell`            ``softmax.div``, then ``pv.output_quant`` once
:math:`L`                   ``flash.lse``
==========================  ==================================================

- **Tiles.** The operand quantizers run per tile: ``q`` per query tile, and
  ``k`` and ``v`` per key tile. A :class:`~mptorch.quant.ScaledQuant` then
  scales each tile by its own maximum, as FA3 scales its FP8 blocks.
  Elementwise quantizers, and MX block formats along the head dimension,
  give the same words either way. Everything inside runs in
  ``FlashFormats.carrier``, binary32 by default for float32, float16 and
  bfloat16 operands. The output is stored in the operands' dtype.
- **Causal tiles.** With ``is_causal``, the key tiles wholly above the
  diagonal are skipped, which gives the words of computing them under the
  same mask, since a masked tile adds exactly nothing. The tiles are known
  from the shapes, so flash compiles into one graph. An explicit
  ``attn_mask``'s tiles are all computed.
- **Batched tiles.** Where the arithmetic cannot tell the tiles apart, each
  key tile meets every query tile it reaches in one call: each step runs
  once over all those rows, in the order of the key tiles, which gives the
  words of one query tile at a time. That holds when both products and the
  softmax's sums are mptorch's (a mac in each pass: ``FusedMac(None)``
  multiplies in the carrier) and every slot is an elementwise, deterministic
  :class:`~mptorch.quant.Quant`. The forward and the backward decide apart,
  so a recipe whose gradients are PyTorch's still batches its forward.
  These run one query tile at a time:

  - PyTorch's own products, since cuBLAS's words for a taller product are
    not its words for a tile;
  - stochastic rounding, whose draws are keyed on an element's index in its
    call;
  - a :class:`~mptorch.quant.ScaledQuant`, which scales each tile by its own
    maximum;
  - a block format's quantizer or mac, and any other callable.
- **The backward** is FA2's. First, :math:`D = \sum (dO \circ O)` per query,
  under ``softmax.bwd_mul`` and ``softmax.bwd_acc``. Then for each tile:
  :math:`P = \exp(S - L)`, unrounded, as the standard softmax's backward
  takes the unrounded softmax; :math:`dV`, :math:`dP = dO\,V^\top`,
  :math:`dS = P \circ (dP - D)` under ``softmax.bwd_sub`` and
  ``softmax.bwd_out``, the scale's gradient as ``scores`` rounds it, then
  :math:`dQ` and :math:`dK`. Each product is in its pass's mac and gradient
  quantizer, and ``combine`` rounds each tile's addition to the three
  gradients. The scores are recomputed from the saved quantized tiles.
  When their product, its rounding or the scale's can draw random bits,
  the forward keeps each tile's scores instead, so the backward sees the
  forward's words.
- **Rotations.** ``qk.rotation`` rotates ``q`` and ``k`` whole along the
  head dimension, before the tile loop. Their gradients are summed in the
  rotated basis and rotated back once. ``pv.rotation`` runs along the keys
  within each tile.
- **What it refuses.**

  - ``dropout_p > 0``, since flash would draw its dropout per tile.
  - A mask that requires grad, since flash gives the mask no gradient.
  - :meth:`~mptorch.quant.Recipe.attention_parts`, since an eager
    attention is three calls.
  - ``QMultiheadAttention``'s ``need_weights=True``, since flash never forms
    the weights.

On GPT-2 small, the recipe ``fp8+attention`` of
``examples/instrument/gpt2_perplexity.py`` puts every projection but
``lm_head`` in FP8, the attention's operands and probabilities in E4M3, and
the softmax in Binary14p8. Its products are mptorch's in the forward and
PyTorch's in the backward, so flash batches its forward. Under it, flash gives
a lower perplexity than the standard path, over WikiText-2's first 32,768
tokens (an RTX 4060 laptop GPU, 2026-11-26, the mean of two runs):

===========================  ==========  ===========  ===================
attention                    perplexity  tokens/s     one tile at a time
===========================  ==========  ===========  ===================
standard path                28.59       2,965
flash, 128 x 128 tiles       28.19       3,376        2,755
flash, 64 x 64 tiles         28.25       3,081        1,495
===========================  ==========  ===========  ===================

Flash alone, nothing rounded, gives float32's 26.76. A likely reason for
flash's lower perplexity, not verified: the probabilities are rounded to
E4M3. The standard path rounds them after the division, where most of a
1,024-key row's probabilities lie near :math:`1/1024`, below E4M3's smallest
normal value :math:`2^{-6}`. Flash rounds :math:`\tilde P`, whose largest
entry in each row is 1.

**Cost.** A causal attention over ``[1, 8, L, 64]`` under E4M3 operand
quantizers, with 128 x 128 tiles (``dev/benchmarks/timing/flash_attention.py``):

- **Memory.** On CUDA, flash holds 42 MiB at :math:`L = 4096` where the
  standard path holds 1,592 MiB (91 and 2,080 MiB with the backward). The
  batched path holds a key tile's scores for every query row it reaches,
  about 1.6 times the peak of one tile at a time, and under a tenth of the
  standard path's.
- **CUDA time.** One query tile at a time, each pair of tiles is about
  twenty kernel launches, and flash takes 5 to 8 times the standard path's
  time. Batched, with every product and sum ``FusedMac(None)``, it runs 1.7
  times faster than that at :math:`L = 512` and 4.4 times at 4,096, and from
  :math:`L = 1024` it is faster than the standard path in the same recipe
  (0.44 of its time at 4,096, 0.57 with the backward), since it skips the
  causal tiles.
- **CUDA graphs.** ``torch.compile(fn, fullgraph=True,
  backend="cudagraphs")`` records flash's kernels and replays them, which
  saves the launches' host time and changes no word, forward or backward.
  The batched path then runs another 1.09-1.17 times faster, after a compile
  of a few seconds. Under stochastic rounding each call still draws its own
  random bits. Inductor, the default backend, generates kernels of its own
  for the elementwise steps. They round some results differently, by about
  an ulp of binary32, and the compile's time and memory grow with the
  sequence's length.
- **CPU time.** Flash is faster than the standard path, by 1.2-2.6x on 16
  threads and 1.7-3.8x on one. It works on cache-sized tiles, skips the
  causal tiles, and makes no passes over the whole score matrix. Batching
  gains less here (1.05-1.85x on 16 threads, forward), since a CPU call
  costs little next to its arithmetic.

.. literalinclude:: ../../snippets/attention_flash.py
   :language: python
   :caption: docs/snippets/attention_flash.py

.. literalinclude:: ../../snippets/attention_flash.out
   :language: text
   :caption: output
