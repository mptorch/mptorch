Sums, softmax and norms
=======================

A sum whose every step rounds to a format is a matrix product with a ones
vector, so it runs on the core routine of :doc:`index`. Softmax and the norms
combine such sums with pointwise steps, each computed in the carrier and
rounded once, as in :doc:`pointwise`:

- :func:`~mptorch.quant.qsum` and :func:`~mptorch.quant.qmean`;
- :func:`~mptorch.quant.qsoftmax`;
- :func:`~mptorch.quant.qlayernorm`, :func:`~mptorch.quant.qrmsnorm`,
  :func:`~mptorch.quant.qgroupnorm` and :func:`~mptorch.quant.qbatchnorm`.

:doc:`/layers` shows the modules built on them, and :doc:`/instrument` how a
recipe reaches the softmaxes, norms and sums of a model's own code.

Sums on the core routine
------------------------

``qsum(x, dim, mac)`` computes :math:`\mathbf{1} \cdot x` with the matrix
product ``mac`` names, one output per sum. Under ``FusedMac(F)`` each step is

.. math::

   s_k = F\big(\operatorname{fma}(1, x_k, s_{k-1})\big)
       = F\big(\circ(s_{k-1} + x_k)\big),

the carrier's sum :math:`\circ` rounded once, then rounded to :math:`F`.
Under ``SplitMac(M, A)`` each :math:`x_k` is first rounded to :math:`M`, and
each sum to :math:`A`. The elements are taken in row-major order over the
summed dimensions. A sum therefore has everything a matrix product has:

- the accumulate algorithms of :doc:`arithmetic`: KAHAN, BLOCK, and TREE
  under a split mac;
- every rounding mode, stochastic rounding keyed by the output's index;
- ``carrier_rounding=RoundMode.RO``;
- both carriers and every backend;
- ``torch.compile`` and ``torch.export``.

``qmean`` divides the sum by the count once, in the carrier, and rounds that
quotient with its ``quant``: multiplying by :math:`1/n` would round
:math:`1/n` first.

.. literalinclude:: ../../snippets/reductions_sum.py
   :language: python
   :caption: docs/snippets/reductions_sum.py

.. literalinclude:: ../../snippets/reductions_sum.out
   :language: text
   :caption: output

A sum over adjacent dimensions reads ``x`` where it lies: the last
dimensions as rows (``ones[1, K] @ x[R, K].mT``, which the CPU's vector tile
runs across 32 rows at a time), and inner ones through a batch whose ones
matrix every batch element shares. Dimensions that are not adjacent are moved
next to each other first, which copies.

What a sum costs
~~~~~~~~~~~~~~~~

Each output is one thread or one lane running its :math:`K` steps in order,
which uses 1/16 of each 16 x 16 tile on CUDA and MPS. Against ``torch.sum``
over the last dimension, each step rounded under RNE to ``BinaryK(14, 8)``
(``dev/benchmarks/timing/reduction_ops.py``; an RTX 4060 laptop GPU and a
Ryzen 9 7940HS at 4.7 GHz, 2026-11-24):

=============  ======  ===============  ===============
rows x K       CUDA    one CPU thread   sixteen threads
=============  ======  ===============  ===============
1024 x 512     5.9x    33x              2.7x
8192 x 256     8.7x    22x              3.4x
8192 x 4096    3.9x    35x              4.8x
65536 x 2048   3.7x
=============  ======  ===============  ===============

Stochastic rounding and KAHAN cost 3-4x RNE's time on CUDA, and 2-6x and
5-17x on one CPU thread, where they run without the vector tile. A format
whose smallest value lies below what binary32's fast cast handles (such as
``BinaryK(16, 8)``, bias 128) costs about twice as much. Built on these sums,
a softmax's or a norm's forward costs 7-17x PyTorch's fused kernel on CUDA,
and forward and backward together 2-9x. In a model whose matrix products are
simulated too, that is a few percent of a step: the products, at the same
16 x 16 tiles, dominate. A dedicated row-reduction kernel could bring the
sums near ``torch.sum``'s time, and is not built.

Softmax
-------

``qsoftmax(x, dim, formats)`` computes, along ``dim``,

.. math::

   e = \exp(x - \max x), \qquad y = e \,/\, \textstyle\sum e,

with the maximum exact. :class:`~mptorch.quant.SoftmaxFormats` rounds
``x - max x`` (``sub``), :math:`e` (``exp``), each step of the sum (``acc``,
a mac) and the quotient (``div``). The exponential is PyTorch's, which is
not correctly rounded (:doc:`pointwise`). ``dtype=`` converts ``x`` first, as
``F.softmax`` does.

Norms
-----

A norm computes :math:`\hat x = (x - \mu) / \sqrt{v + \varepsilon}` over its
statistics' elements, then :math:`\hat x\,\gamma + \beta`:

- ``qlayernorm`` over the last dimensions, ``qrmsnorm`` the same with no
  mean (:math:`v` is the mean of :math:`x^2`);
- ``qgroupnorm`` over each group's channels and positions, with a
  per-channel affine part;
- ``qbatchnorm`` over the batch and positions of each channel in training,
  and over the running statistics in eval.

:class:`~mptorch.quant.NormFormats` rounds each step: the statistics' sums
(``acc``, a mac), the mean and the variance (``div``), :math:`x - \mu`
(``sub``), the squares the variance sums (``sq``), :math:`1/\sqrt{v +
\varepsilon}` (``rsqrt``), :math:`\hat x` (``norm``), the affine part (``mul``,
``add``) and the result (``out``). A square is the carrier's product, rounded
by ``sq`` when set: summing :math:`(x - \mu)^2` as a fused dot product of a
row with itself would take one matrix product per row. A batch norm in
training updates its running statistics in place, by PyTorch's formula and in
their own dtype, from the rounded mean and the rounded variance, made
unbiased.

When every step before the result is ``None``, an op is PyTorch's own
function in the carrier, and its result is rounded once by ``div`` or
``out``. With every slot ``None`` it is PyTorch's function.

Gradients
---------

The backward is the derivative of the unrounded function at the input; the
forward's roundings pass the gradient through, as a straight-through
estimator does. It is computed one of two ways:

- **With a backward slot set**, in closed form, in the carrier, from the
  unrounded statistics. Softmax's is :math:`y \odot (g - \sum g \odot y)`,
  and a layer norm's

  .. math::

     \partial x = \tfrac{1}{\sigma}\Big(\partial\hat y - \operatorname{mean}(\partial\hat y)
                  - \hat x \operatorname{mean}(\partial\hat y \odot \hat x)\Big),
     \qquad \partial\hat y = g \odot \gamma,

  with no :math:`\operatorname{mean}(\partial\hat y)` for an RMS norm. The
  sums run on ``bwd_acc``'s mac, the products are rounded by ``bwd_mul``,
  softmax's difference by ``bwd_sub``, and each gradient (the input's, and
  :math:`\gamma`'s and :math:`\beta`'s, summed over the batch) by
  ``bwd_out``.
- **With none**, it is autograd's gradient of PyTorch's own function, word
  for word: the op runs PyTorch's function as well, and its rounded result
  passes the gradient on to it. That costs a second forward. One exception:
  compiled, PyTorch's CUDA batch norm is decomposed rather than run through
  cuDNN, so its gradient differs from eager's in the last places, in plain
  PyTorch too.

On MPS, PyTorch's own elementwise arithmetic flushes binary32 subnormals, so
a step whose carrier result is subnormal differs from the CPU's
(:doc:`/concepts`, "On an Apple GPU").
