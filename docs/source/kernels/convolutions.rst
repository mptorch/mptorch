Convolutions
============

A convolution layer runs three computations per step, the forward
convolution, the input gradient and the weight gradient, and each is a set of
dot products whose every product and every addition is rounded
(:doc:`arithmetic`). This page says which dot products those are, in which
order their terms are summed, and how :func:`~mptorch.quant.conv_formats`
runs them on the core routine of :doc:`index` without ever forming the
matrices a textbook lowering to a matrix product would build. The layer
guide (:doc:`/layers`, "Convolutions") has the worked example.

The three sums
--------------

Take a convolution in :math:`n_d \in \{1, 2, 3\}` spatial dimensions of an
input :math:`x` of :math:`C` channels with a weight :math:`W` of
:math:`C_{\text{out}}` filters over a kernel of extent :math:`k`, with stride
:math:`s`, padding :math:`p` and dilation :math:`d` (each a vector over the
spatial dimensions; products and comparisons of such vectors are per
dimension). Without groups, output position :math:`o` of filter :math:`c'` is

.. math::

   y[b, c', o] \;=\; \sum_{c=0}^{C-1} \sum_{i \in [0, k)} W[c', c, i]\;
   x[b, c, \, o s - p + i d]
   \;+\; \text{bias}[c'],

where :math:`x` reads zero outside its extent (the padding). With
:math:`G = \partial L / \partial y`, the two gradients are

.. math::

   \frac{\partial L}{\partial W}[c', c, i] &= \sum_{b} \sum_{o} G[b, c', o]\;
   x[b, c, \, o s - p + i d], \\[4pt]
   \frac{\partial L}{\partial x}[b, c, h] &= \sum_{c'} \sum_{\substack{j \in [0, k)\\
   o = (h + p - j d) / s \,\in\, \mathbb{Z}^{n_d} \cap [0,\, \text{out})}}
   W[c', c, j]\; G[b, c', o].

The last one is the sum over every (filter, tap) pair whose output position
:math:`o` was computed from input position :math:`h`. With ``groups`` =
:math:`g` the channels split into :math:`g` independent convolutions of
:math:`C/g` input and :math:`C_{\text{out}}/g` output channels each.

Each element of each result is one dot product, and the simulation rounds it
the way :doc:`arithmetic` rounds a matrix product's. For a ``SplitMac`` whose terms
are :math:`a_1 b_1, \dots, a_K b_K`,

.. math::

   s_0 = 0, \qquad s_t = Q_{\text{acc}}\big(s_{t-1} + Q_{\text{mul}}(a_t b_t)\big),
   \qquad \text{result} = s_K,

and for a ``FusedMac`` :math:`s_t = Q_{\text{fma}}(a_t b_t + s_{t-1})`;
``KAHAN``, ``BLOCK`` and ``TREE`` replace the recurrence with theirs (:doc:`arithmetic`,
"Summing differently"). A rounded sum depends on its order, so the order is
part of the definition, and it is fixed per pass:

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - pass
     - terms :math:`t = 1 \dots K`, in order
     - :math:`K`
   * - forward
     - :math:`(c, i)`, :math:`c` outer, the kernel positions row-major
       (``F.unfold``'s order)
     - :math:`\tfrac{C}{g} \prod k`
   * - weight gradient
     - :math:`(b, o)`, :math:`b` outer, the output positions row-major
     - :math:`B \prod \text{out}`
   * - input gradient
     - :math:`(c', j)`, :math:`c'` outer, the taps of the position's class
       *descending* (below)
     - :math:`\tfrac{C_{\text{out}}}{g}\,\lvert T_r \rvert`

Terms that read the padding are zeros, and they are in the sum: a position at
the border of the forward sums the same :math:`K` terms as one in the middle,
some of them :math:`W \cdot 0`.

Lowering to a matrix product, without the matrices
--------------------------------------------------

The forward is a matrix product. Unfold the input into the matrix
:math:`\tilde{X}_b` with one row per :math:`(c, i)` and one column per output
position,

.. math::

   \tilde{X}_b[(c, i),\, o] = x[b, c,\, o s - p + i d],
   \qquad
   y_b = W_{\text{flat}}\; \tilde{X}_b, \qquad
   W_{\text{flat}}[c', (c, i)] = W[c', c, i],

and the forward is the GEMM :math:`[C_{\text{out}}, K] \times [K, \text{OUT}]`
per sample, in the rounded arithmetic of the mac, in exactly the order of the
table above. The weight gradient is likewise
:math:`G_{\text{flat}}\,\tilde{X}^{\top}` with the samples stacked along
:math:`K`. The catch is size: :math:`\tilde{X}` has :math:`\prod k` entries
for every entry of :math:`x` (nine for a 3x3 kernel), which is what
``F.unfold`` allocates, per call, forward and backward.

``conv_formats`` runs these GEMMs on the core routine of :doc:`index` with
:math:`\tilde{X}` left implicit: the kernel stages a :math:`16 \times 16` tile
of each operand per step of its K-loop, and for a convolution it computes each
tile element's address in :math:`x` (or :math:`G`, or :math:`W`) from the
geometry and reads it in place, or substitutes zero where the element is
padding. Splitting an index into coordinates -- :math:`t \mapsto (c, i)`, a
column into the output position :math:`o` -- divides by the geometry's
extents, which are fixed for the call, so each division is a multiply-high and
a shift by constants computed once on the host; nothing is divided per
element. The arithmetic between the loads is the matrix product's, unchanged,
so every format, rounding mode and accumulate algorithm applies, and each pass
is **bit for bit the GEMM over the explicit matrices** --
``tests/test_qconv_gemm.py`` builds them and checks it with ``torch.equal``,
stochastic rounding included. What is gone is the memory: a pass allocates its
result and nothing else.

Groups do not multiply the launches: group :math:`\gamma` of sample
:math:`b` is element :math:`b g + \gamma` of the kernel's batch dimension, so
every group of every sample is one batch element of one launch, and each
result lands in its own layout (for the forward, :math:`[B, g,
C_{\text{out}}/g, \text{OUT}]` *is* :math:`[B, C_{\text{out}}, \text{OUT}]`).

The input gradient, class by class
----------------------------------

The input gradient is the one that is not a plain GEMM with a strided
convolution. The textbook route writes it as another convolution: insert
:math:`s - 1` zeros between the entries of :math:`G`, pad by
:math:`d(k - 1) - p`, and correlate with the flipped kernel. That is a GEMM
again, but a fraction :math:`1 - 1/\prod s` of its terms are the inserted
zeros -- three quarters at stride 2 in two dimensions -- and they cost what
real terms cost.

Which terms are real follows from the sum above. Tap :math:`j` reaches input
position :math:`h` when :math:`h + p - j d` is a multiple of :math:`s`, which
depends only on the residue :math:`r = (h + p) \bmod s`. So the input
positions split into :math:`\prod s` *classes*

.. math::

   H_r = \{\, h : (h + p) \bmod s = r \,\} = \{\, h_{0,r} + m s \,\}, \qquad
   T_r = \{\, j \in [0, k) : j d \equiv r \pmod{s} \,\},

and every position of a class takes the same taps. Per dimension :math:`T_r`
is empty unless :math:`\gcd(d, s)` divides :math:`r`, and otherwise every
:math:`s / \gcd(d, s)`-th tap. For :math:`h = h_{0,r} + m s` and
:math:`j \in T_r` the output position is

.. math::

   o = m + \frac{h_{0,r} + p - j d}{s},

an exact division, done once per tap on the host. The input gradient is then
one GEMM per class, :math:`[C/g,\; (C_{\text{out}}/g)\,\lvert T_r \rvert] \times
[(C_{\text{out}}/g)\,\lvert T_r \rvert,\; \lvert H_r \rvert]`, whose columns are
scattered to their positions of :math:`\partial L / \partial x`. Together they
do :math:`1/\prod s` of the zero-inserted work: at stride 2 in two dimensions
the four classes of a 3x3 kernel take 4, 2, 2 and 1 taps, nine in all, where
the zero-inserted sum takes nine for every position. At stride 1 there is one
class, every position and every tap, and it is the transposed convolution
itself. A class with no taps (a 1x1 kernel at stride 2) is a sum of nothing,
and its positions are zero. Within a class the taps run in descending
:math:`j`, the flipped kernel's order, so the terms that remain are in the
zero-inserted sum's order.

Leaving the zeros out is invisible where a zero term changes nothing. Under a
deterministic rounding a plain sum does not move: the product of a weight and
an inserted zero rounds to zero, and :math:`Q(s + 0) = s` because :math:`s`
is already a value of the format. So for ``NAIVE`` and every rounding mode
but ``SR`` the result is the zero-inserted one bit for bit (except that a
weight of :math:`\pm\infty` or NaN no longer turns the positions it does not
reach into NaN, as :math:`\infty \cdot 0` did). Elsewhere a zero term was not
free, and its absence changes the rounding, not the value being rounded:

- under ``SR`` each term spends a random draw, so the draws land on
  different terms;
- ``KAHAN`` applies its pending compensation on every step, zero or not;
- ``BLOCK`` and ``TREE`` count a block in terms. In the zero-inserted sum
  only one step in :math:`\prod s` was a term, so a block of ``block_size``
  steps held about ``block_size`` :math:`/ \prod s` terms and more of the sum
  ran in the outer format. Now a block holds ``block_size`` terms, as it does
  in the other two passes and in a matrix product.

Held to the float64 gradient, the class sum is as accurate as the
zero-inserted one: within 0.5% under ``SR`` and ``KAHAN``, and for ``BLOCK``
and ``TREE`` 1-3.5% *more* accurate than a zero-inserted sum with the same
number of terms per block (``block_size`` times :math:`\prod s`), which is
the like-for-like comparison. ``tests/test_qconv_gemm.py`` checks both.

Stochastic rounding
-------------------

Under ``SR`` each element of a GEMM draws from a stream of its own, keyed on
its index into the GEMM's :math:`[\text{batch}, M, N]` result, and a GEMM
takes one seed from the generator per call. A convolution
pass follows the same rule, so its draws depend on neither the tiling, the
thread count nor the device's launch geometry: the forward and weight
gradient are one GEMM call each, keyed on the result's own index; the input
gradient is one call per class, in row-major order of :math:`r`, keyed on the
element's index within its class. ``torch.manual_seed`` therefore reproduces
every pass.

Formats per element
-------------------

A palette mac (:doc:`arithmetic`, "Palettes") picks a format per element of each
pass's result through an integer map indexed like that result with the
spatial dimensions flattened: :math:`[C_{\text{out}}, \text{OUT}]` for the
forward, :math:`[C, \text{IN}]` for the input gradient and
:math:`[C_{\text{out}}, (C/g) \prod k]` for the weight gradient, each also as
:math:`[\cdot, 1]` (one format per channel), :math:`[1, \cdot]` (one per
position), or with a leading batch dimension. The classes of the input
gradient read the map at their own positions, so a map means the same thing
whatever the stride.

What it costs
-------------

Per pass the work is that of its GEMMs -- :math:`B\,C_{\text{out}}\,
\text{OUT}\,K` multiply-adds for the forward, the same for the weight
gradient, and :math:`1/\prod s` of the zero-inserted count for the input
gradient -- each at the rate of the matrix product in the same arithmetic,
which a few percent of address arithmetic in the tile loads does not change.
Memory is the result. Two shapes run below the kernel's best: a depthwise
convolution (one channel per group) has :math:`M = 1` and fills one row of
the :math:`16 \times 16` tile, and a class of the input gradient with few
positions or few taps is a small GEMM. The kernels compute in binary32 on the
CPU and on CUDA; a float64 model and MPS tensors raise.
