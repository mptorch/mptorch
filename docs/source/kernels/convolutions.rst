Convolutions
============

A convolution layer runs three computations per training step: the forward
convolution, the input gradient and the weight gradient. Each is a set of dot
products in which every product and every addition is rounded
(:doc:`arithmetic`). This page says which dot products those are and in
which order their terms are summed. It also explains how
:func:`~mptorch.quant.conv_formats` runs them on the core routine of
:doc:`index` without forming the matrices that a textbook lowering to a
matrix product would build. The layer guide (:doc:`/layers`,
"Convolutions") has the worked example.

The three sums
--------------

Take a convolution in :math:`n_d \in \{1, 2, 3\}` spatial dimensions. Its
input :math:`x` has :math:`C` channels, and its weight :math:`W` has
:math:`C_{\text{out}}` filters over a kernel of extent :math:`k`. It has
stride :math:`s`, padding :math:`p` and dilation :math:`d`. Each of these is
a vector over the spatial dimensions, and products and comparisons of such
vectors are per dimension. The batch has :math:`B` samples. The output's
spatial extent is the vector :math:`\text{out}`, and
:math:`\text{OUT} = \prod \text{out}` and :math:`\text{IN}` are the numbers
of output and input positions. Without groups, output position :math:`o` of
filter :math:`c'` is

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

The last one sums over every (filter, tap) pair whose output position
:math:`o` was computed from input position :math:`h`. With ``groups`` =
:math:`g` the channels split into :math:`g` independent convolutions, each
with :math:`C/g` input and :math:`C_{\text{out}}/g` output channels.

Each element of each result is one dot product, and the simulation rounds it
as :doc:`arithmetic` rounds a matrix product's. For a ``SplitMac`` whose
terms are :math:`a_1 b_1, \dots, a_K b_K`,

.. math::

   s_0 = 0, \qquad s_t = Q_{\text{acc}}\big(s_{t-1} + Q_{\text{mul}}(a_t b_t)\big),
   \qquad \text{result} = s_K,

and for a ``FusedMac`` :math:`s_t = Q_{\text{fma}}(a_t b_t + s_{t-1})`.
``KAHAN``, ``BLOCK`` and ``TREE`` replace the recurrence with their own
(:doc:`arithmetic`, "Summing differently"). A rounded sum depends on its
order, so the order is part of the definition. It is fixed per pass:

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
     - :math:`B \cdot \text{OUT}`
   * - input gradient
     - :math:`(c', j)`, :math:`c'` outer, the taps of the position's class
       *descending* (below)
     - :math:`\tfrac{C_{\text{out}}}{g}\,\lvert T_r \rvert`

Terms that read the padding are zeros, and they are part of the sum. In the
forward pass a position at the border sums the same :math:`K` terms as one in
the middle, with some of them equal to :math:`W \cdot 0`.

Lowering to a matrix product, without the matrices
--------------------------------------------------

The forward pass is a matrix product. Unfold the input into the matrix
:math:`\tilde{X}_b`, with one row per :math:`(c, i)` and one column per
output position:

.. math::

   \tilde{X}_b[(c, i),\, o] = x[b, c,\, o s - p + i d],
   \qquad
   y_b = W_{\text{flat}}\; \tilde{X}_b, \qquad
   W_{\text{flat}}[c', (c, i)] = W[c', c, i],

The forward pass is then the GEMM
:math:`[C_{\text{out}}, K] \times [K, \text{OUT}]` per sample, in the rounded
arithmetic of the mac and in exactly the order of the table above. The
weight gradient is likewise :math:`G_{\text{flat}}\,\tilde{X}^{\top}`, with
the samples stacked along :math:`K`.

The problem is size. :math:`\tilde{X}` has :math:`\prod k` entries for every
entry of :math:`x` (nine for a 3x3 kernel). ``F.unfold`` allocates that on
every call, forward and backward.

``conv_formats`` runs these GEMMs on the core routine of :doc:`index` and
leaves :math:`\tilde{X}` implicit. For each step of its K-loop the kernel
stages a :math:`16 \times 16` tile of each operand. For a convolution, it
computes the address in :math:`x` (or :math:`G`, or :math:`W`) of each tile
element from the geometry and reads the element in place, or uses zero where
the element is padding.

Splitting an index into coordinates (:math:`t \mapsto (c, i)`, or a column
into the output position :math:`o`) divides by the geometry's extents. These
are fixed for the call, so each division is a multiply-high and a shift by
constants computed once on the host. Nothing is divided per element.

The arithmetic between the loads is the matrix product's, unchanged. Every
format, rounding mode and accumulate algorithm therefore applies, and each
pass is **bit for bit the GEMM over the explicit matrices**.
``tests/test_qconv_gemm.py`` builds those matrices and checks this with
``torch.equal``, stochastic rounding included. The difference is in memory:
a pass allocates its result and nothing else.

Groups do not add launches. Group :math:`\gamma` of sample :math:`b` is
element :math:`b g + \gamma` of the kernel's batch dimension, so every group
of every sample is one batch element of a single launch. Each result lands
in its own layout: for the forward pass,
:math:`[B, g, C_{\text{out}}/g, \text{OUT}]` *is*
:math:`[B, C_{\text{out}}, \text{OUT}]`.

The input gradient, class by class
----------------------------------

With a strided convolution, the input gradient is the one pass that is not a
plain GEMM. The textbook approach writes it as another convolution: insert
:math:`s - 1` zeros between the entries of :math:`G`, pad by
:math:`d(k - 1) - p`, and correlate with the flipped kernel. That is again a
GEMM, but a fraction :math:`1 - 1/\prod s` of its terms are the inserted
zeros (three quarters at stride 2 in two dimensions), and they cost as much
as real terms.

The sum above tells which terms are real. Tap :math:`j` reaches input
position :math:`h` when :math:`h + p - j d` is a multiple of :math:`s`. That
depends only on the residue :math:`r = (h + p) \bmod s`. The input positions
therefore split into :math:`\prod s` *classes*

.. math::

   H_r = \{\, h : (h + p) \bmod s = r \,\} = \{\, h_{0,r} + m s \,\}, \qquad
   T_r = \{\, j \in [0, k) : j d \equiv r \pmod{s} \,\},

and every position of a class takes the same taps. Per dimension,
:math:`T_r` is empty unless :math:`\gcd(d, s)` divides :math:`r`; otherwise
it holds every :math:`s / \gcd(d, s)`-th tap. For :math:`h = h_{0,r} + m s`
and :math:`j \in T_r` the output position is

.. math::

   o = m + \frac{h_{0,r} + p - j d}{s},

which is an exact division, done once per tap on the host. The input
gradient is then one GEMM per class,
:math:`[C/g,\; (C_{\text{out}}/g)\,\lvert T_r \rvert] \times
[(C_{\text{out}}/g)\,\lvert T_r \rvert,\; \lvert H_r \rvert]`, whose columns
are scattered to their positions in :math:`\partial L / \partial x`.

Together the classes do :math:`1/\prod s` of the zero-inserted work. At
stride 2 in two dimensions, the four classes of a 3x3 kernel take 4, 2, 2
and 1 taps, nine in all, where the zero-inserted sum takes nine for every
position. Some special cases:

- At stride 1 there is one class, with every position and every tap. It is
  the transposed convolution itself.
- A class with no taps (a 1x1 kernel at stride 2) is an empty sum, and its
  positions are zero.

Within a class the taps run in descending :math:`j`, the flipped kernel's
order. The remaining terms are therefore in the same order as in the
zero-inserted sum.

Leaving the zeros out changes nothing where a zero term had no effect. Under
a deterministic rounding, a plain running sum is not moved by a zero term:
the product of a weight and an inserted zero rounds to zero, and
:math:`Q(s_{t-1} + 0) = s_{t-1}` because the running sum is already a value
of the format. So for ``NAIVE`` and every rounding mode except ``SR``, the
result is bit for bit the zero-inserted one. (One exception: a weight of
:math:`\pm\infty` or NaN no longer turns the positions it does not reach
into NaN, as :math:`\infty \cdot 0` did.)

In the other cases a zero term was not free. Leaving it out changes the
rounding, not the value being rounded:

- Under ``SR`` each term uses a random draw, so the draws now fall on
  different terms.
- ``KAHAN`` applies its pending compensation on every step, zero or not.
- ``BLOCK`` and ``TREE`` count a block in terms. In the zero-inserted sum
  only one step in :math:`\prod s` was a real term, so a block of
  ``block_size`` steps held about ``block_size`` :math:`/ \prod s` terms, and
  more of the sum ran in the outer format. Now a block holds ``block_size``
  terms, as in the other two passes and in a matrix product.

Compared with the float64 gradient, the class sum is as accurate as the
zero-inserted one. It is within 0.5% under ``SR`` and ``KAHAN``. For
``BLOCK`` and ``TREE`` it is 1-3.5% *more* accurate than a zero-inserted sum
with the same number of terms per block (``block_size`` times
:math:`\prod s`), which is the like-for-like comparison.
``tests/test_qconv_gemm.py`` checks both.

Stochastic rounding
-------------------

Under ``SR`` each element of a GEMM draws from its own stream, keyed on its
index into the GEMM's :math:`[\text{batch}, M, N]` result, and a GEMM takes
one seed from the generator per call. A convolution pass follows the same
rule, so its draws do not depend on the tiling, the thread count or the
device's launch geometry:

- The forward pass and the weight gradient are one GEMM call each, keyed on
  the result's own index.
- The input gradient is one call per class, in row-major order of :math:`r`,
  keyed on the element's index within its class.

``torch.manual_seed`` therefore reproduces every pass.

Formats per element
-------------------

A palette mac (:doc:`arithmetic`, "Palettes") picks a format per element of
each pass's result. It does so through an integer map indexed like that
result, with the spatial dimensions flattened:

- :math:`[C_{\text{out}}, \text{OUT}]` for the forward pass;
- :math:`[C, \text{IN}]` for the input gradient;
- :math:`[C_{\text{out}}, (C/g) \prod k]` for the weight gradient.

Each can also be given as :math:`[\cdot, 1]` (one format per channel), as
:math:`[1, \cdot]` (one per position), or with a leading batch dimension.
The classes of the input gradient read the map at their own positions, so a
map means the same thing whatever the stride.

What it costs
-------------

The work of a pass is that of its GEMMs:
:math:`B\,C_{\text{out}}\,\text{OUT}\,K` multiply-adds for the forward pass,
the same for the weight gradient, and :math:`1/\prod s` of the zero-inserted
count for the input gradient. Each runs at the rate of a matrix product in
the same arithmetic; the address arithmetic in the tile loads adds a few
percent. Memory use is the result.

Two shapes run below the kernel's best speed:

- A depthwise convolution (one channel per group) has :math:`M = 1` and
  fills only one row of the :math:`16 \times 16` tile.
- A class of the input gradient with few positions or few taps is a small
  GEMM.

The kernels compute in binary32 on the CPU and on CUDA. A float64 model and
MPS tensors raise.
