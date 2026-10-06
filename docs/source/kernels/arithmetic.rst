Dot-product arithmetic
======================

This page states what one step of the core routine rounds -- the *mac* --
and in what order the terms of a dot product are summed, with the equations
the kernels follow and a run for each. The same arithmetic runs in every
entry point of :doc:`index`; :doc:`calling` lists the objects that name it.

What is simulated
-----------------

For :math:`C = AB` with :math:`A \in \mathbb{R}^{M \times K}` and
:math:`B \in \mathbb{R}^{K \times N}`, each output element is a dot product
of length :math:`K`, computed as a running sum in order
:math:`k = 1, \dots, K`. Two arithmetics are implemented, named after the
kernel's own policies.

**SplitMac** -- the product and the sum are rounded separately, to formats
that may differ:

.. math::

   s_0 = 0, \qquad
   s_k = Q_{\text{acc}}\Big(s_{k-1} + Q_{\text{mul}}(a_{ik}\, b_{kj})\Big), \qquad
   c_{ij} = s_K .

With ``acc=None`` the accumulation is left unrounded and only the products
are: :math:`s_k = s_{k-1} + Q_{\text{mul}}(a_{ik} b_{kj})`.

**FusedMac** -- one fused multiply-add per step, rounded once, the way a
hardware FMA unit behaves:

.. math::

   s_k = Q_{\text{fma}}\big(a_{ik}\, b_{kj} + s_{k-1}\big).

The product is exact inside the FMA (the FMA the kernel uses keeps it so),
and with ``fma=None`` nothing is rounded at all: the dot product is a
sequence of FMAs.

In both, the operands :math:`a_{ik}` and :math:`b_{kj}` are read as they
are. Rounding *them* is a quantizer's job, and the layers apply one before
the GEMM; see :doc:`/layers`. One rounding mode serves both roundings of a
``SplitMac`` -- it is a compile-time parameter of the kernel, and rounding
the product one way and the sum another would multiply the number of
kernels -- while saturation, subnormal handling and the number of stochastic
bits are properties of each format and can differ between the two.

Everything the formats do not round -- the product before its rounding, the
sum before its, the unrounded accumulator -- is computed in the operands'
*carrier*: binary32 for float32, float16 and bfloat16 operands, binary64 for
float64 ones (`float64 and the carrier`_, below).

The following run reproduces a ``SplitMac`` dot product by hand, step by
step, and gets the same bits.

.. literalinclude:: ../../snippets/gemm_splitmac.py
   :language: python
   :caption: docs/snippets/gemm_splitmac.py

.. literalinclude:: ../../snippets/gemm_splitmac.out
   :language: text
   :caption: output

The same for ``FusedMac``, using float64 to hold the exact product-and-add
before the single rounding:

.. literalinclude:: ../../snippets/gemm_fusedmac.py
   :language: python
   :caption: docs/snippets/gemm_fusedmac.py

.. literalinclude:: ../../snippets/gemm_fusedmac.out
   :language: text
   :caption: output

The accumulation format is usually the one that matters. Over a long dot
product the rounding of each partial sum compounds, and a format with three
mantissa bits cannot hold a sum of thousands of terms in any useful way:

.. literalinclude:: ../../snippets/gemm_accumulation.py
   :language: python
   :caption: docs/snippets/gemm_accumulation.py

.. literalinclude:: ../../snippets/gemm_accumulation.out
   :language: text
   :caption: output

Three things to read off this table. Rounding the products to Binary8p4 costs a
relative error of a few percent whatever the accumulator, because each
product carries the format's :math:`2^{-4}` relative error; a float32
accumulator (``acc=None``, or the same thing spelled as a 24-bit format) adds
nothing to it. A Binary16p8 accumulator, which has bfloat16's eight bits of
precision, is noticeably worse and a Binary8p4 one is useless. And stochastic rounding does *not* help here -- it is worse than
round-to-nearest -- because the terms have random signs: there is no
systematic bias to remove, only variance to add. Stochastic rounding earns
its keep when the increments are systematically below half a spacing, which
is the accumulator case shown in :doc:`/concepts` and the weight-update case
in the :doc:`/tutorial`.

Summing differently: KAHAN, BLOCK and TREE
------------------------------------------

What the table above charges a narrow accumulator for is the *order* of the
sum as much as its width: one running sum, rounded :math:`K` times, each time
against a total that has grown far past the term being added.
``accumulate_algorithm`` changes the order and keeps the formats. Take one
output element and drop its indices :math:`i, j`, so that its terms are
:math:`a_k b_k` for :math:`k = 1, \dots, K`. With
:math:`p_k = Q_{\text{mul}}(a_k b_k)` the rounded product, :math:`s` the
running sum, :math:`Q_{\text{acc}}` the accumulate rounding and
:math:`Q_{\text{out}}` a third one, the mac's ``outer`` format (each the
identity when its format is ``None``):

:attr:`~mptorch.AccumulateAlgorithm.NAIVE`
   :math:`s \leftarrow Q_{\text{acc}}(s + p_k)`, the recurrence above.
:attr:`~mptorch.AccumulateAlgorithm.KAHAN`
   Kahan's compensated sum, with the compensation :math:`c` held in the
   accumulate format like the sum:
   :math:`y = Q_{\text{acc}}(p_k - c)`, :math:`t = Q_{\text{acc}}(s + y)`,
   :math:`c \leftarrow Q_{\text{acc}}(Q_{\text{acc}}(t - s) - y)`,
   :math:`s \leftarrow t`. Four accumulate roundings a step where ``NAIVE``
   has one.
:attr:`~mptorch.AccumulateAlgorithm.BLOCK`
   Two levels. The ``NAIVE`` recurrence runs on a block's sum; after every
   ``block_size`` products, and after the last, the block is folded into a
   total :math:`T`, :math:`T \leftarrow Q_{\text{out}}(T + s)`, and restarted from
   zero. A block's sum stays small, so a narrow ``acc`` loses little in it,
   and ``outer`` -- wider, or ``None`` for the carrier -- is rounded once per
   block rather than once per product. ``block_size`` divides 16 or is a
   multiple of it.
:attr:`~mptorch.AccumulateAlgorithm.TREE`
   The products of a block of ``block_size`` :math:`= 2^L` (up to 256, 16 by
   default) are summed pairwise,
   :math:`Q_{\text{acc}}(Q_{\text{acc}}(p_1 + p_2) + Q_{\text{acc}}(p_3 + p_4))`
   and so on up, so that every addition is of two terms of like size, and the
   block's root is folded into the total as in ``BLOCK``. ``SplitMac`` only:
   a fused multiply-add has no product term to pair.

A fused mac's step, :math:`Q_{\text{fma}}(s + a_k b_k)`, stands in for the split
one in ``KAHAN`` and ``BLOCK``. A last partial block is folded like a whole
one, nothing is padded, and the CPU and CUDA kernels return the same bits in
every deterministic mode; the result holds the values of the last rounding
applied, which is ``outer``'s when there is one.
:class:`~mptorch.AccumulateAlgorithm` states each algorithm to the step,
``RoundMode.SR``'s draws included.

.. literalinclude:: ../../snippets/gemm_kahan_block_tree.py
   :language: python
   :caption: docs/snippets/gemm_kahan_block_tree.py

.. literalinclude:: ../../snippets/gemm_kahan_block_tree.out
   :language: text
   :caption: output

In float32 the three are what they are in any numerical library: ``KAHAN``
recovers about a digit and a half over a 4096-term sum, and blocking about
half of that. With formats, the second table is the one to read. An
accumulator of eight bits of precision costs ``NAIVE`` three times the error
the Binary8p4 products already carry; every other row is back at the products'
own error, which no summation can go below. The third table is the narrowest
case, sums in Binary8p4 itself, where ``NAIVE`` is useless and the other three are
within a factor of two of the products' error -- while an ``outer`` format
*narrower* than the carrier costs a little, as it should: it is one more
rounding, and what it buys is that the total is a value of a format you
chose.

The price is kernel time, in proportion to the roundings per step: on an RTX
4060 laptop GPU a 1024\ :sup:`3` split-mac GEMM takes 1.2 times ``NAIVE``'s
time under ``BLOCK``, 1.3 under ``TREE`` and 2.2 under ``KAHAN``
(``dev/gemm_roadmap.md``, R-2). The three are implemented for the
single-format ops, on the CPU and CUDA: a palette, an ``"mps"`` tensor,
float64 operands and ``carrier=torch.float64`` raise, the last two because
the binary64 kernels are not built yet (next section).

float64 and the carrier
-----------------------

A float64 GEMM is computed in binary64 end to end: every product, every sum
and every rounding -- and so is a float32, float16 or bfloat16 one that names
``carrier=torch.float64``. That makes it a different arithmetic from the
binary32 one, not the same one widened, in three ways.

- **Products of narrow operands are rounded once.** In binary32 a product is
  rounded to 24 bits before the multiply format rounds it again; in binary64
  the product of two operands of at most 26 significant bits -- two float32
  values, or two values a quantizer has rounded -- is exact, and only the
  format rounds it. So a binary64 GEMM can round differently even when every
  operand is a float32 value, as the run's first lines show, and agrees with
  the binary32 GEMM exactly when every product and sum is exact in both --
  which operands already rounded to a narrow format usually give (the FP8
  recipe in :doc:`/layers` agrees to the bit).
- **Wider formats.** Each call holds the formats to its carrier's bounds
  (:doc:`/concepts`), so formats past binary32 -- a 30-bit multiply format
  into a 40-bit accumulator, say -- are refused on float32 operands and
  computed exactly on float64 ones.
- **Wider random draws.** Under ``SR`` each rounding draws two words of its
  element's stream, and ``P - 1 + prng_bits`` may reach 52.

``carrier=torch.float64`` on a ``SplitMac`` / ``FusedMac``, or on the flat
function, computes narrower operands the binary64 way: both operands are
widened, the binary64 kernel runs, and the result is narrowed back to their
dtype with one rounding -- bit for bit the float64 call on the widened
operands, random draws included, then stored. The stored result is held to
what that dtype can store (:doc:`/concepts`): a float32 one to 24 bits of
precision in its accumulate or fused format. So for float32 operands binary64
buys everything *before* the last rounding -- an unquantized sum that does not
lose a bit a step, a multiply format wider than the result -- and the run
measures it. The carrier is never narrower than the operands:
``carrier=torch.float32`` on float64 operands raises, and a float32 and a
float64 operand cannot be mixed in one call with either.

One thing binary64 does not have yet is ``KAHAN``, ``BLOCK`` and ``TREE``:
their kernels are binary32's only, so a mac that names one of them refuses
float64 operands and ``carrier=torch.float64``, with an error that says so,
rather than quietly summing in the order it was not asked for.

.. literalinclude:: ../../snippets/gemm_float64.py
   :language: python
   :caption: docs/snippets/gemm_float64.py

.. literalinclude:: ../../snippets/gemm_float64.out
   :language: text
   :caption: output

The float32 table shows the other side of the choice. Where the formats round
every intermediate the carrier would -- 24 bits, or Binary8p4's four on either side -- the
two carriers are within a rounding of each other or equal to the bit: past what
the formats keep, the carrier adds little, and on a GPU it costs (see
:doc:`calling`, "Performance notes"), besides the float64 copies of the
operands and the result.


Palettes: a format per output element
-------------------------------------

A ``Palette`` in any format slot selects the *spatially-varying* kernel: each
output element's dot product runs in the palette entry that an integer map,
``prec_idx``, assigns to it. The map has the output's shape ``[M, N]``, or
``[M, 1]`` for one format per row, or ``[1, N]`` for one per column,
optionally with a leading batch dimension (of the batch size, or 1 to share
the map). Values must lie in ``[0, len(palette))``. A palette's entries must
agree on everything but their widths -- sign, stochastic bits, saturation and
subnormal policy -- because the kernel tabulates only the widths.

.. literalinclude:: ../../snippets/gemm_palette.py
   :language: python
   :caption: docs/snippets/gemm_palette.py

.. literalinclude:: ../../snippets/gemm_palette.out
   :language: text
   :caption: output

A format for every element
~~~~~~~~~~~~~~~~~~~~~~~~~~

A dense ``[M, N]`` map is the general case: every output element's dot
product runs in the palette entry its own map entry names, independently of
its row, its column and its neighbours. How the map is chosen is up to the
caller -- a sensitivity analysis, a hardware assignment, a search. The run
below makes one per element: for each of the 20 elements of a product it takes
the narrowest of eight BinaryK formats, from 3 to 24 bits of precision, whose
result stays within 0.1% of the exact dot product there. It then runs the
whole product as a single palette GEMM and checks every element against the
single-format GEMM in that element's format -- equal, to the bit, because
an element's dot product reads only its own row and column of the operands
and its own palette entry.

The same map picks a *pair* when the multiply and the accumulate palettes
differ: entry ``i`` of a ``SplitMac`` palette pairs ``mul[i]`` with ``acc[i]``,
so one element can multiply in 4 bits into a 24-bit sum while its neighbour
does the opposite. A ``FusedMac`` takes a palette the same way, a batched
product takes a ``[B, M, N]`` map with a format per element of every sample,
and the fields a palette shares -- sign, stochastic bits, saturation and
subnormals -- are shared by every element; a palette whose entries disagree
on one is refused, naming the entry.

.. literalinclude:: ../../snippets/gemm_palette_elements.py
   :language: python
   :caption: docs/snippets/gemm_palette_elements.py

.. literalinclude:: ../../snippets/gemm_palette_elements.out
   :language: text
   :caption: output

