Dot-product arithmetic
======================

This page states what one step of the core routine rounds (the *mac*) and in
what order the terms of a dot product are summed. It gives the equations the
kernels follow and a run for each. The same arithmetic runs in every entry
point of :doc:`index`, and :doc:`calling` lists the objects that specify it.

What is simulated
-----------------

For :math:`C = AB` with :math:`A \in \mathbb{R}^{M \times K}` and
:math:`B \in \mathbb{R}^{K \times N}`, each output element is a dot product
of length :math:`K`, computed as a running sum in the order
:math:`k = 1, \dots, K`. Two arithmetics are implemented, named after the
kernel's own policies.

**SplitMac.** The product and the sum are rounded separately, to formats that
may differ:

.. math::

   s_0 = 0, \qquad
   s_k = Q_{\text{acc}}\Big(s_{k-1} + Q_{\text{mul}}(a_{ik}\, b_{kj})\Big), \qquad
   c_{ij} = s_K .

With ``acc=None`` the sum is left unrounded and only the products are
rounded: :math:`s_k = s_{k-1} + Q_{\text{mul}}(a_{ik} b_{kj})`.

**FusedMac.** Each step is one fused multiply-add, rounded once, as in a
hardware FMA unit:

.. math::

   s_k = Q_{\text{fma}}\big(a_{ik}\, b_{kj} + s_{k-1}\big).

The product is exact inside the FMA; the carrier still rounds the FMA's
result once before the format does (`What the carrier rounds`_, below). With
``fma=None`` nothing is rounded at all, and the dot product is a sequence of
FMAs.

In both arithmetics the operands :math:`a_{ik}` and :math:`b_{kj}` are read
as they are. Rounding *them* is a quantizer's job, and the layers apply one
before the GEMM (see :doc:`/layers`).

One rounding mode serves both roundings of a ``SplitMac``. The mode is a
compile-time parameter of the kernel, and rounding the product one way and
the sum another would multiply the number of kernels. Saturation, subnormal
handling and the number of stochastic bits are properties of each format and
can differ between the two.

Everything the formats do not round is computed in the operands' *carrier*:
the product before its rounding, the sum before its rounding, and an
unrounded accumulator. The carrier is binary32 for float32, float16 and
bfloat16 operands, and binary64 for float64 ones (`float64 and the
carrier`_, below).

The following run reproduces a ``SplitMac`` dot product by hand, step by
step, and gets the same bits.

.. literalinclude:: ../../snippets/gemm_splitmac.py
   :language: python
   :caption: docs/snippets/gemm_splitmac.py

.. literalinclude:: ../../snippets/gemm_splitmac.out
   :language: text
   :caption: output

The next run does the same for ``FusedMac``, using float64 to hold the exact
product-and-add before the single rounding:

.. literalinclude:: ../../snippets/gemm_fusedmac.py
   :language: python
   :caption: docs/snippets/gemm_fusedmac.py

.. literalinclude:: ../../snippets/gemm_fusedmac.out
   :language: text
   :caption: output

The accumulation format usually matters most. Over a long dot product the
rounding errors of the partial sums build up, and a format with three
mantissa bits cannot usefully hold a sum of thousands of terms:

.. literalinclude:: ../../snippets/gemm_accumulation.py
   :language: python
   :caption: docs/snippets/gemm_accumulation.py

.. literalinclude:: ../../snippets/gemm_accumulation.out
   :language: text
   :caption: output

The table shows three things:

- Rounding the products to Binary8p4 costs a relative error of a few percent
  whatever the accumulator, because each product carries the format's
  :math:`2^{-4}` relative error. A float32 accumulator (``acc=None``, or the
  same thing spelled as a 24-bit format) adds nothing to it.
- A Binary16p8 accumulator, which has bfloat16's eight bits of precision, is
  noticeably worse, and a Binary8p4 one is useless.
- Stochastic rounding does *not* help here; it is worse than
  round-to-nearest. The terms have random signs, so there is no systematic
  bias to remove, only variance to add. Stochastic rounding helps when the
  increments are systematically below half a spacing. That is the
  accumulator case shown in :doc:`/concepts` and the weight-update case in
  the :doc:`/tutorial`.

The two formats of a ``SplitMac`` may come from different families:
:math:`Q_{\text{mul}}` a :class:`~mptorch.BinaryK` and :math:`Q_{\text{acc}}`
a :class:`~mptorch.SuperFP`, or the reverse. The equation is the one above,
each rounding in its own format, and the result holds the accumulate format's
values. Such a mac holds one format per slot (the palette kernels are
single-family), and its sum is always rounded (with ``acc=None`` the mac has
one family again). Every algorithm of the next section applies, with an
``outer`` format of the accumulate's family, in either carrier, and so does
the odd carrier step (`What the carrier rounds`_) under ``NAIVE``.

.. literalinclude:: ../../snippets/gemm_cross_family.py
   :language: python
   :caption: docs/snippets/gemm_cross_family.py

.. literalinclude:: ../../snippets/gemm_cross_family.out
   :language: text
   :caption: output

What the carrier rounds
-----------------------

The equations above name the formats' roundings. Each step also has one they
do not name: the carrier computes the product, the sum and the FMA to its own
precision first, 24 bits in binary32 and 53 in binary64, rounding to nearest,
and the format's cast then rounds that. Usually the second rounding absorbs
the first. It does not when the exact value lies within the carrier's last
place of a format value: the carrier rounds it onto that value, and a
directed mode then keeps it instead of going to the neighbour it should have
gone to. ``1 + 2^-30`` rounded up to eight bits is ``1 + 2^-7``, but its
binary32 sum is exactly 1, which every mode keeps. binary64 holds that gap,
but not ``2^-61``:

.. literalinclude:: ../../snippets/gemm_carrier_rounding.py
   :language: python
   :caption: docs/snippets/gemm_carrier_rounding.py

.. literalinclude:: ../../snippets/gemm_carrier_rounding.out
   :language: text
   :caption: output

On random data the effect follows the formats' precision.
``dev/benchmarks/timing/carrier_double_rounding.py rates`` counts the outputs on
which the binary32 and binary64 carriers disagree, at M = N = 256 over N(0, 1)
operands quantized to the format. At K = 256 FP8 formats disagree on at most
0.04% of outputs, and formats of up to eleven bits of precision on at most
0.06% (split) and 0.5% (fused), under the directed modes. The widest
disagree most: Binary16p12 fused on 1.4%, Binary24p16 on 4.6% split and 22%
fused. A fused product of two P-bit operands has 2P bits before the add, more
than binary32 keeps.

**Rounding the carrier step to odd.** ``carrier_rounding=RoundMode.RO`` on a
:class:`~mptorch.quant.SplitMac` or :class:`~mptorch.quant.FusedMac` (and on
the flat functions and the ``*_gemm_formats`` layer factories) rounds the
carrier's product, sum and FMA to odd instead. This is round toward zero onto
the carrier's grid with the last bit set whenever the result is inexact. An
odd result stays strictly between the two carrier values next to the exact
one, so every format whose rounding boundaries are all even carrier values
rounds it exactly as it would the exact value (Boldo and Melquiond, "When
double rounding is odd"). In terms of the format's precision ``P``:

========================= ======================== =========================
the format's rounding     binary32 (p = 24)        binary64 (p = 53)
========================= ======================== =========================
``RNE``, ``RNA``          ``P <= 22``              ``P <= 51``
``RU``, ``RD``, ``RZ``,   ``P <= 23``              ``P <= 52``
``RO``
``SR``                    ``P + prng_bits <= 23``  ``P + prng_bits <= 52``
========================= ======================== =========================

The midpoints that ``RNE`` and ``RNA`` round at take the extra bit. The bound
holds below the carrier's normal range and for saturating formats, but not
for ``SR`` below the format's smallest value, whose arms compute in the
carrier before they draw. A call whose format is outside the bound raises.
In binary32 the error names ``carrier=torch.float64`` where binary64 holds
the format.

Two results change beyond the corrected neighbours:

- A finite exact sum larger than the carrier's largest value reaches the cast
  as that value instead of as infinity, so ``SAT_PROPAGATE`` saturates it, as
  P3109 says a finite value should.
- With ``acc=None`` only the product is rounded to odd: no format rounds the
  sum after it, so it stays the carrier's nearest add.

The odd step is implemented for every ``NAIVE`` reduction: the single-format
GEMMs, the palettes (:class:`~mptorch.quant.Palette`, each entry held to the
bound) and the cross-family macs (both formats held to it), split and fused,
and the three passes of a convolution over any of them
(:func:`~mptorch.quant.conv_formats`; :doc:`convolutions`), on CPU and CUDA
tensors in both carriers. The ``custom_matmul_*_odd`` and
``custom_conv_*_odd`` ops are separate, so the default ones are what they
were, and a build with ``MPTORCH_NO_ODD=1`` leaves their kernels out (the ops
then raise, naming the flag). ``KAHAN``, ``BLOCK`` and ``TREE`` rely on
error-free transforms rounded to nearest, and refuse it. So does an Apple
GPU. On the CPU it assumes the default floating-point environment; a
flush-to-zero mode (``torch.set_flush_denormal(True)``) breaks it in
subnormal ranges, as it does the carrier arithmetic everywhere else.

It costs time, mostly on the CPU (``carrier_double_rounding.py bench``,
measured on a Ryzen 9 7940HS and an RTX 4060 laptop GPU):

============================== ===================== =====================
time against the nearest step  fused                 split
============================== ===================== =====================
CUDA, binary32                 1.05-1.16x            1.05-1.14x
CUDA, binary64 (GeForce)       1.1x (SR) - 2.0x (RU) 1.1x (SR) - 1.5x
CPU, binary32, vector tiles    1.2x                  0.9-1.0x
CPU, binary32, scalar loop     1.2-1.8x              1.4-2.4x
CPU, binary64                  1.4-2.7x              1.2-2.1x
============================== ===================== =====================

The palettes, the cross-family macs and the convolutions cost what the GEMM
of their step costs: 1.03-1.14x on CUDA, and on the CPU the scalar loop's
ratio, or the tile's for a convolution over a format the tiles take.

On a GPU, binary32 rounds to odd with two FMAs in the hardware's directed
modes and a select. The binary64 kernels of a consumer GPU are limited by its
FP64 rate, which the second FMA and the directed modes' own FP64 compares
share. The CPU's scalar loop has no directed rounding to call, so each step is
a sequence of error-free transforms on the sum's dependency chain: a binary64
sum and an exact correction in binary32, and in binary64 an exact product
split, two sums and a fix of the last bit. The vector tiles take the cases the
default op runs in vector code (binaryK in binary32 under ``RNE``, one format
per slot: :doc:`calling`, "Performance notes") and run a whole tile's step at
once: with AVX-512 through its embedded directed rounding, as the GPU does, at
about the default tile's cost; with AVX2 or NEON through the binary64
transforms, at 3-5x it. Before them the odd step of those cases ran the scalar
loop, 15-22x the default tile.

Summing differently: KAHAN, BLOCK and TREE
------------------------------------------

A narrow accumulator does badly in the table above because of the *order* of
the sum as much as its width. There is one running sum, rounded :math:`K`
times, each time against a total that has grown far larger than the term
being added. ``accumulate_algorithm`` changes the order and keeps the
formats.

Take one output element and drop its indices :math:`i, j`, so that its terms
are :math:`a_k b_k` for :math:`k = 1, \dots, K`. Let
:math:`p_k = Q_{\text{mul}}(a_k b_k)` be the rounded product, :math:`s` the
running sum, :math:`Q_{\text{acc}}` the accumulate rounding and
:math:`Q_{\text{out}}` a third rounding, the mac's ``outer`` format. Each is
the identity when its format is ``None``. The algorithms are:

:attr:`~mptorch.AccumulateAlgorithm.NAIVE`
   :math:`s \leftarrow Q_{\text{acc}}(s + p_k)`, the recurrence above.
:attr:`~mptorch.AccumulateAlgorithm.KAHAN`
   Kahan's compensated sum. The compensation :math:`c` is held in the
   accumulate format, like the sum:
   :math:`y = Q_{\text{acc}}(p_k - c)`, :math:`t = Q_{\text{acc}}(s + y)`,
   :math:`c \leftarrow Q_{\text{acc}}(Q_{\text{acc}}(t - s) - y)`,
   :math:`s \leftarrow t`. This is four accumulate roundings per step, where
   ``NAIVE`` has one.
:attr:`~mptorch.AccumulateAlgorithm.BLOCK`
   Two levels. The ``NAIVE`` recurrence runs on a block's sum. After every
   ``block_size`` products, and after the last one, the block is added to a
   total :math:`T`, :math:`T \leftarrow Q_{\text{out}}(T + s)`, and restarted
   from zero. A block's sum stays small, so a narrow ``acc`` loses little in
   it. ``outer`` (wider, or ``None`` for the carrier) is rounded once per
   block instead of once per product. ``block_size`` divides 16 or is a
   multiple of it.
:attr:`~mptorch.AccumulateAlgorithm.TREE`
   The products of a block of ``block_size`` :math:`= 2^L` (up to 256, 16 by
   default) are summed pairwise:
   :math:`Q_{\text{acc}}(Q_{\text{acc}}(p_1 + p_2) + Q_{\text{acc}}(p_3 + p_4))`
   and so on up. Every addition is then of two terms of similar size. The
   block's root is added to the total as in ``BLOCK``. This is for
   ``SplitMac`` only: a fused multiply-add has no product term to pair.

In ``KAHAN`` and ``BLOCK``, a fused mac's step,
:math:`Q_{\text{fma}}(s + a_k b_k)`, replaces the split one. A last partial
block is added like a whole one, and nothing is padded. The CPU and CUDA
kernels return the same bits in every deterministic mode. The result holds
the values of the last rounding applied, which is ``outer``'s when there is
one. :class:`~mptorch.AccumulateAlgorithm` states each algorithm step by
step, including the draws of ``RoundMode.SR``.

.. literalinclude:: ../../snippets/gemm_kahan_block_tree.py
   :language: python
   :caption: docs/snippets/gemm_kahan_block_tree.py

.. literalinclude:: ../../snippets/gemm_kahan_block_tree.out
   :language: text
   :caption: output

In float32 the three behave as in any numerical library: ``KAHAN`` recovers
about a digit and a half over a 4096-term sum, and blocking about half of
that. The other two tables show them with formats:

- **Second table.** An accumulator with eight bits of precision costs
  ``NAIVE`` three times the error that the Binary8p4 products already carry.
  Every other row is back at the products' own error, which no summation can
  improve on.
- **Third table.** This is the narrowest case, sums in Binary8p4 itself.
  ``NAIVE`` is useless, and the other three are within a factor of two of the
  products' error. An ``outer`` format *narrower* than the carrier costs a
  little, as expected: it is one more rounding, and in exchange the total is
  a value of a format you chose.
- **Fourth table.** The second, in binary64 (``carrier=torch.float64``). A
  product of two float32 operands is exact there, so the multiply format
  rounds it once instead of twice. The errors are the second table's to three
  digits: the formats round every intermediate far more than binary32 does.

The price is kernel time, in proportion to the roundings per step. On an RTX
4060 laptop GPU a 1024\ :sup:`3` split-mac GEMM takes 1.2 times ``NAIVE``'s
time under ``BLOCK``, 1.4 under ``TREE`` and 2.25 under ``KAHAN``
(``BinaryK(8, 4)`` products into ``BinaryK(16, 11)`` sums under ``RNE``,
November 2026). The three are implemented for the
single-format ops, on the CPU, CUDA and Apple GPUs, in both carriers (next
section) where the device has float64. A palette raises.

float64 and the carrier
-----------------------

A float64 GEMM is computed in binary64 from end to end: every product, every
sum and every rounding. So is a float32, float16 or bfloat16 GEMM that names
``carrier=torch.float64``. This is a different arithmetic from the binary32
one, not the same one widened, in three ways:

- **Products of narrow operands are rounded once.** In binary32 a product is
  rounded to 24 bits before the multiply format rounds it again. In binary64
  the product of two operands of at most 26 significant bits is exact, and
  only the format rounds it. Two float32 values qualify, as do two values a
  quantizer has rounded. A binary64 GEMM can therefore round differently
  even when every operand is a float32 value, as the run's first lines show.
  It agrees with the binary32 GEMM exactly when every product and sum is
  exact in both. Operands already rounded to a narrow format usually give
  that (the FP8 recipe in :doc:`/layers` agrees to the bit).
- **Wider formats.** Each call holds the formats to its carrier's bounds
  (:doc:`/concepts`). Formats beyond binary32, such as a 30-bit multiply
  format into a 40-bit accumulator, are refused on float32 operands and
  computed exactly on float64 ones.
- **Wider random draws.** Under ``SR`` each rounding draws two words of its
  element's stream, and ``P - 1 + prng_bits`` may reach 52.

``carrier=torch.float64`` on a ``SplitMac`` / ``FusedMac``, or on the flat
function, computes narrower operands the binary64 way. Both operands are
widened, the binary64 kernel runs, and the result is narrowed back to their
dtype with one rounding. This is bit for bit the float64 call on the widened
operands, random draws included, then stored.

The stored result is held to what that dtype can store (:doc:`/concepts`). A
float32 result is held to 24 bits of precision in its accumulate or fused
format. For float32 operands, binary64 therefore improves everything
*before* the last rounding: an unquantized sum no longer loses a bit per
step, and the multiply format can be wider than the result. The run measures
this.

The carrier is never narrower than the operands. ``carrier=torch.float32``
on float64 operands raises, and a float32 and a float64 operand cannot be
mixed in one call under either carrier.

Every accumulate algorithm runs in binary64 as ``NAIVE`` does, and so do the
cross-family macs and the convolutions (:doc:`convolutions`). A ``KAHAN``
compensation in binary64 carries the bits a binary64 sum drops.

.. literalinclude:: ../../snippets/gemm_float64.py
   :language: python
   :caption: docs/snippets/gemm_float64.py

.. literalinclude:: ../../snippets/gemm_float64.out
   :language: text
   :caption: output

In the float64 table, ``KAHAN`` adds nothing to a 40-bit accumulator, which
already keeps what the 30-bit products carry, and a cross-family mac (a
superfp multiply into a 40-bit binaryK sum) runs in binary64 like the
single-family ones.

The float32 table shows the other side of the choice. Where the formats round
every intermediate the carrier would (24 bits, or Binary8p4's four on either
side), the two carriers are within one rounding of each other or equal to
the bit. Beyond what the formats keep, the carrier adds little. On a GPU it
also costs time (see :doc:`calling`, "Performance notes"), in addition to
the float64 copies of the operands and the result.

Palettes: a format per output element
-------------------------------------

A ``Palette`` in any format slot selects the *spatially-varying* kernel. Each
output element's dot product runs in the palette entry that an integer map,
``prec_idx``, assigns to it. The map can have:

- the output's shape ``[M, N]``;
- shape ``[M, 1]``, for one format per row;
- shape ``[1, N]``, for one format per column.

It can also have a leading batch dimension, of the batch size or of size 1
to share the map. Its values must lie in ``[0, len(palette))``. A palette's
entries must agree on everything except their widths (sign, stochastic bits,
saturation and subnormal policy), because the kernel tabulates only the
widths.

.. literalinclude:: ../../snippets/gemm_palette.py
   :language: python
   :caption: docs/snippets/gemm_palette.py

.. literalinclude:: ../../snippets/gemm_palette.out
   :language: text
   :caption: output

A format for every element
~~~~~~~~~~~~~~~~~~~~~~~~~~

A dense ``[M, N]`` map is the general case. Every output element's dot
product runs in the palette entry named by its own map entry, independently
of its row, its column and its neighbours. How the map is chosen is up to the
caller: a sensitivity analysis, a hardware assignment, a search.

The run below builds one per element. For each of the 20 elements of a
product it takes the narrowest of eight BinaryK formats, from 3 to 24 bits of
precision, whose result stays within 0.1% of the exact dot product there. It
then runs the whole product as a single palette GEMM and checks every element
against the single-format GEMM in that element's format. They are equal to
the bit, because an element's dot product reads only its own row and column
of the operands and its own palette entry.

The same map picks a *pair* when the multiply and the accumulate palettes
differ. Entry ``i`` of a ``SplitMac`` palette pairs ``mul[i]`` with
``acc[i]``, so one element can multiply in 4 bits into a 24-bit sum while its
neighbour does the opposite. A ``FusedMac`` takes a palette the same way. A
batched product takes a ``[B, M, N]`` map, with a format per element of every
sample. The fields a palette shares (sign, stochastic bits, saturation and
subnormals) are shared by every element; a palette whose entries disagree on
one is refused, with an error naming the entry.

.. literalinclude:: ../../snippets/gemm_palette_elements.py
   :language: python
   :caption: docs/snippets/gemm_palette_elements.py

.. literalinclude:: ../../snippets/gemm_palette_elements.out
   :language: text
   :caption: output

