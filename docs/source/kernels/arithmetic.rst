Dot-product arithmetic
======================

This page covers two things: what one multiply-accumulate step of the core
routine rounds (the *mac*), and in what order the terms of a dot product are
summed. It gives the equations the kernels follow and a run for each. The
same arithmetic runs in every entry point of :doc:`index`, and
:doc:`calling` lists the objects that specify it.

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

The product is exact inside the FMA. The *carrier*, the IEEE binary float
that the kernel computes in, still rounds the FMA's result once before the
format does (`What the carrier rounds`_, below). With ``fma=None`` no format
rounds anything, and the dot product is a sequence of the carrier's own
FMAs.

In both arithmetics the operands :math:`a_{ik}` and :math:`b_{kj}` are read
as they are. Rounding *them* is a quantizer's job, and the layers apply one
before the GEMM (see :doc:`/layers`).

One rounding mode serves both roundings of a ``SplitMac``. The mode is a
compile-time parameter of the kernel, and rounding the product in one mode
and the sum in another would multiply the number of kernels. Saturation,
subnormal handling and the number of stochastic bits are properties of each
format, so they can differ between the two.

Everything the formats do not round is computed in the carrier: the product
before its rounding, the sum before its rounding, and an unrounded
accumulator. By default the carrier is binary32 for float32, float16 and
bfloat16 operands, and binary64 for float64 ones. A call can also ask for
binary64 on narrower operands (`float64 and the carrier`_, below).

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

- Rounding the products to Binary8p4 alone costs a relative error of a few
  percent, because each product carries the format's :math:`2^{-4}` relative
  error. A float32 accumulator (``acc=None``, or the same thing spelled as a
  24-bit format) adds nothing to it.
- A narrower accumulator adds its own error. A Binary16p8 accumulator, which
  has bfloat16's eight bits of precision, is noticeably worse. A Binary8p4 or
  Binary8p3 one is useless.
- Stochastic rounding does *not* help here: it is worse than
  round-to-nearest. The terms have random signs, so there is no systematic
  bias to remove, only variance to add. Stochastic rounding helps when the
  increments are systematically below half a spacing. That is the
  accumulator case shown in :doc:`/concepts` and the weight-update case in
  the :doc:`/tutorial`.

The two formats of a ``SplitMac`` may come from different families:
:math:`Q_{\text{mul}}` a :class:`~mptorch.BinaryK` and
:math:`Q_{\text{acc}}` a :class:`~mptorch.SuperFP`, or the reverse.
The equation is the one above, each rounding in its own format, and the
result holds the accumulate format's values. Two rules apply to such a
*cross-family* mac:

- It holds one format per slot. A *palette*, a list of formats from which
  each output element selects one (`Palettes: a format per output element`_,
  below), must be of one family.
- Its sum is always rounded. With ``acc=None`` the mac has one family again.

Every accumulate algorithm of the next section applies to it, with an
``outer`` format of the accumulate format's family, in either carrier. So
does the odd carrier step (`What the carrier rounds`_).

.. literalinclude:: ../../snippets/gemm_cross_family.py
   :language: python
   :caption: docs/snippets/gemm_cross_family.py

.. literalinclude:: ../../snippets/gemm_cross_family.out
   :language: text
   :caption: output

What the carrier rounds
-----------------------

The equations above name the formats' roundings. Each step also has a
rounding they do not name. The carrier first computes the product, the sum
or the FMA to its own precision, 24 bits in binary32 and 53 in binary64,
rounding to nearest. The format's cast then rounds that result.

Usually the second rounding absorbs the first. It does not when the exact
value lies within the carrier's last place of a format value. The carrier
then rounds the exact value onto that format value, and a directed mode keeps
it, instead of going to the neighbour that the exact value rounds to. For
example, ``1 + 2^-30`` rounded up to eight bits of precision is
``1 + 2^-7``. Its binary32 sum, however, is exactly 1, which every mode
keeps. binary64 holds the term ``2^-30``, but not ``2^-61``:

.. literalinclude:: ../../snippets/gemm_carrier_rounding.py
   :language: python
   :caption: docs/snippets/gemm_carrier_rounding.py

.. literalinclude:: ../../snippets/gemm_carrier_rounding.out
   :language: text
   :caption: output

How often this happens on random data depends on the formats' precision.
``dev/benchmarks/timing/carrier_double_rounding.py rates`` counts the outputs on
which the binary32 and binary64 carriers disagree. It multiplies N(0, 1)
operands, quantized to the format, at M = N = K = 256, under the directed
modes:

- FP8 formats disagree on at most 0.04% of outputs.
- Formats of up to eleven bits of precision disagree on at most 0.06% of
  outputs with a split mac, and 0.5% with a fused one.
- The widest formats disagree most: Binary16p12 fused on 1.4%, and
  Binary24p16 on 4.6% split and 22% fused. A fused product of two P-bit
  operands has 2P bits before the add, more than binary32 keeps.

**Rounding the carrier step to odd.** ``carrier_rounding=RoundMode.RO`` makes
the carrier round its product, sum and FMA to odd instead of to nearest.
:class:`~mptorch.quant.SplitMac`, :class:`~mptorch.quant.FusedMac`,
the flat functions and the ``*_gemm_formats`` layer factories take this
keyword. Rounding to odd rounds toward zero onto the carrier's grid, then
sets the last bit whenever the result is inexact. An inexact result is then
an odd carrier value, and it lies on the same side of every even carrier
value as the exact value does. So a format whose rounding boundaries are all
even carrier values rounds it exactly as it would round the exact value
(Boldo and Melquiond, "When double rounding is odd"). That holds when the format's
precision ``P`` is narrow enough:

========================= ======================== =========================
the format's rounding     binary32 (p = 24)        binary64 (p = 53)
========================= ======================== =========================
``RNE``, ``RNA``          ``P <= 22``              ``P <= 51``
``RU``, ``RD``, ``RZ``,   ``P <= 23``              ``P <= 52``
``RO``
``SR`` (``SR_A``)         ``P + prng_bits <= 23``  ``P + prng_bits <= 52``
``SR_B``, ``SR_C``        ``P + prng_bits <= 22``  ``P + prng_bits <= 51``
========================= ======================== =========================

``RNE`` and ``RNA`` need one bit more than the directed modes, because they
round at the midpoints between format values. ``SR_B`` and ``SR_C`` need one
bit more than ``SR``, because each also reads the bit of the value just below
its random bits. The bound holds in the carrier's subnormal range too, and
for saturating formats. A call whose format exceeds the bound raises. In
binary32, when binary64 would hold the format, the error names
``carrier=torch.float64``.

Three more details:

- A finite exact sum larger than the carrier's largest value reaches the cast
  as that largest value, not as infinity. ``SAT_PROPAGATE`` therefore
  saturates it, as P3109 says a finite value should.
- With ``acc=None`` only the product is rounded to odd. No format rounds the
  sum after it, so the sum stays the carrier's add, rounded to nearest.
- ``FusedMac(None)`` refuses ``RoundMode.RO``. A step rounded to odd is
  correct only when a format rounds it again, and no format rounds that
  mac's step.

The odd step is implemented for every ``NAIVE`` reduction, split and fused:

- the single-format GEMMs;
- the palettes (:class:`~mptorch.quant.Palette`), with each entry held
  to the bound;
- the cross-family macs, with both formats held to it;
- the three passes of a convolution over any of these
  (:func:`~mptorch.quant.conv_formats`; :doc:`convolutions`).

It runs on CPU and CUDA tensors in both carriers, and on Apple GPU tensors
in binary32.

``KAHAN``, ``BLOCK`` and ``TREE`` (below) take the odd step too, over a
single format or a cross-family mac, in a GEMM or a convolution. A palette
sums with ``NAIVE`` only. Under these algorithms every carrier operation that
a format rounds next is rounded to odd:

- ``KAHAN``'s term, sum and both differences;
- ``TREE``'s products and merges;
- the additions of ``BLOCK``'s and ``TREE``'s block sums to the total, when an
  outer format rounds them. The outer format is then held to the bound too.

Each format operation is then correctly rounded. Kahan's compensation assumes
that, and the carrier's nearest step can break it. These kernels are as many
as the default accumulated ones, a large part of the build, so they are
built only with ``MPTORCH_ODD_ACCUMULATED=1`` (CI sets it). Without it, such
a call raises, naming the flag.

The round-to-odd steps run in operators of their own
(``custom_matmul_*_odd``, ``custom_matmul_*_accumulated_odd`` and
``custom_conv_*_odd``), so the default operators do not change with them. A
build with ``MPTORCH_NO_ODD=1`` leaves all their kernels out, and the
operators then raise, naming the flag.

On the CPU the odd step assumes the default floating-point environment. A
flush-to-zero mode (``torch.set_flush_denormal(True)``) breaks it in
subnormal ranges, as it breaks the carrier arithmetic everywhere else.

The odd step costs time, mostly on the CPU. The measurements below are
``carrier_double_rounding.py bench`` on a Ryzen 9 7940HS and an RTX 4060
laptop GPU, and ``mps_backend.py bench-odd`` on an M3 Pro. The GPUs run
:math:`1024^3` products in binary32 and :math:`512^3` in binary64. The CPU
runs :math:`256^3` products, on one thread and on sixteen.

============================== ===================== =====================
time against the nearest step  fused                 split
============================== ===================== =====================
CUDA, binary32                 1.05-1.14x            1.04-1.15x
CUDA, binary64 (GeForce)       1.1x (SR) - 1.45x     1.1x (SR) - 1.5x
CPU, binary32, vector tiles    1.05-1.09x            1.07-1.11x
CPU, binary32, scalar loop     1.3-2.1x              1.4-2.4x
CPU, binary64                  1.4-2.7x              1.2-2.0x
Apple GPU, binary32            1.9-3.1x              0.75-2.4x
============================== ===================== =====================

The palettes, the cross-family macs and the convolutions cost what the GEMM
of their step costs: 1.03-1.16x on CUDA. On the CPU the cross-family macs
cost 1.5-2.8x through the scalar loop, a convolution over a format that the
vector tiles take 1.06-1.07x, and a BinaryK palette 1.0-1.3x its
nearest-step palette (1.6-1.7x with a map per element), since the odd tiles
serve palettes too ("Palettes: a format per output element", below).

Under ``KAHAN``, ``BLOCK`` and ``TREE`` the odd step costs the following.
The formats are Binary8p4 products and Binary16p11 sums, with blocks of 16
added to a Binary16p11 total. The CPU and CUDA rows are
``carrier_double_rounding.py bench-accumulated``, on the same Ryzen and
GeForce and on an Apple M3 Pro. The Apple GPU row is ``mps_backend.py
bench-odd``'s, for ``KAHAN`` only.

============================== ===================== =====================
time against the nearest step  binary32              binary64
============================== ===================== =====================
CUDA (a GeForce)               1.04-1.20x            1.3-1.5x
Ryzen, one thread              1.6-2.7x              1.7-2.4x
Ryzen, sixteen threads         1.4-2.5x              1.6-2.5x
M3 Pro, one thread             1.8-2.85x             1.5-2.2x
M3 Pro, eleven threads         1.8-2.8x              1.4-2.3x
Apple GPU (``KAHAN``)          1.2-2.8x              (no binary64)
============================== ===================== =====================

The CPU has no vector tiles for these algorithms, default or odd. ``KAHAN``
puts four rounded operations on each step's dependency chain, so that is
where the CPU pays most for the odd step's error-free transforms. On the
formats of that table, the binary32 and binary64 odd steps agree on every
output (``rates-accumulated``). The nearest carriers disagree on up to 0.1% of
outputs under ``KAHAN``, and on 3.3% for a fused ``BLOCK`` over Binary24p16.

Each backend rounds to odd in its own way:

- **CUDA.** The GPU has directed rounding. In both carriers it computes the
  operation twice, rounded down and rounded up, and selects the result whose
  last bit is odd. On a consumer GPU the binary64 kernels are limited by the
  FP64 rate. A step issues only a couple of FP64 instructions (the FMA and a
  compare in the cast), and the odd step adds a second FMA to them.
- **CPU, scalar loop.** The scalar loop has no directed rounding to call, so
  each step runs error-free transforms on the sum's dependency chain. In
  binary32 it computes the exact product and an exact sum in binary64,
  narrows the sum to binary32, and sets the last bit from the sign of what
  the narrowing dropped. In binary64 it splits the product exactly, runs two
  exact sums, and fixes the last bit the same way.
- **CPU, vector tiles.** The vector tiles take the cases that the default op
  runs in vector code: binaryK in binary32 under ``RNE``, with one format per
  slot (:doc:`calling`, "Performance notes"). They run a whole tile's step at
  once, as the GPU does. AVX-512 uses its embedded directed rounding. AVX2
  and NEON use the processor's rounding mode, set once for each pass over
  the tile. The tiles cost 1.04-1.4x the default tile, depending on the
  instruction set. The scalar loop costs 16-26x the default tile in those
  cases.
- **Apple GPU.** It has neither directed rounding nor binary64, and its float
  unit flushes binary32 subnormals (:ref:`apple-gpu`). binary32 therefore
  rounds to odd with error-free transforms in binary32: the product's error
  from an FMA, two exact sums and a fast exact sum, then the last bit set on
  the word. A test on the operands' words catches every input that could
  make an intermediate subnormal, or overflow. Such an input takes a slower
  path, which scales it exactly by a power of two on the words, runs the same
  transforms, and scales the result back. The sums that no format rounds next
  stay the carrier's operations rounded to nearest, computed exactly on the
  words where a subnormal is involved. An odd op's result is therefore the
  CPU's, word for word. The step costs most under ``RNE``, whose casts cost
  least: 2.4x for a split mac and 3.1x for a fused one, against 1.2x for a
  split mac under ``SR``.

Summing differently: KAHAN, BLOCK and TREE
------------------------------------------

A narrow accumulator does badly in the table above because of the *order* of
the sum as much as its width. There is one running sum, rounded :math:`K`
times, each time against a total that has grown far larger than the term
being added. The mac's *accumulate algorithm* (its ``accumulate_algorithm``
keyword) changes the order and keeps the formats.

Take one output element and drop its indices :math:`i, j`, so that its terms
are :math:`a_k b_k` for :math:`k = 1, \dots, K`. The notation is:

- :math:`p_k = Q_{\text{mul}}(a_k b_k)`, the rounded product;
- :math:`s`, the running sum;
- :math:`Q_{\text{acc}}`, the accumulate rounding;
- :math:`Q_{\text{out}}`, a third rounding, to the mac's ``outer`` format.

Each rounding is the identity when its format is ``None``. The algorithms
are:

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
   Two levels. The ``NAIVE`` recurrence runs on a block's sum :math:`s`.
   After every ``block_size`` products, and after the last one, the block's
   sum is added to a total :math:`T`,
   :math:`T \leftarrow Q_{\text{out}}(T + s)`, and :math:`s` restarts from
   zero. A block's sum stays small, so a narrow ``acc`` loses little in it.
   The ``outer`` format (a wider one, or ``None`` for the carrier) rounds
   once per block instead of once per product. ``block_size`` must divide 16
   or be a multiple of 16, and has no default.
:attr:`~mptorch.AccumulateAlgorithm.TREE`
   The products of each block of ``block_size`` :math:`= 2^L` are summed
   pairwise, :math:`Q_{\text{acc}}(Q_{\text{acc}}(p_1 + p_2) + Q_{\text{acc}}(p_3 + p_4))`
   and so on up, so that every addition is of two terms of similar size.
   ``block_size`` is a power of two from 2 to 256, and 16 by default. The
   block's root is added to the total as in ``BLOCK``. When the last block
   is incomplete, the complete subtrees it has are merged, smallest first.
   ``TREE`` is for ``SplitMac`` only: a fused multiply-add has no product
   term to pair.

In ``KAHAN`` and ``BLOCK``, a fused mac's step,
:math:`Q_{\text{fma}}(s + a_k b_k)`, replaces the split one. A last partial
block is added like a whole one, and nothing is padded with zeros. The CPU
and CUDA kernels return the same bits in every deterministic mode. The
result holds the values of the last rounding applied, which is ``outer``'s
when there is one. :class:`~mptorch.AccumulateAlgorithm` states each
algorithm step by step, including the random draws of the stochastic modes.

.. literalinclude:: ../../snippets/gemm_kahan_block_tree.py
   :language: python
   :caption: docs/snippets/gemm_kahan_block_tree.py

.. literalinclude:: ../../snippets/gemm_kahan_block_tree.out
   :language: text
   :caption: output

The run prints four tables:

- **First table: float32 alone.** The three algorithms behave as in any
  numerical library. ``KAHAN`` recovers about a digit and a half over a
  4096-term sum, and blocking about half of that.
- **Second table.** An accumulator with eight bits of precision costs
  ``NAIVE`` three times the error that the Binary8p4 products already carry.
  Every other row is back at the products' own error, which no summation can
  improve on.
- **Third table.** This is the narrowest case, sums in Binary8p4 itself.
  ``NAIVE`` is useless, and the other three come within about a factor of two
  of the products' error. An ``outer`` format *narrower* than the carrier
  costs a little, as expected: it is one more rounding, and in exchange the
  total is a value of a format you chose.
- **Fourth table.** The second, in binary64 (``carrier=torch.float64``). A
  product of two float32 operands is exact there, so the multiply format
  rounds it once instead of twice. The errors are the second table's to three
  digits: the formats round every intermediate far more than binary32 does.

The price is kernel time, in proportion to the roundings per step. On an RTX
4060 laptop GPU a 1024\ :sup:`3` split-mac GEMM takes 1.2 times ``NAIVE``'s
time under ``BLOCK``, 1.4 times under ``TREE`` and 2.25 times under
``KAHAN`` (``BinaryK(8, 4)`` products into ``BinaryK(16, 11)`` sums under
``RNE``, November 2026).

The three algorithms are implemented for the single-format and cross-family
macs, the convolutions over them, and the block product (:doc:`block`). They
run on the CPU, CUDA and Apple GPUs, in both carriers (next section) where
the device has float64. A palette raises: it sums with ``NAIVE`` only.

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
  simulated exactly on float64 ones.
- **Wider random draws.** Under the stochastic modes each rounding draws two
  words of its element's random stream, and ``P - 1 + prng_bits`` may reach
  52.

``carrier=torch.float64`` on a mac or a flat function computes narrower
operands the binary64 way:

1. Both operands are widened to float64.
2. The binary64 kernel runs.
3. The result is narrowed back to the operands' dtype with one rounding.

This is bit for bit the float64 call on the widened operands, random draws
included, then stored in the narrower dtype.

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
already keeps what the 30-bit products carry. A cross-family mac (a superfp
multiply into a 40-bit binaryK sum) runs in binary64 like the single-family
ones.

The float32 table shows both sides of the choice. Its first three rows are
what binary64 gains: a sum or a fused step that no format rounds, and a
multiply format wider than binary32. Where the formats round every
intermediate at least as much as binary32 would, the two carriers are within
one rounding of each other or equal to the bit. That is the case with 24-bit
formats on both sides, and with Binary8p4 on both sides (the last row).
Beyond what the formats keep, the carrier adds little. On a GPU binary64
also costs time (see :doc:`calling`, "Performance notes"), in addition to the
float64 copies of the operands and the result.

Exact accumulation
------------------

Some FP8 dot-product units sum their products exactly, in a fixed-point
register wide enough for every product (a Kulisch accumulator), and round
only the result. The binary64 carrier gives the same sums for the narrower
formats. With ``FusedMac(None, carrier=torch.float64)`` no format rounds a
product or a sum, and binary64 rounds none of them while each partial sum
fits in its 53 bits:

- **The bound.** Take a format whose smallest positive value is
  :math:`2^{-s}` and whose largest is :math:`M`. Its products are multiples
  of :math:`2^{-2s}` no larger than :math:`M^2`. A sum of :math:`K` of them
  is exact in binary64, in any order, while :math:`K M^2 2^{2s} \le 2^{53}`.
- **E4M3** (:math:`s = 9`, :math:`M = 448`): the products span 36 bits, so
  the sums are exact on any data for :math:`K \le 2^{17}`. PyTorch's own
  float64 matmul gives the same sums there.
- **E5M2** (:math:`s = 16`, :math:`M = 57344`): the products span 64 bits,
  more than binary64 holds. The sums are exact only on data whose products
  span at most :math:`53 - \log_2 K` bits. In the run, data spread over 20
  binades makes every sum inexact. Formats with a wider range, such as
  float16 and bfloat16, are further out still.
- **Rounding the exact sum once.** In a float32 GEMM, ``carrier=torch.float64``
  narrows the result to binary32 before an output rounding rounds it again.
  The second rounding changes a few results in :math:`10^7` (the run's last
  line). Float64 operands keep the exact sum until the output rounding.
- MPS has no binary64, so these exact sums need the CPU or CUDA.
- No mac sums in a finite fixed-point register, one with a fixed binary
  point that saturates or drops the bits below it: the fixed-point GEMM
  (below) takes fixed-point formats only. Its exact sums of 8-bit codes run
  on an integer GEMM, with the same words ("Exact sums on an integer GEMM",
  below).

.. literalinclude:: ../../snippets/exact_accumulation.py
   :language: python
   :caption: docs/snippets/exact_accumulation.py

.. literalinclude:: ../../snippets/exact_accumulation.out
   :language: text
   :caption: output

Fixed-point macs
----------------

A ``SplitMac`` or ``FusedMac`` whose formats are :class:`~mptorch.FixedPoint`
(:doc:`/concepts`, "FixedPoint: one step across the range") computes a dot
product in fixed point. Its equations are the float macs', with one
difference: the carrier rounds no sum. With :math:`\mathrm{RN}_c` the
carrier's rounding to nearest,

.. math::

   \text{split:}\quad
   s_k = Q_{\text{acc}}\Big(s_{k-1} + Q_{\text{mul}}\big(\mathrm{RN}_c(a_{ik}\, b_{kj})\big)\Big),
   \qquad
   \text{fused:}\quad
   s_k = Q_{\text{fma}}\big(\mathrm{RN}_c(a_{ik}\, b_{kj}) + s_{k-1}\big),

where every sum inside a :math:`Q` is exact. The product is the carrier's,
as in every mac: for an exact one, quantize the operands first, as the run
below does, so that their product fits the carrier's significand, or use
``carrier=torch.float64``. The sum is what a fixed-point adder computes: the
exact sum of two values, rounded once to the format's grid.

**How the kernels compute it.** The definition is an integer accumulator: a
running sum in 64-bit integers, each step rounded to the accumulate format's
grid once. Where the carrier holds every step exactly, the kernels sum in the
carrier instead, which is faster and gives the same bits. A call chooses the
carrier when

- the mac is split and its sum saturates;
- the carrier's cast rounds both formats with no finding, not even a
  warning (:doc:`/concepts`, "What the carrier can hold");
- an exact sum of a product and a running sum fits the carrier's
  significand: :math:`\max(n - \text{fl}) + 1 + \max(\text{fl})`, over the
  two formats' magnitude bits :math:`n` and fractional lengths, is at most
  24 bits in binary32 and 53 in binary64.

Every step saturates onto its format, so the choice depends on the formats
alone, never on :math:`K`. Nothing selects it but the formats.

**Overflow.** By default a sum past the accumulate format's range saturates
onto its end, as the elementwise quantizer does.
``overflow=Overflow.WRAP`` (on the mac, the flat functions and the layer
factories) wraps it around instead, as an integer adder does: modulo
:math:`2^{\text{wl}}` times the step, in a signed format and an unsigned one
alike. A symmetric format has no modulus to wrap around, so it refuses
``WRAP``. Modular arithmetic forgives an overflow along the way: when the
products are on the accumulate format's grid, so that no step rounds, a
wrapped sum is the exact sum whenever that fits the format, whatever the
order of the terms. A saturated one is not (the run's second block).
Wrapping runs on the integer accumulator.

**Wide accumulators.** The accumulate format may have up to 53 magnitude
bits, more than the operands' dtype holds: 24 for float32, 11 for float16
and 8 for bfloat16. Such a result is computed into float64:

- :func:`~mptorch.quant.fixedpoint_matmul` returns it in float64.
- A layer's forward pass adds the bias to it in float64, rounds it once with
  the layer's output rounding, and stores it in the layer's dtype
  (``output_quant`` on a ``GemmFormats`` or on
  :func:`~mptorch.quant.fixedpoint_gemm_formats`). Without an output
  rounding the forward pass raises, rather than round the result to the
  dtype a second time.
- A gradient pass raises too, since a gradient is stored in its dtype with
  no rounding of its own: give a pass that needs a wide accumulator float64
  operands.

On float64 operands nothing is wide. An accumulate format may also draw more
random bits under stochastic rounding than the elementwise quantizer allows
beside it, up to the carrier's draw (23 bits in binary32, 52 in binary64):
the integer accumulator draws them separately from the format's significand.

**What it does not take.** The fixed-point GEMM runs the ``NAIVE`` order
only, one format per slot (no palette), both formats in fixed point, and
``carrier_rounding=RoundMode.RNE``. Its multiply format must be rounded by
the carrier's cast without a finding, since both paths round the product
with that cast. The integer accumulator's exact sums are at most 62 bits.
A mac or a call that asks for more raises, with the rule that refused it.

It runs on the CPU and on CUDA, in both carriers, and on MPS in binary32,
where it returns the CPU's words except that a product below
:math:`2^{-126}` flushes to zero before it is rounded (:doc:`/concepts`, "On
an Apple GPU"). MPS has no float64, so it refuses a wide result in every
pass. It serves every entry point that takes a mac:
:func:`~mptorch.quant.qmatmul` and the layers,
:func:`~mptorch.quant.conv_formats` (:doc:`convolutions`), recipes
(:doc:`/instrument`), the flat functions
:func:`~mptorch.quant.fixedpoint_matmul` and
:func:`~mptorch.quant.fixedpoint_matmul_fma`, and the layer factories
:func:`~mptorch.quant.fixedpoint_gemm_formats` and
:func:`~mptorch.quant.fixedpoint_gemm_formats_fma`. The run repeats a split
mac by hand, then sums int8 values both ways, then computes a 32-bit
accumulation on float32 operands:

.. literalinclude:: ../../snippets/gemm_fixedpoint.py
   :language: python
   :caption: docs/snippets/gemm_fixedpoint.py

.. literalinclude:: ../../snippets/gemm_fixedpoint.out
   :language: text
   :caption: output

The second block's exact sum, 40, fits int8, and the wrapped sums reach it.
The saturated sum stops at 127 after the second term, and the last two terms
take it to -33.

**Cost.** Per mac step, with ``FixedPoint(16, 8)`` products into
``FixedPoint(16, 4)`` sums, against a BinaryK split mac (``Binary16p9`` into
``Binary16p12``, formats whose casts take the fast path), on an RTX 4060
laptop GPU at :math:`2048^3` and a Ryzen 7940HS at :math:`512^3`
(``dev/benchmarks/timing/fixedpoint_gemm.py``, 2026-11-25):

==============================  =========  =========  ===========  ===========
mac                             CUDA, RNE  CUDA, SR   1 CPU, RNE   1 CPU, SR
==============================  =========  =========  ===========  ===========
BinaryK split                   7.5 ps     20.3 ps    0.41 ns      8.2 ns
fixed point, the carrier path   4.1 ps     23.1 ps    0.56 ns      10.1 ns
integer accumulator, split      13.5 ps    26.4 ps    5.4 ns       9.7 ns
integer accumulator, fused      20.1 ps    23.6 ps    9.8 ns       6.3 ns
==============================  =========  =========  ===========  ===========

- The carrier path is the cheapest mac under RNE on CUDA. On the CPU it runs
  a scalar loop the compiler vectorizes, at 1.4 times the BinaryK mac's
  vector tiles.
- A wide result costs about what a narrow one does: 8.6 ps a step on CUDA
  under RNE into ``FixedPoint(40, 8)``.
- The other modes cost more than RNE. On CUDA the carrier path takes 8.3 ps
  a step under ``RZ``, 10.9-11.6 ps under ``RU``, ``RD`` and ``RNA`` and
  12.0 ps under ``RO``, and 21-24 ps in binary64 in every mode but SR. On one
  CPU thread (at :math:`384^3`) it takes 2.8 ns under ``RZ``, 5.2-5.5 ns
  under ``RU`` and ``RD`` and 4.3-4.9 ns under ``RNA`` and ``RO``. On one
  arm64 thread (an M3 Pro), where every deterministic mode but RNE takes the
  GEMM's own rounding, it takes 1.2 ns under RNE and 2.3-3.6 ns under the
  others, against 11-12 ns for ``RNA`` and ``RO`` through the elementwise
  casts.
- On an M3 Pro's GPU the integer accumulator takes 7 to 10 times the
  carrier path under RNE, since Apple GPUs emulate 64-bit integer
  arithmetic. The carrier path takes 5.0 ps a step under RNE, 19-23 ps under
  the other deterministic modes and 61 ps under SR, against 10.4, 17-110 and
  62 ps for the BinaryK split mac.

**Exact sums on an integer GEMM.** When the operands are values of
fixed-point formats of at most 8 bits, a multiply format that holds every
product and an accumulate format whose grid is as fine as the products' keep
every step exact for sums up to some length. No step rounds or saturates,
and the result is the exact integer sum of the operands' codes, scaled. Any
exact engine computes that sum, in any order, so a pass that qualifies runs
on a vendor GEMM and returns the kernel's words, word for word:

- **The engine.** ``torch._int_mm`` (an int8 x int8 GEMM into int32) on
  CUDA, on MPS, and on a CPU whose PyTorch has oneDNN. On a CPU without
  oneDNN, where ``torch._int_mm`` is a scalar loop, PyTorch's float matmul:
  each partial sum is an integer number of the products' steps, so float32
  sums exactly while there are at most :math:`2^{24}` of them (1,024 terms of
  ``FixedPoint(8, 7)`` products) and float64 beyond. It takes float64 too
  when ``torch.set_float32_matmul_precision`` has lowered float32's
  precision, and inside a compiled graph, which cannot read that setting.

- **Where.** In a layer built from a :class:`~mptorch.quant.GemmFormats`
  (:meth:`~mptorch.quant.GemmFormats.affine`,
  :meth:`~mptorch.quant.GemmFormats.matmul`, and so recipes, ``convert`` and
  ``simulate``), for each pass whose two operand quantizers are a ``Quant``
  of such a format. The flat functions and :func:`~mptorch.quant.qmatmul`
  do not know their operands' formats and always run the kernel.
- **When.** The mac saturates and rounds deterministically, and the planner
  above sends the call to the integer accumulator (an exact sum of 8-bit
  codes longer than a few hundred terms always goes there). Each call checks
  that its sum is short enough for neither the accumulate format nor int32
  to overflow (1,023 terms for ``FixedPoint(8, 7)`` operands into
  ``FixedPoint(25, 14)``, 131,071 into ``FixedPoint(32, 14)``), that its
  operands are float32 or float64, and that it is large enough to gain: at
  least :math:`2^{20}` mac steps on the CPU and :math:`2^{25}` on CUDA and
  MPS, where each call spends about 200 us launching the route's small
  operations.
- **NaN.** A NaN operand reaches the same rows and columns as in the kernel,
  as the carrier's canonical quiet NaN, which is what the integer
  accumulator returns.
- **Memory.** On ``torch._int_mm`` it adds the operands' codes (a byte an
  element) and an int32 product computed in blocks of at most 64 MB; on the
  float matmul nothing in float32, and float64 copies of the operands past
  float32's exact range. Nothing is saved for backward.

At :math:`512^3`, with ``FixedPoint(8, 7)`` operands into
``FixedPoint(25, 14)``, a step takes 6.7 ps on one Ryzen thread and 2.9 ps on
sixteen, against 1,000 and 136 ps for the kernel; on the RTX 4060 1.8 ps
against 9.8 at :math:`512^3`, and 0.18-0.24 against 8.6 from :math:`2048^3`
into ``FixedPoint(32, 14)`` (``dev/benchmarks/timing/fixedpoint_gemm.py
exact``, 2026-11-27). On an M3 Pro, ``torch._int_mm`` runs at about 2 ps a
step on the GPU, against 24 for the kernel, and the float matmul at 1.3
(float32) and 5.0 (float64) on the CPU, against 746.

Palettes: a format per output element
-------------------------------------

A *palette* is a list of up to eight formats of one family. A ``Palette`` (or
a plain list of formats) in any format slot of a mac selects the palette
kernel, in which each output element's dot product runs in the palette entry
that an integer map, ``prec_idx``, assigns to it. The map can have:

- the output's shape ``[M, N]``;
- shape ``[M, 1]``, for one format per row;
- shape ``[1, N]``, for one format per column.

It can also have a leading batch dimension, of the batch size or of size 1
to share the map. Its values must lie in ``[0, len(palette))``.

The kernel tabulates only each entry's widths and bias. The entries must
therefore agree on every other field: the sign, the number of stochastic
bits, the saturation mode and, for :class:`~mptorch.BinaryK`, the subnormals
mode.

.. literalinclude:: ../../snippets/gemm_palette.py
   :language: python
   :caption: docs/snippets/gemm_palette.py

.. literalinclude:: ../../snippets/gemm_palette.out
   :language: text
   :caption: output

**What a palette costs.** On CUDA a palette GEMM costs about what its
single-format twin costs (1.1x), and 1.2-1.8x on an Apple GPU. On the CPU a
BinaryK palette under RNE runs the vector tiles its twin runs, a tile's
elements grouped by their entry: a tile whose elements share one entry runs
the twin's tile, a map per row runs each entry's rows, and maps per column
and per element run each element on its own entry's constants. Against the
twin on one Ryzen thread, at :math:`256^3` and at
:math:`32 \times 128 \times 128`:

============================  ==========  ==========  ==========  ===========
map                           one entry   per row     per column  per element
============================  ==========  ==========  ==========  ===========
nearest step                  1.0-1.1x    1.1-1.2x    1.4-1.5x    1.5-1.6x
``carrier_rounding=RO``       1.0-1.1x    1.1-1.3x    1.3-1.6x    2.2-2.4x
============================  ==========  ==========  ==========  ===========

(on sixteen threads, 1.0-1.8x). Before these tiles a palette ran the scalar
loop, at 9-12x its twin's time (25x for the odd macs). The other palettes
(SuperFP, the other round modes) run the scalar loop as their twins do
(``dev/benchmarks/timing/cpu_palette_gemm.py``).

A recipe gives a model's layers palettes, with one map per layer (a format
per output neuron or channel): :doc:`/instrument`, "Palettes: a format per
neuron".

A format for every element
~~~~~~~~~~~~~~~~~~~~~~~~~~

A dense ``[M, N]`` map is the general case. Every output element's dot
product runs in the palette entry named by its own map entry, independently
of its row, its column and its neighbours. How the map is chosen is up to the
caller: a sensitivity analysis, a hardware assignment, a search.

The run below builds one map per element. For each of the 20 elements of a
product, it takes the narrowest of eight BinaryK formats, from 3 to 24 bits
of precision, whose result stays within 0.1% of the exact dot product there
(or the widest, where none does). It then runs the whole product as a single
palette GEMM, and checks every element against the single-format GEMM in
that element's format. They are equal to the bit, because an element's dot
product reads only its own row and column of the operands and its own
palette entry.

The same map picks a *pair* when the multiply and the accumulate palettes
differ. Entry ``i`` of a ``SplitMac`` palette pairs ``mul[i]`` with
``acc[i]``, so one element can multiply in 4 bits into a 24-bit sum while its
neighbour does the opposite. Two palettes must have the same length. A single
format opposite a palette serves every entry. A ``FusedMac`` takes a palette
the same way.

A batched product takes a ``[B, M, N]`` map, with a format per element of
every sample. The fields a palette shares (sign, stochastic bits, saturation
and subnormals) are the same for every element. A palette whose entries
disagree on one is refused, with an error naming the entry.

.. literalinclude:: ../../snippets/gemm_palette_elements.py
   :language: python
   :caption: docs/snippets/gemm_palette_elements.py

.. literalinclude:: ../../snippets/gemm_palette_elements.out
   :language: text
   :caption: output

