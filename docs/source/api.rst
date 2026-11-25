.. _api-reference:

API reference
=============

This page lists MPTorch's public classes and functions, grouped as the
guides are: formats, quantizers, kernels and layers. The formats and the
modes are importable from ``mptorch``, and the quantizers, kernels and
layers from ``mptorch.quant``. There are two exceptions: the format checks
are importable from ``mptorch.number``, and
:func:`~mptorch.quant.matmul.as_matmul_formats` from
``mptorch.quant.matmul``.

Number formats and modes
------------------------

See :doc:`concepts`. Each format is a frozen value: it cannot be modified,
and it is compared and hashed by value. The format classes are:

- :class:`~mptorch.BinaryK` and :class:`~mptorch.SuperFP`, the two
  floating-point families;
- :class:`~mptorch.FixedPoint`, integers times a fixed step;
- :class:`~mptorch.BlockFormat`, narrow elements that share a scale per
  block.

The presets are importable from ``mptorch``: the element and scale formats
``E8M0``, ``E4M3``, ``E5M2``, ``E2M1``, ``E2M3`` and ``E3M2`` (each a
:class:`~mptorch.BinaryK`), and the block formats ``MXFP8_E4M3``,
``MXFP8_E5M2``, ``MXFP6_E2M3``, ``MXFP6_E3M2``, ``MXFP4_E2M1`` and ``NVFP4``
(each a :class:`~mptorch.BlockFormat`). :func:`~mptorch.bfp` builds a block
floating-point format.

The enums :class:`~mptorch.RoundMode`, :class:`~mptorch.SaturationMode`,
:class:`~mptorch.SubnormalsMode` and :class:`~mptorch.AccumulateAlgorithm`
have the same names and integer values as the C++ enums in
``mptorch/csrc/common/modes.h``, because the kernels receive their
``.value``. :class:`~mptorch.Domain` exists only in Python: a format records
its domain in its saturation mode, which is all the kernels see.

.. module:: mptorch

Formats
~~~~~~~

.. autoclass:: BinaryK
   :undoc-members:

.. autoclass:: SuperFP
   :undoc-members:

.. autoclass:: FixedPoint
   :undoc-members:

.. autoclass:: BlockFormat
   :undoc-members:

.. autoclass:: ScaleRounding
   :undoc-members:

.. autofunction:: bfp

Modes
~~~~~

.. autoclass:: RoundMode
   :undoc-members:

.. autoclass:: SaturationMode
   :undoc-members:

.. autoclass:: Domain
   :undoc-members:

.. autoclass:: SubnormalsMode
   :undoc-members:

.. autoclass:: AccumulateAlgorithm
   :undoc-members:

Base classes and warnings
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Number

.. autoclass:: FloatFormat

.. autoexception:: FormatRangeWarning

Format checks
~~~~~~~~~~~~~

These functions test a format against two floating-point types: the
*carrier*, the binary float (binary32 or binary64) that a cast rounds in,
and the *storage* dtype that the result is written to, when that dtype is
narrower than the carrier. They raise :exc:`ValueError` when the format
cannot work in that type at all, and warn with
:class:`~mptorch.FormatRangeWarning` when it works over only part of its
range. A format runs them when it is built, to reject what no carrier can
hold. The quantizers and matrix products report their findings on each
call, for the carrier and the dtype of that call. The checks are importable
from ``mptorch.number``.

.. module:: mptorch.number

.. autofunction:: check_binaryK

.. autofunction:: check_binaryK_carrier

.. autofunction:: check_binaryK_storage

.. autofunction:: check_superfp

.. autofunction:: check_superfp_carrier

.. autofunction:: check_superfp_storage

.. autofunction:: check_fixedpoint

.. autofunction:: check_fixedpoint_carrier

.. autofunction:: check_fixedpoint_storage

.. autofunction:: check_block_carrier

.. autofunction:: check_block_storage

Quantizers
----------

See :doc:`quantizers`.

.. module:: mptorch.quant

.. autofunction:: binaryK_quantize

.. autofunction:: superfp_quantize

.. autofunction:: fixedpoint_quantize

.. autofunction:: binaryK_quantize_

.. autofunction:: superfp_quantize_

.. autofunction:: fixedpoint_quantize_

.. autoclass:: Quant
   :special-members: __call__

.. autoclass:: ScaledQuant
   :special-members: __call__

.. autoclass:: Quantizer

Block quantizers
~~~~~~~~~~~~~~~~

.. autofunction:: block_quantize

.. autofunction:: block_quantize_

.. autofunction:: block_pack

.. autofunction:: block_unpack

.. autoclass:: BlockPacked

.. autoclass:: BlockQuant
   :special-members: __call__

Kernels
-------

See :doc:`kernels/index`.

The arithmetic of a dot product
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A *mac* describes the arithmetic of one multiply-accumulate step of a dot
product. :class:`~mptorch.quant.SplitMac` rounds the product and the sum
separately, and :class:`~mptorch.quant.FusedMac` rounds one fused
multiply-add. A :class:`~mptorch.quant.Palette` in a mac's format slot holds
several formats, from which each output element selects one.
:class:`~mptorch.quant.BlockMac` describes the product of two tensors in
block formats. Each is a frozen value, so a mac is resolved to its kernel
call once, and the result is cached on the mac's value.

.. autoclass:: SplitMac

.. autoclass:: FusedMac

.. autoclass:: Palette

.. autoclass:: BlockMac

Matrix products
~~~~~~~~~~~~~~~

.. automodule:: mptorch.quant.matmul
   :no-members:

.. currentmodule:: mptorch.quant

.. autofunction:: qmatmul

.. autofunction:: qmm

.. autofunction:: qbmm

.. autofunction:: mptorch.quant.matmul.as_matmul_formats

.. autoclass:: QMatmulFormats

.. autoclass:: QMatmul

.. autofunction:: matmul_formats

.. autofunction:: block_matmul

.. autofunction:: block_matmul_formats

The schema tier
~~~~~~~~~~~~~~~

One function per kernel operation. Each takes every argument of the
operation explicitly, and none of them is differentiable. See
:doc:`kernels/calling` for when to use these instead of ``qmatmul``.

.. autofunction:: binaryK_matmul

.. autofunction:: superfp_matmul

.. autofunction:: binaryK_matmul_fma

.. autofunction:: superfp_matmul_fma

.. autofunction:: binaryK_matmul_mixed

.. autofunction:: superfp_matmul_mixed

.. autofunction:: binaryK_matmul_fma_mixed

.. autofunction:: superfp_matmul_fma_mixed

.. autofunction:: binaryK_superfp_matmul

.. autofunction:: superfp_binaryK_matmul

Layers
------

See :doc:`layers`.

.. autoclass:: QAffineFormats

.. autoclass:: QLinear

.. autoclass:: QPackedLinear
   :exclude-members: forward, reset_parameters, extra_repr, get_extra_state, set_extra_state

.. autofunction:: freeze_packed_

.. autoclass:: QConv1d

.. autoclass:: QConv2d

.. autoclass:: QConv3d

Layer arithmetic factories
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: binaryK_gemm_formats

.. autofunction:: binaryK_gemm_formats_fma

.. autofunction:: superfp_gemm_formats

.. autofunction:: superfp_gemm_formats_fma

.. autofunction:: binaryK_superfp_gemm_formats

.. autofunction:: superfp_binaryK_gemm_formats

.. autofunction:: conv_formats

.. autofunction:: block_gemm_formats

Pointwise ops and activations
-----------------------------

See :doc:`kernels/pointwise`, and :doc:`layers` for the modules.

.. autoclass:: PointwiseFormats
   :members: uniform, label

.. autoclass:: ActivationFormats
   :members: uniform, label

.. autofunction:: qadd

.. autofunction:: qsub

.. autofunction:: qmul

.. autofunction:: qdiv

.. autofunction:: qsqrt

.. autofunction:: qrsqrt

.. autofunction:: qexp

.. autofunction:: qpow

.. autofunction:: qgelu

.. autofunction:: qsilu

.. autofunction:: qsigmoid

.. autofunction:: qtanh

.. autofunction:: qhardswish

.. autofunction:: qhardsigmoid

.. autofunction:: qrelu

.. autofunction:: qembedding

.. autoclass:: QGELU

.. autoclass:: QSiLU

.. autoclass:: QSigmoid

.. autoclass:: QTanh

.. autoclass:: QHardswish

.. autoclass:: QHardsigmoid

.. autoclass:: QReLU

.. autoclass:: QEmbedding

Piecewise-linear tables, and the approximate inverse square roots
(:doc:`kernels/pointwise`, "Lookup-table activations";
:doc:`kernels/reductions`, "Norm variants"):

.. autoclass:: PWLTable
   :members: fit, rsqrt, max_error, label, segments

.. autofunction:: silu_table

.. autofunction:: gelu_table

.. autofunction:: qpwl

.. autoclass:: QPWLActivation

.. autoclass:: FISR

Sums, softmax and norms
-----------------------

See :doc:`kernels/reductions`, and :doc:`layers` for the modules.

.. autoclass:: ReductionFormats
   :members: uniform, label

.. autoclass:: SoftmaxFormats
   :members: uniform, label

.. autoclass:: NormFormats
   :members: uniform, label

.. autofunction:: qsum

.. autofunction:: qmean

.. autofunction:: qsoftmax

.. autofunction:: qlayernorm

.. autofunction:: qrmsnorm

.. autofunction:: qgroupnorm

.. autofunction:: qbatchnorm

.. autoclass:: QSoftmax

.. autoclass:: QLayerNorm

.. autoclass:: QRMSNorm

.. autoclass:: QGroupNorm

.. autoclass:: QBatchNorm1d

.. autoclass:: QBatchNorm2d

.. autoclass:: QBatchNorm3d

Attention
---------

See :doc:`kernels/attention`, :doc:`layers` for the module and :doc:`instrument` for the
recipes.

.. autoclass:: AttentionFormats
   :members: uniform, label

.. autoclass:: FlashFormats
   :members: label

.. autofunction:: qattention

.. autoclass:: QMultiheadAttention

.. autoclass:: QAttentionFormats

.. autoclass:: MultiheadAttentionFormats
   :members: label, container

Converting or simulating a model
--------------------------------

See :doc:`instrument`.

.. autoclass:: GemmFormats

.. autoclass:: Rule

.. autoclass:: Recipe

.. autofunction:: convert_

.. autofunction:: convert

.. autofunction:: unconvert_

.. autoclass:: simulate
   :members: report, wrap

.. autoclass:: SimulateReport

.. autoclass:: SimulateRow

.. autoclass:: SimulatedModule
   :members: close

.. autofunction:: register

.. autoclass:: ConvertReport

.. autoclass:: ConvertRow

.. autofunction:: profile

.. autoclass:: Profile

.. autoclass:: ProfileRecord

.. autoclass:: ElementRecord

.. autoclass:: UncountedRecord

.. autofunction:: set_palette_maps_

.. autofunction:: palette_maps

A :class:`~mptorch.quant.GemmFormats`' ``rotation`` rotates its product's
operands along K before they are quantized (:doc:`kernels/block`,
"Rotating before block quantization"):

.. autoclass:: Hadamard
   :members: block, label

.. autofunction:: hadamard_rotate

The ready-made recipes are in ``mptorch.quant.presets``:

.. autofunction:: mptorch.quant.presets.uniform

.. autofunction:: mptorch.quant.presets.fp8

.. autofunction:: mptorch.quant.presets.mx
