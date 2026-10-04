.. _api-reference:

API reference
=============

The public surface, grouped as the guides are: formats, quantizers, kernels
and layers. Everything documented here is importable from the two packages
``mptorch`` (formats and enums) and ``mptorch.quant`` (quantizers, kernels,
layers).

Number formats and modes
------------------------

See :doc:`concepts`. The presets are importable from ``mptorch``: the element
and scale formats ``E8M0``, ``E4M3``, ``E5M2``, ``E2M1``, ``E2M3`` and
``E3M2`` (each a :class:`~mptorch.BinaryK`), and the block formats
``MXFP8_E4M3``, ``MXFP8_E5M2``, ``MXFP6_E2M3``, ``MXFP6_E3M2``,
``MXFP4_E2M1`` and ``NVFP4`` (each a :class:`~mptorch.BlockFormat`).

.. automodule:: mptorch.number
   :members: RoundMode, SaturationMode, SubnormalsMode, AccumulateAlgorithm, Number, FloatFormat, BinaryK, SuperFP, BlockFormat, ScaleRounding, FormatRangeWarning
   :undoc-members:
   :member-order: bysource

Quantizers
----------

See :doc:`quantizers`.

.. autofunction:: mptorch.quant.binaryK_quantize

.. autofunction:: mptorch.quant.superfp_quantize

.. autofunction:: mptorch.quant.binaryK_quantize_

.. autofunction:: mptorch.quant.superfp_quantize_

.. autoclass:: mptorch.quant.Quant
   :members: __call__

.. autoclass:: mptorch.quant.Quantizer
   :members: forward

Block quantizers
~~~~~~~~~~~~~~~~

.. autofunction:: mptorch.quant.block_quantize

.. autofunction:: mptorch.quant.block_quantize_

.. autofunction:: mptorch.quant.block_pack

.. autofunction:: mptorch.quant.block_unpack

.. autoclass:: mptorch.quant.BlockPacked
   :members: mT, unpack, to, nbytes, cols, device

.. autoclass:: mptorch.quant.BlockQuant
   :members: __call__, pack, pack_fresh

Kernels
-------

See :doc:`kernels/index`.

The arithmetic of a dot product
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mptorch.quant.mac
   :members: SplitMac, FusedMac, Palette, BlockMac
   :no-undoc-members:

Matrix products
~~~~~~~~~~~~~~~

.. automodule:: mptorch.quant.matmul
   :members: qmatmul, qmm, qbmm, as_matmul_formats

.. autoclass:: mptorch.quant.QMatmulFormats

.. autoclass:: mptorch.quant.QMatmul
   :members: forward

.. autofunction:: mptorch.quant.matmul_formats

.. autofunction:: mptorch.quant.block_matmul

.. autofunction:: mptorch.quant.block_matmul_formats

The schema tier
~~~~~~~~~~~~~~~

One function per kernel, with every schema argument spelled out and no
autograd. See :doc:`kernels/calling` for when to use these instead of
``qmatmul``.

.. autofunction:: mptorch.quant.binaryK_matmul

.. autofunction:: mptorch.quant.superfp_matmul

.. autofunction:: mptorch.quant.binaryK_matmul_fma

.. autofunction:: mptorch.quant.superfp_matmul_fma

.. autofunction:: mptorch.quant.binaryK_matmul_mixed

.. autofunction:: mptorch.quant.superfp_matmul_mixed

.. autofunction:: mptorch.quant.binaryK_matmul_fma_mixed

.. autofunction:: mptorch.quant.superfp_matmul_fma_mixed

Layers
------

See :doc:`layers`.

.. autoclass:: mptorch.quant.QAffineFormats

.. autoclass:: mptorch.quant.QLinear
   :members: forward

.. autoclass:: mptorch.quant.QConv1d

.. autoclass:: mptorch.quant.QConv2d

.. autoclass:: mptorch.quant.QConv3d

Layer arithmetic factories
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mptorch.quant.binaryK_gemm_formats

.. autofunction:: mptorch.quant.binaryK_gemm_formats_fma

.. autofunction:: mptorch.quant.superfp_gemm_formats

.. autofunction:: mptorch.quant.superfp_gemm_formats_fma

.. autofunction:: mptorch.quant.conv_formats

.. autofunction:: mptorch.quant.block_gemm_formats
