"""The named formats of OCP MX v1.0 and NVIDIA's NVFP4, and :func:`bfp` for block floating point."""

from .binaryk import BinaryK
from .block import BlockFormat
from .fixedpoint import FixedPoint
from .modes import SubnormalsMode

# The element and scale formats of OCP MX v1.0 and NVFP4, written as the
# binaryK formats with the same bit layout. OCP uses IEEE 754's biases, one
# below P3109's defaults.
E8M0 = BinaryK(8, 1, bias=127, is_signed=False, subnormals=SubnormalsMode.EXTENDED_NORMALS)
E4M3 = BinaryK(8, 4, bias=7)
E5M2 = BinaryK(8, 3, bias=15)
E2M1 = BinaryK(4, 2, bias=1)
E2M3 = BinaryK(6, 4, bias=1)
E3M2 = BinaryK(6, 3, bias=3)

# OCP MX v1.0's concrete formats (blocks of 32, E8M0 scale) and NVFP4 (blocks
# of 16, E4M3 scale, plus a float32 per-tensor scale). OCP's E4M3 reserves a
# code for NaN, and its E5M2 codes for NaN and infinity, so their largest
# values, given as elem_max, are below the binaryK layout's SAT_FINITE maximum.
MXFP8_E4M3 = BlockFormat(E4M3, E8M0, 32, elem_max=448.0, nan_code=0x7F)
MXFP8_E5M2 = BlockFormat(E5M2, E8M0, 32, elem_max=57344.0, nan_code=0x7F, inf_code=0x7C)
MXFP6_E2M3 = BlockFormat(E2M3, E8M0, 32)
MXFP6_E3M2 = BlockFormat(E3M2, E8M0, 32)
MXFP4_E2M1 = BlockFormat(E2M1, E8M0, 32)
NVFP4 = BlockFormat(E2M1, E4M3, 16)


def bfp(bits: int, block_size: int = 16, *, block_rows: int = 1) -> BlockFormat:
    """Block floating point: ``bits``-bit integer elements that share a power-of-two scale.

    The elements are ``FixedPoint(bits, bits - 2, symmetric=True)``: a sign
    bit and ``bits - 1`` magnitude bits, evenly spaced ``2**-(bits - 2)``
    apart from zero up to the largest value, ``2 - 2**-(bits - 2)``. So
    ``emax`` is 0, and OCP's rule gives a block the E8M0 scale
    ``2**floor(log2(amax))``. Any other ``fl`` gives the same block values and
    changes only ``emax``; build the :class:`BlockFormat` directly to use one.

    ``bfp(8, 32)`` is OCP's MXINT8 except for one code: MXINT8's elements are
    two's complement, and their most negative value, ``-2``, has no
    sign-magnitude code. The block GEMM takes element codes of at most 8 bits,
    so ``bfp(9)`` to ``bfp(16)`` can be packed, unpacked and quantized, but
    not multiplied.

    Args:
        bits: the element's word length, including the sign bit: 2 to 16.
        block_size: elements per block: a power of two in [2, 128], with
            ``bits * block_size`` a multiple of 8.
        block_rows: rows per tile; ``block_rows > 1`` shares one scale over a
            ``block_rows x block_size`` tile.
    """
    return BlockFormat(
        FixedPoint(bits, bits - 2, symmetric=True), E8M0, block_size, block_rows=block_rows
    )
