from .block import (
    BlockPacked,
    block_matmul,
    block_pack,
    block_quantize,
    block_quantize_,
    block_unpack,
)
from .conv import conv_formats
from .gemm import (
    binaryK_gemm_formats,
    binaryK_gemm_formats_fma,
    block_gemm_formats,
    block_matmul_formats,
    matmul_formats,
    superfp_gemm_formats,
    superfp_gemm_formats_fma,
)
from .mac import BlockMac, BlockQuant, FusedMac, Palette, Quant, SplitMac
from .matmul import qbmm, qmatmul, qmm
from .modules import (
    QAffineFormats,
    QConv1d,
    QConv2d,
    QConv3d,
    QLinear,
    QMatmul,
    QMatmulFormats,
    Quantizer,
)
from .ops import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
    binaryK_quantize,
    binaryK_quantize_,
    superfp_matmul,
    superfp_matmul_fma,
    superfp_matmul_fma_mixed,
    superfp_matmul_mixed,
    superfp_quantize,
    superfp_quantize_,
)

__all__ = [
    # elementwise quantization
    "binaryK_quantize",
    "superfp_quantize",
    "binaryK_quantize_",
    "superfp_quantize_",
    "Quant",
    "Quantizer",
    # the schema tier: one function per op, every schema argument spelled out
    "binaryK_matmul",
    "superfp_matmul",
    "binaryK_matmul_fma",
    "superfp_matmul_fma",
    "binaryK_matmul_mixed",
    "superfp_matmul_mixed",
    "binaryK_matmul_fma_mixed",
    "superfp_matmul_fma_mixed",
    # dot-product arithmetic, as values
    "SplitMac",
    "FusedMac",
    "Palette",
    # block formats: packing, the block GEMM, and their value spellings
    "BlockPacked",
    "block_pack",
    "block_unpack",
    "block_quantize",
    "block_quantize_",
    "block_matmul",
    "BlockQuant",
    "BlockMac",
    # differentiable entry points
    "qmm",
    "qbmm",
    "qmatmul",
    # layers and their formats
    "QAffineFormats",
    "QMatmulFormats",
    "QLinear",
    "QMatmul",
    "QConv1d",
    "QConv2d",
    "QConv3d",
    "binaryK_gemm_formats",
    "superfp_gemm_formats",
    "binaryK_gemm_formats_fma",
    "superfp_gemm_formats_fma",
    "matmul_formats",
    "conv_formats",
    "block_matmul_formats",
    "block_gemm_formats",
]
