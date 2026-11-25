"""The low-level tier: one plain function per kernel op, every op argument spelled out.

Each function here calls one ``torch.ops.mptorch`` op and carries no gradient:
the elementwise quantizers (``binaryK_quantize`` and its counterparts), one
wrapper per GEMM op (``binaryK_matmul`` and its variants, the cross-family
GEMMs, and ``fixedpoint_matmul`` and ``fixedpoint_matmul_fma``), and the block
format ops (``block_pack``, ``block_unpack``, ``block_quantize``,
``block_project`` and ``block_matmul``). The tiers above build on these: the format objects of
:mod:`mptorch.quant.mac`, the differentiable matrix products of
:mod:`mptorch.quant.matmul`, and the layer factories of
:mod:`mptorch.quant.gemm` and :mod:`mptorch.quant.conv`. Every public name is
also importable from :mod:`mptorch.quant`.
"""

# One module per format family, over the shared modules they build on:
#
#   binaryk      binaryK_quantize(_), binaryK_matmul{,_fma}{,_mixed}, and their spec builders
#   superfp      the superfp counterparts
#   fixedpoint   fixedpoint_quantize(_), fixedpoint_matmul{,_fma}, and their spec builders
#   cross        the cross-family GEMMs: binaryK_superfp_matmul, superfp_binaryK_matmul
#   block        the block format ops: block_pack, block_unpack, block_quantize(_), block_project,
#                block_matmul
#   _checks      each call's checks of its formats against the carrier and the storage dtype
#   _spec        format defaults, the resolved GEMM call (_GemmSpec), and running it
#   _accumulate  KAHAN, BLOCK and TREE: the arguments of the *_accumulated ops
#   _odd         the carrier step rounded to odd: the checks of the *_odd ops
#   _operands    torch.matmul's operand rules, mapped onto the rank-2 and rank-3 ops
#   _conv        the conv ops: the table from each GEMM op to its conv op, and the call
#
# A new family is a module of its own (its wrappers and their spec builders),
# a line in this table, and its public names imported below. Code inside
# mptorch imports a private name from the module that defines it.

from .binaryk import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
    binaryK_quantize,
    binaryK_quantize_,
)
from .block import (
    BlockPacked,
    block_matmul,
    block_pack,
    block_project,
    block_quantize,
    block_quantize_,
    block_unpack,
)
from .cross import binaryK_superfp_matmul, superfp_binaryK_matmul
from .fixedpoint import (
    fixedpoint_matmul,
    fixedpoint_matmul_fma,
    fixedpoint_quantize,
    fixedpoint_quantize_,
)
from .superfp import (
    superfp_matmul,
    superfp_matmul_fma,
    superfp_matmul_fma_mixed,
    superfp_matmul_mixed,
    superfp_quantize,
    superfp_quantize_,
)

__all__ = [
    "binaryK_quantize",
    "binaryK_quantize_",
    "superfp_quantize",
    "superfp_quantize_",
    "fixedpoint_quantize",
    "fixedpoint_quantize_",
    "binaryK_matmul",
    "superfp_matmul",
    "binaryK_matmul_fma",
    "superfp_matmul_fma",
    "binaryK_matmul_mixed",
    "superfp_matmul_mixed",
    "binaryK_matmul_fma_mixed",
    "superfp_matmul_fma_mixed",
    "binaryK_superfp_matmul",
    "superfp_binaryK_matmul",
    "fixedpoint_matmul",
    "fixedpoint_matmul_fma",
    "BlockPacked",
    "block_matmul",
    "block_pack",
    "block_quantize",
    "block_quantize_",
    "block_project",
    "block_unpack",
]
