"""Differentiable functions that round as their formats say (``torch.nn.functional``'s role).

Each function computes in a carrier and rounds each step whose slot is set,
and carries its own gradient; :mod:`mptorch.quant.modules` holds the layers
that call them, in files of the same names. Every public name is importable
from :mod:`mptorch.quant`, and this package re-exports none: its modules
import one another in an order a re-exporting ``__init__`` would break.
"""

# One module per operation:
#
#   pointwise    qadd, qsub, qmul, qdiv, qsqrt, qrsqrt, qexp, qpow, PointwiseFormats, and the slot,
#                carrier and fused-op helpers the other modules share
#   activation   qgelu, qsilu, qsigmoid, qtanh, qhardswish, qhardsigmoid, qrelu, ActivationFormats
#   embedding    qembedding
#   pwl          piecewise-linear tables: PWLTable, qpwl, FISR
#   reduce       qsum, qmean, ReductionFormats
#   norm         qsoftmax, qlayernorm, qrmsnorm, qgroupnorm, qbatchnorm, SoftmaxFormats, NormFormats
#   matmul       qmm, qbmm, qmatmul, and as_matmul_formats (the one name imported from its module)
#   attention    qattention, AttentionFormats, FlashFormats, and the nn.MultiheadAttention port
#   _flash       attention's flash path: FlashAttention-2's order, the loop and the batched path
#   blockwise    IEEE P3109's block operations on packed tensors, and two reductions
