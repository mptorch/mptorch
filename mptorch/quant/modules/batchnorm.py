from typing import Callable
import torch
import torch.nn as nn
from ..functional import qmean, qadd, qpow, qdiv, qsqrt, qmul

__all__ = ["QBatchNorm1d", "QBatchNorm2d"]

# TODO: arrive at parity with the PyTorch baseline implementation, add support for the 3D version


def batch_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    moving_mean: torch.Tensor,
    moving_var: torch.Tensor,
    eps: float,
    momentum: float,
    fwd_quant: Callable[[torch.Tensor], torch.Tensor],
    bwd_quant: Callable[[torch.Tensor], torch.Tensor],
):
    if not torch.is_grad_enabled():
        x_hat = fwd_quant(
            fwd_quant(x - moving_mean)
            / fwd_quant(torch.sqrt(fwd_quant(moving_var + eps)))
        )
    else:
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            mean = qmean(x, fwd_quant, bwd_quant, 0, False)
            var = qmean(
                qpow(
                    qadd(x, -mean, fwd_quant, bwd_quant),
                    fwd_quant,
                    bwd_quant,
                    2,
                ),
                fwd_quant,
                bwd_quant,
                0,
                False,
            )
        else:
            mean = qmean(x, fwd_quant, bwd_quant, (0, 2, 3), True)
            var = qmean(
                qpow(
                    qadd(x, -mean, fwd_quant, bwd_quant),
                    fwd_quant,
                    bwd_quant,
                    2,
                ),
                fwd_quant,
                bwd_quant,
                (0, 2, 3),
                True,
            )
        x_hat = qdiv(
            qadd(x, -mean, fwd_quant, bwd_quant),
            qsqrt(var + eps, fwd_quant, bwd_quant),
            fwd_quant,
            bwd_quant,
        )
        # moving mean and moving average do not have gradients that need to be recorded
        mfactor = fwd_quant(torch.tensor(1.0 - momentum, device=x.device))
        moving_mean = fwd_quant(momentum * moving_mean)
        diff_mean = fwd_quant(mfactor * mean)
        moving_mean = fwd_quant(moving_mean + diff_mean)
        moving_var = fwd_quant(momentum * moving_var)
        diff_var = fwd_quant(mfactor * var)
        moving_var = fwd_quant(moving_var + diff_var)

    y = qadd(
        qmul(weight, x_hat, fwd_quant, bwd_quant),
        bias,
        fwd_quant,
        bwd_quant,
    )
    return y, moving_mean.data, moving_var.data


class QBatchNorm(nn.Module):

    def __init__(
        self,
        num_features: int,
        num_dims: int,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()

        self.fwd_quant = fwd_quant
        self.bwd_quant = bwd_quant
        if num_dims == 2:
            self.shape = num_features
        else:
            self.shape = (num_features, 1, 1)

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.moving_mean = torch.zeros(self.shape)
        self.moving_var = torch.ones(self.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Applies the batch normalization operation on the input tensor.

        Args:
            x: input tensor

        Returns:
            the result of the batch normalization operation over the input. Behaves differently
            depending if the module is in inference or train mode.
        """
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)

        y, self.moving_mean, self.moving_var = batch_norm(
            x,
            self.weight.view(self.shape),
            self.bias.view(self.shape),
            self.moving_mean,
            self.moving_var,
            eps=1e-5,
            momentum=0.9,
            fwd_quant=self.fwd_quant,
            bwd_quant=self.bwd_quant,
        )
        return y


class QBatchNorm1d(QBatchNorm):
    r"""Applies Batch Normalization over a 2D input.

    Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0.
    At train time in the forward pass, the variance is calculated via the biased estimator,
    equivalent to ``torch.var(input, unbiased=False)``. However, the value stored in the
    moving average of the variance is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default momentum of 0.1.

    .. note::
        This momentum is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`, where :math:`N` is the batch size,
          :math:`C` is the number of features or channels, and :math:`L` is the sequence length
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""Because the Batch Normalization is done over the `C` dimension, computing statistics
        on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

        Args:
            num_features: number of features or channels :math:`C` of the input
            fwd_quant: quantization function to use during FWD operations
            bwd_quant: quantization function to use during BWD operations
        """
        super().__init__(
            num_features, num_dims=2, fwd_quant=fwd_quant, bwd_quant=bwd_quant
        )


class QBatchNorm2d(QBatchNorm):
    r"""Applies Batch Normalization over a 4D input.

    4D is a mini-batch of 2D inputs
    with additional channel dimension. Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, unbiased=False)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default momentum
    of 0.1.

    .. note::
        This momentum different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, H, W)`
            fwd_quant: quantization function to use during FWD operations
            bwd_quant: quantization function to use during BWD operations
        """
        super().__init__(
            num_features, num_dims=4, fwd_quant=fwd_quant, bwd_quant=bwd_quant
        )
