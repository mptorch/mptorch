import torch
import torch.nn as nn
from ..functional import qmean, qadd, qpow, qdiv, qsqrt, qmul

__all__ = ["QBatchNorm", "QBatchNorm1d", "QBatchNorm2d"]


def batch_norm(
    x, weight, bias, moving_mean, moving_var, eps, momentum, fwd_quant, bwd_quant
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
    def __init__(self, num_features, num_dims, fwd_quant, bwd_quant):
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

    def forward(self, x):
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
    def __init__(self, num_features, fwd_quant, bwd_quant):
        super().__init__(
            num_features, num_dims=2, fwd_quant=fwd_quant, bwd_quant=bwd_quant
        )


class QBatchNorm2d(QBatchNorm):
    def __init__(self, num_features, fwd_quant, bwd_quant):
        super().__init__(
            num_features, num_dims=4, fwd_quant=fwd_quant, bwd_quant=bwd_quant
        )
