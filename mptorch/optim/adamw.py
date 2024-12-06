import torch
from torch.optim import AdamW
from typing import Callable
import math

__all__ = ["QAdamW"]

# NOTE: for inspiration, see:
# https://github.com/huggingface/transformers/blob/main/src/transformers/optimization.py


class QAdamW(AdamW):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        acc_quant=None,
        momentum_quant=None,
        compensated=False,
    ):
        super(QAdamW, self).__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.acc_quant = acc_quant
        self.momentum_quant = momentum_quant

        if compensated:
            for group in self.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["compensated_buffer"] = torch.zeros_like(p.data)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # disable the autograd engine while we are performing the updates
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:

                    amsgrad = group["amsgrad"]

                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if d_p.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")

                    param_state = self.state[p]

                    # state initialization
                    if len(param_state) == 0:
                        param_state["step"] = 0
                        # exponential moving average of gradient values
                        param_state["exp_avg"] = torch.zeros_like(p.data)
                        # exponential moving average of squared gradient values
                        param_state["exp_avg_sq"] = torch.zeros_like(p.data)
                        if amsgrad:
                            # maintains max of all exp. moving avg. of sq. grad. values
                            param_state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = (
                        param_state["exp_avg"],
                        param_state["exp_avg_sq"],
                    )
                    if amsgrad:
                        max_exp_avg_sq = param_state["max_exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    # update step
                    param_state["step"] += 1

                    # decay the first and second moment running average coefficient
                    if self.momentum_quant is not None:
                        exp_avg = param_state["exp_avg"] = self.momentum_quant(
                            exp_avg.mul_(beta1), d_p, 1.0 - beta1
                        )
                        exp_avg_sq = param_state["exp_avg_sq"] = self.momentum_quant(
                            exp_avg_sq.mul_(beta2), d_p**2, 1.0 - beta2
                        )
                    else:
                        exp_avg.mul_(beta1).add_(1.0 - beta1, d_p)
                        exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, d_p, d_p)

                    bias_correction1 = 1.0 - beta1 ** param_state["step"]
                    bias_correction2 = 1.0 - beta2 ** param_state["step"]
                    step_size = group["lr"] / bias_correction1

                    if amsgrad:
                        # maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # use the max. for normalizing running avg. of gradient
                        denom = (
                            max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                        ).add_(group["eps"])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                            group["eps"]
                        )

                    # TODO: quantize
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
