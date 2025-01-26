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
                    param_state["compensated_buffer"] = torch.zeros_like(
                        p.data, device=p.data.device
                    )

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
                    if len(param_state) == 0 or "compensated_buffer" in param_state:
                        param_state["step"] = 0
                        # exponential moving average of gradient values
                        param_state["exp_avg"] = torch.zeros_like(
                            p.data, device=p.data.device
                        )
                        # exponential moving average of squared gradient values
                        param_state["exp_avg_sq"] = torch.zeros_like(
                            p.data, device=p.data.device
                        )
                        if amsgrad:
                            # maintains max of all exp. moving avg. of sq. grad. values
                            param_state["max_exp_avg_sq"] = torch.zeros_like(
                                p.data, device=p.data.device
                            )

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
                        param_state["exp_avg"] = exp_avg = self.momentum_quant(
                            exp_avg.mul_(beta1),
                            d_p,
                            (1.0 - beta1) * torch.ones_like(d_p, device=d_p.device),
                        )
                        param_state["exp_avg_sq"] = exp_avg_sq = self.momentum_quant(
                            exp_avg_sq.mul_(beta2),
                            d_p**2,
                            (1.0 - beta2) * torch.ones_like(d_p, device=d_p.device),
                        )
                    else:
                        exp_avg.mul_(beta1).add_(d_p, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1.0 - beta2)

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

                    # p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    if self.acc_quant is not None:
                        if "compensated_buffer" not in param_state:
                            p.data.addcdiv_(exp_avg, denom, value=-step_size)
                            p.data = self.acc_quant(
                                p.data,
                                torch.zeros_like(p.data, device=p.data.device),
                                torch.zeros_like(p.data, device=p.data.device),
                            )
                        else:
                            u = self.acc_quant(
                                exp_avg / denom,
                                torch.zeros_like(exp_avg, device=exp_avg.device),
                                torch.zeros_like(exp_avg, device=exp_avg.device),
                            )
                            y = -self.acc_quant(
                                param_state["compensated_buffer"],
                                u,
                                step_size * torch.ones_like(u, device=u.device),
                            )
                            s = self.acc_quant(
                                p.data, y, torch.ones_like(p.data, device=p.data.device)
                            )
                            param_state["compensated_buffer"] = self.acc_quant(
                                self.acc_quant(
                                    s, p.data, -torch.ones_like(s, device=s.device)
                                ),
                                y,
                                -torch.ones_like(y, device=y.device),
                            )
                            p.data = s
                    else:
                        if "compensated_buffer" not in param_state:
                            p.data.addcdiv_(exp_avg, denom, value=-step_size)
                        else:
                            u = -step_size * exp_avg / denom
                            y = u - param_state["compensated_buffer"]
                            s = p.data + y
                            param_state["compensated_buffer"] = (s - p.data) - y
                            p.data = s
