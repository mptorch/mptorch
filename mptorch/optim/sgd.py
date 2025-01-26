import torch
from torch.optim import SGD

__all__ = ["QSGD"]


# TODO: I'm not yet sure how this version plays around with grad scaling; some more
# investigation is needed (see `grad_scale` and `found_inf` parameters in:
# https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD)
# TODO: later refactor: investigate calling torch.SGD directly if quantizer objects
# are not specified (good to reimplement and check against torch defaults for now)
# TODO: there is not a one-to-one match between QOptim and QSGD when using them in the
# same update quantization mode; this requires more investigation
# TODO: `maximize` behavior is currently not implemented
# NOTE: `momentum_quant` and `acc_quant` should either be None (in which case no
# LP weight and momentum updates are done), or they should be trivariate functions that
# simulate an FMA operation, meaning that if x, y and z are the parameters, they should
# compute Q(x + y*z). This interface seems to be quite flexible, since it allows one
# to simulate various kinds of quantizers more easily.
# NOTE: compensated accumulation is supported, with a compensated buffer being stored
# inside the optimizer; it is enabled through the `compensated` flag; note that the
# acc_quant function will be used inside the compensated (Kahan) summation for the
# parameter updates, so it is the user's responsibility to choose a quantizer that works
# well in a compensated setting (precision-p quantization is a recommended default)
# NOTE: if no quantizers are specified, the behaviour of this optimizer should be the same
# as that of the PyTorch vanilla SGD; that being said, it shouldn't be used if no quantizers
# are specified
# NOTE: we should probably think of an extension to `momentum_quant` that in fact works with
# 2D dot product x*y + z*t, which would be useful when a dampening coefficient is specified
class QSGD(SGD):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        momentum_quant=None,
        acc_quant=None,
        compensated=False,
    ):

        super(QSGD, self).__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.momentum_quant = momentum_quant
        self.acc_quant = acc_quant

        if compensated:
            for group in self.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["compensated_buffer"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # disable the autograd engine while we are performing the updates
        with torch.no_grad():
            for group in self.param_groups:
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                lr = group["lr"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data

                    if weight_decay != 0:
                        # NOTE: this needs to be potentially quantized as well
                        d_p.add_(p.data, alpha=weight_decay)

                    if momentum != 0:
                        param_state = self.state[p]
                        if "momentum_buffer" not in param_state:
                            # initialize the momentum buffer
                            if self.momentum_quant is not None:
                                buf = param_state["momentum_buffer"] = (
                                    self.momentum_quant(
                                        torch.clone(d_p).detach(),
                                        torch.zeros_like(d_p, device=d_p.device),
                                        torch.zeros_like(d_p, device=d_p.device),
                                    )
                                )
                            else:
                                buf = param_state["momentum_buffer"] = torch.clone(
                                    d_p
                                ).detach()
                        else:
                            # update the momentum buffer, optionally quantizing it
                            buf = param_state["momentum_buffer"]
                            if self.momentum_quant is not None:
                                buf = param_state["momentum_buffer"] = (
                                    self.momentum_quant(
                                        buf.mul_(momentum),
                                        d_p,
                                        (1.0 - dampening)
                                        * torch.ones_like(d_p, device=d_p.device),
                                    )
                                )
                            else:
                                buf.mul_(momentum).add_(d_p, alpha=(1.0 - dampening))

                        # apply Nesterov momentum (if enabled)
                        if nesterov:
                            # NOTE: this needs to be potentially quantized as well
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    # update parameters, potentially quantizing the updated result
                    if self.acc_quant is not None:
                        # NOTE: update rule is
                        # w_{k+1} = Q_w(w_{k} - lr * g_{k})
                        if "compensated_buffer" not in param_state:
                            p.data = self.acc_quant(
                                p.data,
                                d_p,
                                -lr * torch.ones_like(p.data, device=p.data.device),
                            )
                        else:
                            y = -self.acc_quant(
                                param_state["compensated_buffer"],
                                d_p,
                                lr * torch.ones_like(d_p, device=d_p.device),
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
                        # NOTE: update rule is
                        # w_{k+1} = w_{k} - lr * g_{k}
                        if "compensated_buffer" not in param_state:
                            p.add_(d_p, alpha=-lr)
                        else:
                            u = -lr * d_p
                            y = u - param_state["compensated_buffer"]
                            s = p.data + y
                            param_state["compensated_buffer"] = (s - p.data) - y
                            p.data = s

            return loss
