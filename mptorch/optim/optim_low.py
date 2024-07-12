import torch
from torch.optim import Optimizer, SGD, Adam

__all__ = ["OptimMP", "KahanSGD"]


class OptimMP(Optimizer):
    """
    A low-precision optimizer wrapper that handles weight, gradient, accumulator quantization.
    Args:
        - :attr: `optim`: underlying optimizer to use
        - :attr: `weight_quant`: a weight quantization function which takes a pytorch tensor and returns a tensor. If None, does not quantize weight.
        - :attr: `grad_quant`: a gradient quantization function which takes a pytorch tensor and returns a tensor. If None, does not quantize weight.
        - :attr: `grad_scaling`: float, scaling factor before apply gradient quantization.
        - :attr: `momentum_quant`: a momentum quantization function which takes a pytorch tensor and returns a tensor.
                                   If None, does not quantize weight.
        - :attr: `acc_quant`: a accumulator quantization function which takes
                              a pytorch tensor and returns a tensor. If not None, a
                              OptimLP object would create memory copies of model parameters that serve as
                              gradient accumulators. If None, does not use gradient accumulators.
    Example:
        >>> weight_q = quantizer(...) # define weight quantization
        >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer = OptimLP(optimizer, weight_quant=weight_q)
    """

    def __init__(
        self,
        optim,
        weight_quant=None,
        # grad_scaling=1.0,
        grad_quant=None,
        momentum_quant=None,
        acc_quant=None,
    ):
        assert isinstance(optim, SGD) or isinstance(optim, Adam)
        super(OptimMP, self).__init__(
            optim.param_groups, optim.defaults
        )  # place holder

        # python dictionary does not copy by default
        self.param_groups = optim.param_groups
        self.optim = optim

        # assert grad_scaling > 0, "gradient scaling must be positive"
        # self.grad_scaling = grad_scaling

        self.weight_quant = weight_quant
        self.grad_quant = grad_quant
        self.momentum_quant = momentum_quant
        self.acc_quant = acc_quant

        if isinstance(self.optim, SGD):
            self.momentum_keys = ["momentum_buffer"]
        elif isinstance(self.optim, Adam):
            # TODO: support amsgrad
            self.momentum_keys = ["exp_avg", "exp_avg_sq"]
        else:
            raise NotImplementedError("Only supporting Adam and SGD for now. ")

        if self.acc_quant != None:
            self.weight_acc = {}
            for group in self.param_groups:
                for p in group["params"]:
                    self.weight_acc[p] = p.detach().clone()

    def step(self, closure=None):
        """
        Performs one step of optimization with the underlying optimizer.
        Quantizes gradient and momentum before stepping. Quantizes gradient accumulator and weight after stepping.
        """
        # quantize gradient
        if not self.grad_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    # None gradient is equivalent to 0 gradient, skip
                    if p.grad is None:
                        continue
                    p.grad.data = self.grad_quant(p.grad.data)

        # switch acc into weight before stepping
        if not self.acc_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = self.weight_acc[p].data

        loss = self.optim.step()

        # switch weight into acc after stepping and quantize
        if not self.acc_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = self.weight_acc[p].data = self.acc_quant(p.data).data

        # quantize weight from acc
        if not self.weight_quant is None:
            for group in self.param_groups:
                for p in group["params"]:
                    p.data = self.weight_quant(p.data).data

        # quantize momentum
        if not self.momentum_quant is None:
            for group in self.param_groups:
                if isinstance(self.optim, SGD) and group["momentum"] == 0:
                    continue
                for p in group["params"]:
                    # None gradient is equivalent to 0 gradient, skip
                    if p.grad is None:
                        continue
                    param_state = self.optim.state[p]
                    for key in self.momentum_keys:
                        param_state[key] = self.momentum_quant(param_state[key])

        return loss

    def __repr__(self):
        return "MP Optimizer: {}".format(self.optim.__repr__())

    def __str__(self):
        return "MP Optimizer: {}".format(self.optim.__str__())


class KahanSGD(SGD):
    def __init__(self, params, lr=0.1, momentum=0, weight_decay=0, dampening=0, nesterov=False):
        super().__init__(params, lr, momentum, weight_decay, dampening, nesterov)
        self.c = []

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.c.append(torch.zeros_like(p.data))
                    

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        k = 0
        for i, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for j, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=(1 - dampening))
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                update = -lr * d_p

                t = update - self.c[k]
                temp = p.data + t
                self.c[k] = (temp - p.data) - t
                p.data = temp

                k += 1

        return loss
