from argparse import ArgumentError
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant_format import QAffineFormats
from ..functional import qlinear, qlinear_mp, qlinear_mpv2, qlinear_kernel

__all__ = ["QLinear", "QLazyLinear", "QLinearMP"]


def kappa_phi(v, g, phi):
    return torch.div(torch.abs(v) * torch.abs(g), torch.abs(phi)).nan_to_num(0)


def kappa_v(w, b, x):
    if b is None:
        return torch.div(torch.abs(x) @ torch.abs(w.T), torch.abs(x @ w.T))
    else:
        return torch.div(
            torch.abs(x) @ torch.abs(w.T) + torch.abs(b), torch.abs(x @ w.T + b)
        )


def kappa_v_mp(w, b, x, v, formats, s):
    return torch.div(
        qlinear_mp.apply(torch.abs(x), torch.abs(w), torch.abs(b), formats, s),
        torch.abs(v),
    )


def kappa_v_mpv2(w, b, x, v, formats, s, mans, exps):
    return torch.div(
        qlinear_mpv2.apply(
            torch.abs(x), torch.abs(w), torch.abs(b), formats, s, mans, exps
        ),
        torch.abs(v),
    )


def kappa_v_2dmp(w, b, x, v, formats, s):
    assert len(v.shape) == 3
    v = v.view(v.shape[0] * v.shape[1], v.shape[2])
    return torch.div(
        qlinear_mp.apply(torch.abs(x), torch.abs(w), torch.abs(b), formats, s),
        torch.abs(v),
    )


class QLinear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y=xW^T + b`

    It is a subclass of :class:`torch.nn.Linear` and allows one to specify if I/O
    signals should be quantized during inference & training (needed for instance
    in QAT and PTQ methods), but also the precision(s) to be used in internal GEMM
    computations (addition and multiplication, fused or not). This allows simulating
    the effect of custom precision during GEMM calls in the forward and backward pass
    and is helpful in studying the effect of low precision compute during inference
    and training (not just data quantization).

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        formats: number formats used during compute (addition and multiplication) and
            quantization functions for signals during forward and back propagation (I/O
            activations, weights, biases, and neural gradients)
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out_features}, \text{in_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in_features}}`

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        formats: QAffineFormats,
        bias: bool = True,
    ) -> None:
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self):
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(self, fwd_quant):
        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input):
        if self.formats.fwd_use_default_prec and self.formats.bwd_use_default_prec:
            return self.Qo(
                F.linear(self.Qi(input), self.Qw(self.weight), self.Qb(self.bias))
            )
        else:
            return qlinear(input, self.weight, self.bias, self.formats)


class QLazyLinear(torch.nn.modules.lazy.LazyModuleMixin, QLinear):
    r"""A linear module where `in_features` is inferred.

    In this module (an analogue to :class:`torch.nn.LazyLinear`), the
    ``in_features`` parameter of the quantized linear layer is inferred
    from the input's last dimension (i.e., ``input.shape[-1]``).

    Args:
        out_features: size of each output sample
        formats: number formats used during compute (addition and multiplication) and
            quantization functions for signals during forward and back propagation (I/O
            activations, weights, biases, and neural gradients)
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """

    cls_to_become = QLinear
    weight: torch.nn.UninitializedParameter
    r"""
        The learnable weights of the module of shape
        :math:`(\text{out_features}, \text{in_features})`. The values are
        initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
        :math:`k = \frac{1}{\text{in_features}}`
    """
    bias: torch.nn.UninitializedParameter
    r"""
        The learnable bias of the module of shape :math:`(\text{out_features})`.
        If :attr:`bias` is ``True``, the values are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{1}{\text{in_features}}`
    """

    def __init__(
        self,
        out_features: int,
        formats: QAffineFormats,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # similar to nn.LazyLinear, bias is hardcoded to avoid
        # creating tensor that will soon be overwritten
        super().__init__(0, 0, formats=formats, bias=False)
        self.weight = torch.nn.UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            self.bias = torch.nn.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()


class QLinearMP(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        formats: QAffineFormats,
        bias: bool = True,
        activation=None,
        d_activation=None,
        kappa_phi_f=kappa_phi,
        kappa_v_f=kappa_v_mpv2,
        tol=0.1,
    ) -> None:
        super(QLinearMP, self).__init__(in_features, out_features, bias)
        self.formats = formats

        if activation is not None and d_activation is None:
            raise ArgumentError(
                message="d_activation is required when using mixed precision"
            )

        self.activation = activation
        self.d_activation = d_activation
        self.tol = tol
        self.kappa_phi_f = kappa_phi_f
        self.kappa_v_f = kappa_v_f

        self.reset_quant_function()

    def reset_quant_function(self):
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self):
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(self, fwd_quant):
        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input):
        if self.formats.fwd_use_default_prec:
            return self.Qo(
                F.linear(self.Qi(input), self.Qw(self.weight), self.Qb(self.bias))
            )
        else:
            w = self.Qw(self.weight)
            bias = self.Qb(self.bias)
            mans = torch.tensor([23, 10, 3, 1], dtype=torch.int32, device=w.device)
            exps = torch.tensor([8, 5, 4, 2], dtype=torch.int32, device=w.device)

            prec = (
                torch.ones(input.shape[0], w.shape[0], device=w.device).int()
                * self.formats.s
            ).to(w.device)

            prec_phi = (
                torch.ones(input.shape[0], w.shape[0], device=w.device).int()
                * self.formats.s
            ).to(w.device)

            # analysis with kappa'
            if self.activation is not None:
                prec_fp8 = (torch.ones(input.shape[0], w.shape[0]).int() * 2).to(
                    w.device
                )
                out_fp8 = qlinear_mpv2.apply(
                    input, w, bias, self.formats, prec_fp8, mans, exps
                )
                # out_fp8 = qlinear_mp.apply(input, w, bias, self.formats, prec_fp8)

                kappa_phi_fp8 = self.kappa_phi_f(
                    out_fp8, self.d_activation(out_fp8), self.activation(out_fp8)
                )
                kappa_v_fp8 = self.kappa_v_f(
                    w, bias, input, out_fp8, self.formats, prec_fp8, mans, exps
                )

                deno_fp8 = 1 / torch.abs(out_fp8)
                num_fp8 = qlinear_mpv2.apply(
                    torch.abs(input),
                    torch.abs(w),
                    torch.abs(bias),
                    self.formats,
                    prec_fp8,
                    mans,
                    exps,
                )
                # num_fp8 = qlinear_mp.apply(
                #     torch.abs(input),
                #     torch.abs(w),
                #     torch.abs(bias),
                #     self.formats,
                #     prec_fp8,
                # )
                self.deno_fp8 = deno_fp8.to(w.device)
                self.num_fp8 = num_fp8.to(w.device)

                # relu mask
                prec_phi = torch.where(kappa_phi_fp8 == 0, 2, 1).int()

                prec = torch.where(kappa_phi_fp8 * deno_fp8 <= self.tol, 2, 1).int()
                prec_fp8_mask = torch.where(
                    kappa_phi_fp8 * deno_fp8 <= self.tol, 1, 0
                ).int()

                prec_fp16_mask = torch.ones_like(prec_fp8_mask, dtype=torch.int)
                prec_fp16_mask -= prec_fp8_mask

                prec_fp16 = torch.ones(input.shape[0], w.shape[0]).int().to(w.device)
                out_fp16 = qlinear_mpv2.apply(
                    input, w, bias, self.formats, prec_fp16, mans, exps
                )
                # out_fp16 = qlinear_mp.apply(input, w, bias, self.formats, prec_fp16)
                out = out_fp8 * prec_fp8_mask + out_fp16 * prec_fp16_mask

                # store variables for experiments
                self.kappa_v = kappa_v_fp8
                self.kappa_phi = kappa_phi_fp8
                self.kappa_fp8 = kappa_v_fp8 * kappa_phi_fp8
                self.kappa_prime = deno_fp8 * kappa_phi_fp8

                self.formats.s = prec
                self.prec = prec
                self.prec_phi = prec_phi
            else:
                self.formats.s = prec.to(w.device)
                self.prec = prec.to(w.device)
                out = qlinear_mpv2.apply(input, w, bias, self.formats, prec, mans, exps)
                # out = qlinear_mp.apply(input, w, bias, self.formats, prec)

            self.prec = prec
            self.count = torch.where(prec == 2, 1, 0).sum(tuple(range(len(prec.shape))))
            self.count_phi = torch.where(prec_phi == 2, 1, 0).sum(
                tuple(range(len(prec_phi.shape)))
            )

            return out
