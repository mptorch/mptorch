"""Norm modules that round as a ``NormFormats`` says.

Each class is its torch class with a ``formats`` attribute: ``None`` runs the
torch module's own forward, and a :class:`~mptorch.quant.NormFormats` runs
the matching ``mptorch.quant.norm`` op. Their parameters, buffers and
``state_dict`` are torch's. :func:`~mptorch.quant.convert_` turns torch's
norm modules into these.
"""

from typing import Any

import torch
from torch import nn

# A module reference rather than names: `norm` imports this package for
# `_opaque`, and the names are read when a forward runs.
from .. import norm


class QLayerNorm(nn.LayerNorm):
    """:class:`torch.nn.LayerNorm` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qlayernorm`)."""

    _mptorch_owned = True

    def __init__(self, *args: Any, formats: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The norm, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return norm.qlayernorm(
            input, self.normalized_shape, self.weight, self.bias, self.eps, self.formats
        )


class QRMSNorm(nn.RMSNorm):
    """:class:`torch.nn.RMSNorm` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qrmsnorm`)."""

    _mptorch_owned = True

    def __init__(self, *args: Any, formats: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.formats = formats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The norm, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(x)
        return norm.qrmsnorm(x, self.normalized_shape, self.weight, self.eps, self.formats)


class QGroupNorm(nn.GroupNorm):
    """:class:`torch.nn.GroupNorm` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qgroupnorm`)."""

    _mptorch_owned = True

    def __init__(self, *args: Any, formats: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The norm, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return norm.qgroupnorm(
            input, self.num_groups, self.weight, self.bias, self.eps, self.formats
        )


class _QBatchNorm:
    """The forward of the three batch norm modules: torch's ``_BatchNorm``
    forward (the momentum, ``num_batches_tracked``, the choice of the
    batch's statistics or the running ones), then :func:`~mptorch.quant.qbatchnorm`."""

    formats: Any

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The norm, rounded as ``formats`` says (torch's with ``None``)."""
        bn = self
        assert isinstance(bn, nn.modules.batchnorm._BatchNorm)
        if self.formats is None:
            return nn.modules.batchnorm._BatchNorm.forward(bn, input)
        # the layer's call, and none of the calls it makes, is what a torch function mode sees
        with torch._C.DisableTorchFunction():
            return self._rounded(bn, input)

    def _rounded(self, bn: nn.modules.batchnorm._BatchNorm, input: torch.Tensor) -> torch.Tensor:
        bn._check_input_dim(input)
        factor = 0.0 if bn.momentum is None else bn.momentum
        if bn.training and bn.track_running_stats and bn.num_batches_tracked is not None:
            bn.num_batches_tracked.add_(1)
            if bn.momentum is None:  # a cumulative moving average
                factor = 1.0 / float(bn.num_batches_tracked)
        if bn.training:
            training = True
        else:
            training = bn.running_mean is None and bn.running_var is None
        keep = not bn.training or bn.track_running_stats
        return norm.qbatchnorm(
            input,
            bn.running_mean if keep else None,
            bn.running_var if keep else None,
            bn.weight,
            bn.bias,
            training,
            factor,
            bn.eps,
            self.formats,
        )


class QBatchNorm1d(_QBatchNorm, nn.BatchNorm1d):
    """:class:`torch.nn.BatchNorm1d` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qbatchnorm`)."""

    _mptorch_owned = True

    def __init__(self, *args: Any, formats: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.formats = formats


class QBatchNorm2d(_QBatchNorm, nn.BatchNorm2d):
    """:class:`torch.nn.BatchNorm2d` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qbatchnorm`)."""

    _mptorch_owned = True

    def __init__(self, *args: Any, formats: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.formats = formats


class QBatchNorm3d(_QBatchNorm, nn.BatchNorm3d):
    """:class:`torch.nn.BatchNorm3d` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qbatchnorm`)."""

    _mptorch_owned = True

    def __init__(self, *args: Any, formats: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.formats = formats
