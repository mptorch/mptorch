"""An embedding whose rows and weight gradient round as a ``PointwiseFormats`` says."""

from typing import Any

import torch
from torch import nn

from ..functional.embedding import qembedding


class QEmbedding(nn.Embedding):
    """:class:`torch.nn.Embedding` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qembedding`): ``formats.quant`` rounds the rows it
    gathers, and ``formats.grad_quant`` its weight's gradient, after the sum
    over repeated indices. ``formats=None`` is the torch module's forward. It
    takes :class:`torch.nn.Embedding`'s arguments, and ``formats``.
    """

    _mptorch_owned = True

    def __init__(self, *args: Any, formats: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The rows of ``input``'s indices, rounded as ``formats`` says
        (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return qembedding(
            input,
            self.weight,
            self.formats,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
