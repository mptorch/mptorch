"""Freezing block-format QLinear layers to their codes, and loading them back."""

import dataclasses
import io

import torch
from torch import nn

from mptorch import NVFP4, BinaryK
from mptorch.quant import QLinear, block_gemm_formats, freeze_packed_

w16 = dataclasses.replace(NVFP4, block_rows=16)


def build(device=None):
    def layer(n_in, n_out):
        formats = block_gemm_formats(NVFP4, w16, acc=BinaryK(16, 11))
        return QLinear(n_in, n_out, device=device, formats=formats)

    return nn.Sequential(layer(1024, 1024), nn.ReLU(), layer(1024, 256))


def mib(model):
    tensors = [*model.parameters(), *model.buffers()]
    return sum(t.nbytes for t in tensors) / 2**20


torch.manual_seed(0)
model = build()
x = torch.randn(8, 1024)
with torch.no_grad():
    before = model(x)
print(f"float weights: {mib(model):.3f} MiB")

print("layers frozen:", freeze_packed_(model))
print(f"frozen:        {mib(model):.3f} MiB")
with torch.no_grad():
    print("same output:", torch.equal(model(x), before))
print("weight:", model[0].weight, "| buffers:", [name for name, _ in model[0].named_buffers()])

# Loading: build the model as it was built, here on the meta device so that no
# float weight is ever allocated, freeze it without packing, and load the codes.
file = io.BytesIO()
torch.save(model.state_dict(), file)
file.seek(0)
loaded = build(device="meta")
freeze_packed_(loaded, pack=False)
loaded.load_state_dict(torch.load(file, weights_only=True), assign=True)
with torch.no_grad():
    print("loaded output:", torch.equal(loaded(x), before))
