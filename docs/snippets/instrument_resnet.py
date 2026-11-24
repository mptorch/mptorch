"""A torchvision ResNet-18, converted to an 8-bit recipe, and what its products cost."""

import torch
import torchvision

from mptorch import BinaryK
from mptorch.quant import convert, presets, profile

torch.manual_seed(0)
model = torchvision.models.resnet18().eval()  # weights="DEFAULT" for the pretrained ones
p8 = BinaryK(8, 4)
recipe = presets.uniform(p8, acc=BinaryK(16, 11)).override("fc", linear=None)
qmodel, report = convert(model, recipe, strict=True)
print(f"{len(report.converted)} layers converted, {len(report.kept)} kept: {report.kept[0].name}")

x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    difference = (qmodel(x) - model(x)).abs().max()
print(f"largest difference of the logits: {difference:.3f}\n")

# a Binary8p4 product priced at a quarter of a float32 one, the forward and the backward
print(profile(qmodel, x[:1], cost={p8: 1.0, None: 4.0}, backward=True).summary())
