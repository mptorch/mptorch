"""A torchvision ViT-B/16 by simulation: fp8 products, its attention included."""

import torch
import torchvision
from torch import nn

from mptorch import BinaryK
from mptorch.quant import AttentionFormats, SplitMac, presets, simulate

torch.manual_seed(0)
model = torchvision.models.vit_b_16().eval()  # weights="DEFAULT" for the pretrained ones
nn.init.normal_(model.heads.head.weight, std=0.02)  # torchvision starts the head at zero
attention = AttentionFormats.uniform(SplitMac(BinaryK(8, 4), BinaryK(16, 11)))
recipe = presets.fp8().override("encoder.layers.*.self_attention", attention=attention)

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    reference = model(x)
    with simulate(recipe, model) as sim:
        simulated = model(x)
report = sim.report()
print(f"largest difference of the logits: {(simulated - reference).abs().max():.3f}")
print(f"report: {len(report.simulated)} rows simulated, {len(report.not_covered)} not covered")
for row in report.simulated[:4]:  # each row's formats are in row.formats
    print(f"  {row.scope or '(model)'}: {row.kind}, {row.call}")
