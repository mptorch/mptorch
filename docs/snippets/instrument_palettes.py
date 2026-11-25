"""A format per output neuron: a palette recipe, maps set after conversion, and their cost."""

import torch
from torch import nn

from mptorch import BinaryK
from mptorch.quant import (
    ActivationFormats,
    FusedMac,
    GemmFormats,
    Palette,
    Quant,
    Recipe,
    convert,
    palette_maps,
    profile,
    set_palette_maps_,
)

low, high = BinaryK(8, 4), BinaryK(16, 11)  # E4M3's widths, and float16's
recipe = Recipe(
    linear=GemmFormats(
        weight_quant=Quant(low),
        bias_quant=Quant(low),
        fwd_mac=FusedMac(Palette([low, high])),  # each neuron's products in its entry
        output_quant=Palette([low, high]),  # and its output rounded to it
    ),
    activation=ActivationFormats(out=Quant(low)),
)
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
qmodel, _ = convert(model, recipe)
x = torch.randn(256, 32)

# no map yet: every neuron computes in entry 0; then a seeded quarter of each
# layer's neurons in the high format; then all of them
g = torch.Generator().manual_seed(1)
quarter = {name: torch.rand(n, generator=g) < 0.25 for name, n in (("0", 64), ("2", 10))}
everyone = {"0": torch.ones(64, dtype=torch.long), "2": torch.ones(10, dtype=torch.long)}
with torch.no_grad():
    reference = model(x)
    for label, maps in (("all low", None), ("a quarter high", quarter), ("all high", everyone)):
        if maps is not None:
            set_palette_maps_(qmodel, maps)
        error = (qmodel(x) - reference).abs().mean()
        print(f"{label:>14}: mean difference to float32 {error:.5f}")

set_palette_maps_(qmodel, quarter)
# the forward pass's MACs by entry, a low product priced at half a high one
print(profile(qmodel, x, cost={low: 0.5, high: 1.0}).summary())
saved = palette_maps(qmodel)  # torch.save it beside the checkpoint
print({name: m.tolist()[:10] for name, m in saved.items()})
print("state_dict keys unchanged:", list(qmodel.state_dict()) == list(model.state_dict()))
