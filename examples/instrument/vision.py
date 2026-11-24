"""Top-1 accuracy of pretrained torchvision models in mptorch's arithmetic.

ResNet-18 is converted (``convert``) and ViT-B/16 simulated (``simulate``,
its ``nn.MultiheadAttention`` reached through the ``linear`` kind for the
projections and the ``attention`` kind for the core), each in four recipes:

* ``float32``: the model as it is;
* ``binary8``: every product's operands in Binary8p4 (P3109's 8-bit format
  with 4 bits of precision), each product rounded to it and summed in
  Binary16p11, the gradients alike;
* ``fp8``: ``presets.fp8()``, E4M3 operands scaled per tensor, the sums in
  the carrier, as an FP8 tensor core computes;
* ``mx``: ``presets.mx()``, the linear layers' operands in MXFP8 (E4M3
  elements sharing an E8M0 scale per 32), which leaves a CNN's convolutions
  alone.

For ViT-B/16 the attention's two products take each recipe's arithmetic
too (its softmax stays float32).

The images are Imagenette's validation set (3,925 images of 10 ImageNet
classes, downloaded once by torchvision into ``--data``), scored over all
1,000 classes, or a local ImageNet validation set (``--imagenet DIR``, an
``ImageFolder`` of the 1,000 class folders named by wnid). ``--limit``
takes a fixed random subset.

    uv run python examples/instrument/vision.py --device cuda
    uv run python examples/instrument/vision.py --device cuda --all
"""

import argparse
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder, Imagenette

from mptorch import E4M3, MXFP8_E4M3, BinaryK
from mptorch.quant import (
    AttentionFormats,
    BlockMac,
    FusedMac,
    GemmFormats,
    Quant,
    SplitMac,
    convert,
    presets,
    profile,
    simulate,
)

P8, P16 = BinaryK(8, 4), BinaryK(16, 11)

# Imagenette's classes, in the order its folders sort (by wnid), as indices
# among ImageNet-1k's: tench, English springer, cassette player, chain saw,
# church, French horn, garbage truck, gas pump, golf ball, parachute
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

ATTENTION = "encoder.layers.*.self_attention"


def recipes(model_name: str) -> dict:
    """The four recipes; ViT's attention core takes each one's arithmetic."""
    out = {
        "float32": None,
        "binary8": presets.uniform(P8, acc=P16),
        "fp8": presets.fp8(),
        "mx": presets.mx(),
    }
    if model_name == "vit_b_16":
        e4m3 = GemmFormats(
            input_quant=Quant(E4M3), weight_quant=Quant(E4M3), fwd_mac=FusedMac(None)
        )
        core = {
            "binary8": AttentionFormats(
                qk=GemmFormats.uniform(SplitMac(P8, P16), Quant(P8)),
                pv=GemmFormats.uniform(SplitMac(P8, P16), Quant(P8)),
            ),
            "fp8": AttentionFormats(qk=e4m3, pv=e4m3),
            "mx": AttentionFormats(
                qk=GemmFormats.uniform(BlockMac(MXFP8_E4M3)),
                pv=GemmFormats.uniform(BlockMac(MXFP8_E4M3)),
            ),
        }
        for name, af in core.items():
            out[name] = out[name].override(ATTENTION, attention=af)
    return out


def dataset(args, weights):
    """The images, their labels as ImageNet-1k indices, and a fixed subset."""
    transform = weights.transforms()
    if args.imagenet:
        ds = ImageFolder(args.imagenet, transform=transform)
        labels = None
    else:
        ds = Imagenette(args.data, split="val", size="320px", download=True, transform=transform)
        names = [weights.meta["categories"][i] for i in IMAGENETTE_TO_IMAGENET]
        if names != [c[0] for c in ds.classes]:
            raise RuntimeError(f"Imagenette's classes are not {names}: {ds.classes}")
        labels = torch.tensor(IMAGENETTE_TO_IMAGENET)
    if not args.all and args.limit < len(ds):
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))
        ds = Subset(ds, order[: args.limit].tolist())
    return ds, labels


def evaluate(forward, loader, labels, device) -> tuple[float, float]:
    """Top-1 accuracy and images per second."""
    correct = total = 0
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y if labels is None else labels[y]
            correct += (forward(x).argmax(-1).cpu() == y).sum().item()
            total += len(y)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return correct / total, total / (time.perf_counter() - start)


def run(model_name: str, args) -> None:
    weights = torchvision.models.get_model_weights(model_name)["DEFAULT"]
    model = torchvision.models.get_model(model_name, weights=weights).eval().to(args.device)
    ds, labels = dataset(args, weights)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
    x = next(iter(loader))[0][:1].to(args.device)
    for name, recipe in recipes(model_name).items():
        if args.recipes and name not in args.recipes:
            continue
        note = ""
        if recipe is None:
            forward = model
        elif model_name == "resnet18":  # by conversion
            converted, report = convert(model, recipe)
            forward = converted
            n = len(report.converted)
            note = f"{n} layer{'s' * (n != 1)} converted"
        else:  # by simulation
            sim = simulate(recipe, model)

            def forward(x, sim=sim):
                with sim:
                    return model(x)

        accuracy, rate = evaluate(forward, loader, labels, args.device)
        if recipe is not None and model_name != "resnet18":
            uncovered = sim.report().not_covered
            note = f"{len(uncovered)} calls not covered" if uncovered else "every product covered"
        line = f"{model_name:>9}  {name:>8}  top-1 {accuracy:6.2%}  {rate:7.1f} images/s"
        if recipe is not None:
            counted = profile(model, x, recipe=recipe, run=False)
            share = 1 - counted.by_format().get(None, 0) / counted.macs
            line += f"  {share:7.2%} of the MACs in the recipe's arithmetic, {note}"
        print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["resnet18", "vit_b_16"])
    parser.add_argument("--recipes", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=1000, help="images, a fixed random subset")
    parser.add_argument("--all", action="store_true", help="every validation image")
    parser.add_argument("--imagenet", default=None, help="a local ImageNet validation set")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data"))
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data = "ImageNet" if args.imagenet else "Imagenette"
    count = "all" if args.all else args.limit
    print(f"{data} validation, {count} images, on {args.device}")
    for model_name in args.models:
        run(model_name, args)


if __name__ == "__main__":
    main()
