"""Training an MLP on MNIST in FP8: the signals, the matmuls, and the weights.

Every configuration below trains the same 784-128-96-10 network from the same
initialisation, and differs only in what is rounded, to what, and where:

    signals   what the layer's inputs, weights, biases and gradients are
              rounded to before each matmul (the *_quant slots)
    matmuls   the arithmetic *inside* the three GEMMs of each layer
              (the *_math hooks, via binaryK_gemm_formats)
    weights   the format the master copy of the weights is rounded to after
              every optimizer step, and how

Run:  python3 fp8_05_mlp_mnist.py [--epochs N] [--only name,name]
"""

import argparse
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets

from mptorch import BinaryK, FormatRangeWarning, RoundMode
from mptorch.quant import QAffineFormats, QLinear, Quant, Quantizer, binaryK_gemm_formats

E4M3 = BinaryK(8, 4, bias=7)
E5M2 = BinaryK(8, 3, bias=15)
# An 8-exponent-bit binaryK reaches below 2**-126, where the casts cannot tell
# one input from another, so building one warns (concepts, "What float32 can
# carry"). Nothing here goes anywhere near that small.
warnings.simplefilter("ignore", FormatRangeWarning)

BF16 = BinaryK(16, 8)  # 8 exponent, 7 mantissa bits
FP22 = BinaryK(23, 15)  # 8 exponent, 14 mantissa bits: a reduced-precision accumulator
BF16_SR = BinaryK(16, 8, prng_bits=8)
E4M3_SR = BinaryK(8, 4, bias=7, prng_bits=8)


def fp8_signals(formats: QAffineFormats) -> QAffineFormats:
    """E4M3 on everything the forward pass reads, E5M2 on every gradient."""
    fwd, bwd = Quant(E4M3), Quant(E5M2)
    formats.input_quant, formats.weight_quant, formats.bias_quant = fwd, fwd, fwd
    formats.igrad_quant, formats.wgrad_quant, formats.bgrad_quant = bwd, bwd, bwd
    return formats


def gemm(acc: BinaryK | None, rounding: RoundMode = RoundMode.RNE) -> QAffineFormats:
    """Products exact (bf16 holds an FP8 x FP8 product), running sums in `acc`."""
    return binaryK_gemm_formats(
        mul_K=16,
        mul_P=8,
        accumulate_quant=acc is not None,
        acc_K=acc.K if acc else None,
        acc_P=acc.P if acc else None,
        acc_prng_bits=acc.prng_bits if acc else 0,
        rounding_mode=rounding,
    )


@dataclass
class Config:
    name: str
    formats: Callable[[], QAffineFormats]
    weight_quant: Callable[[torch.Tensor], torch.Tensor] | None = None  # master weights
    loss_scale: float = 1.0


CONFIGS = [
    Config("float32", QAffineFormats),
    Config("fp8 signals, fp32 matmuls", lambda: fp8_signals(QAffineFormats())),
    Config(
        "fp8 signals, fp32 matmuls, loss scale 1024",
        lambda: fp8_signals(QAffineFormats()),
        loss_scale=1024.0,
    ),
    Config("fp8 signals, fp22-accumulated matmuls", lambda: fp8_signals(gemm(FP22))),
    Config("fp8 signals, bf16-accumulated matmuls", lambda: fp8_signals(gemm(BF16))),
    Config(
        "fp8 signals, bf16-accumulated matmuls, SR",
        lambda: fp8_signals(gemm(BF16_SR, RoundMode.SR)),
    ),
    Config("fp8 signals, E4M3-accumulated matmuls", lambda: fp8_signals(gemm(E4M3))),
    Config(
        "fp8 signals, E4M3 master weights (RNE)",
        lambda: fp8_signals(QAffineFormats()),
        weight_quant=Quant(E4M3),
    ),
    Config(
        "fp8 signals, E4M3 master weights (SR)",
        lambda: fp8_signals(QAffineFormats()),
        weight_quant=Quant(E4M3_SR, RoundMode.SR),
    ),
]


def load_mnist(device: str):
    """The whole dataset as normalised [N, 784] tensors on the device."""
    out = []
    for train in (True, False):
        ds = datasets.MNIST("data", train=train, download=True)
        x = (ds.data.float() / 255 - 0.1307) / 0.3081
        out += [x.reshape(len(ds), -1).to(device), ds.targets.to(device)]
    return out


def make_model(cfg: Config, device: str) -> nn.Sequential:
    torch.manual_seed(0)  # same initial weights for every configuration
    return nn.Sequential(
        QLinear(784, 128, formats=cfg.formats()),
        nn.ReLU(),
        QLinear(128, 96, formats=cfg.formats()),
        nn.ReLU(),
        QLinear(96, 10, formats=cfg.formats()),
        # There is no output_quant slot: rounding the logits is a Quantizer,
        # here E4M3 forward with the gradient passed through in E5M2.
        Quantizer(E4M3, E5M2) if cfg.name != "float32" else nn.Identity(),
    ).to(device)


def accuracy(model, x, y, batch=1000) -> float:
    model.eval()
    with torch.no_grad():
        correct = sum(
            (model(x[i : i + batch]).argmax(1) == y[i : i + batch]).sum().item()
            for i in range(0, len(x), batch)
        )
    return correct / len(x)


def train(cfg: Config, data, device: str, epochs: int, batch: int = 64, lr: float = 0.1):
    x_tr, y_tr, x_te, y_te = data
    model = make_model(cfg, device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    if cfg.weight_quant:  # start from weights the master format can hold
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(cfg.weight_quant(p))
    g = torch.Generator(device=device).manual_seed(0)
    accs, t0 = [], time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=g, device=device)
        for i in range(0, len(x_tr), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            (F.cross_entropy(model(x_tr[idx]), y_tr[idx]) * cfg.loss_scale).backward()
            if cfg.loss_scale != 1.0:
                for p in model.parameters():
                    p.grad /= cfg.loss_scale
            opt.step()
            if cfg.weight_quant:  # w <- Q(w - lr * g): the master copy lives in this format
                with torch.no_grad():
                    for p in model.parameters():
                        p.copy_(cfg.weight_quant(p))
        accs.append(accuracy(model, x_te, y_te))
    return accs, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_mnist(device)
    configs = [
        c for c in CONFIGS if not args.only or any(o in c.name for o in args.only.split(","))
    ]

    print(f"MNIST, 784-128-96-10 MLP, SGD lr 0.1, batch 64, {args.epochs} epochs, on {device}")
    print(
        f"{'configuration':<48}"
        + "".join(f"{'ep ' + str(e + 1):>8}" for e in range(args.epochs))
        + f"{'time':>8}"
    )
    for cfg in configs:
        accs, dt = train(cfg, data, device, args.epochs)
        print(f"{cfg.name:<48}" + "".join(f"{a:>8.4f}" for a in accs) + f"{dt:>7.0f}s", flush=True)


if __name__ == "__main__":
    main()
