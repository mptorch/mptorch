"""Does a Hadamard rotation before block quantization help the block formats?

A block format chooses one scale per block (32 elements for MX, 16 for
NVFP4), so one large element in a block costs every other element of it most
of its precision. A rotation spreads a large element across its block before
the scale is chosen. This script measures whether that helps the shipped
formats, ``MXFP8_E4M3``, ``MXFP4_E2M1`` and ``NVFP4``.

Both operands of a product are rotated along the reduction axis K, in blocks
of ``r`` (a power of two dividing K): ``x -> x R`` and ``W -> W R``, with ``R``
block-diagonal, so ``(x R)(W R)^T = x W^T`` in exact arithmetic and a
pretrained layer keeps its function. Two rotations are compared, the
orthonormal Walsh-Hadamard transform ``H_r`` and ``D H_r``, with ``D`` a seeded
diagonal of +1 and -1 over the K channels (QuaRot's randomized Hadamard),
each at ``r`` of 1, 2 and 4 times the format's block size. A layer whose K
``r`` does not divide is left unrotated. The rotation runs in float64 and is
stored in float32, as a ``GemmFormats``' ``input_quant`` and
``weight_quant``, ahead of the ``BlockMac``'s packing.

Four stages:

    check       the rotation's own error: H H^T = I, and (xR)(WR)^T = x W^T
                without quantization
    tensors     the relative error of block_quantize (in the original
                basis) and of the block GEMM's result, on N(0, 1), on
                N(0, 1) with four channels scaled by 100, on GPT-2 small's
                activations at its Conv1D inputs (with each layer kind's
                outlier ratio, its largest channel RMS over the median), and
                on the tutorial MLP's
    perplexity  GPT-2 small (Hugging Face's own model, through simulate) on
                WikiText-2's first 32,768 tokens under each format, with and
                without each rotation; the tied lm_head stays float32
    mlp         the tutorial's MNIST MLP, trained in float32, then its test
                accuracy under each format, with and without each rotation

Run from the repository root, with the hf group (GPT-2 and WikiText-2 are
downloaded once):

    uv run --group hf python dev/experiments/hadamard_rotation.py check
    uv run --group hf python dev/experiments/hadamard_rotation.py tensors --device cuda
    uv run --group hf python dev/experiments/hadamard_rotation.py perplexity --device cuda
    uv run --group hf python dev/experiments/hadamard_rotation.py mlp --device cuda

Results (2026-11-25, an RTX 4060 laptop GPU, torch 2.15, transformers
5.24.0). Every run is deterministic: each repeat gave the same digits.

check: a round trip x -> R^T (x R) is exact to 2.4e-8 and (xR)(WR)^T equals
x W^T to 3.6e-8 (float32's storage), at r = 16 to 256.

perplexity, GPT-2 small on WikiText-2's first 32,768 tokens, float32 26.759
(each run 17-33 s). The increase over float32, and the share of it the
rotation removes; DH32 and DH16 over the seeds 0, 1 and 2:

    format   none     H, 1x            DH, 1x (3 seeds)        H, 4x            DH, 4x
    MXFP8   +1.071   +0.505  (53%)    +0.216/+0.401/+0.419    +0.250  (77%)    +0.220  (79%)
                                       (80%/63%/61%)
    MXFP4  +96.469  +23.124  (76%)   +15.880/+14.187/+14.903 +14.624  (85%)   +13.669  (86%)
                                       (84%/85%/85%)
    NVFP4   +6.393  +10.947 (-71%)   +8.143/+8.860/+9.207     +6.845  (-7%)    +7.012 (-10%)
                                       (-27%/-39%/-44%)

So a rotation removes most of the MX formats' loss and adds to NVFP4's at
every size. tensors says why: a rotation makes each block close to Gaussian.
On GPT-2's activations, MXFP4's quantization error (0.13-0.15) is above its
error on N(0, 1) (0.115), and rotation brings it down to that level
(0.112-0.117). NVFP4's (0.090-0.093) is below its error on N(0, 1) (0.095),
because its E4M3 scale already fits a block's largest element, and rotation
brings it up to that level (0.096-0.104). The block GEMM's error follows: on
MXFP4, GPT-2's c_attn 0.169 -> 0.127 and mlp.c_proj 0.266 -> 0.192 at best; on
NVFP4, 0.098 -> 0.107-0.110 and 0.153 -> 0.162-0.166. On N(0, 1) a rotation
changes nothing (a rotated i.i.d. Gaussian is one), and the four channels
scaled by 100 behave as GPT-2's do. GPT-2's outlier ratios are 9-17 by
layer kind; the MLP's 2.2-3.5.

perplexity --library (2026-11-25, the same GPU): the same runs through
mptorch's own rotation, GemmFormats.rotation, whose signs come from an
integer hash rather than this script's CPU generator. H, the same transform
in float32 butterfly stages rather than a float64 matmul, is within 0.06 of
the runs above (MXFP4 H128 41.441, NVFP4 H64 33.658), and to the digit with
the rotation in binary64 (Hadamard(..., carrier=torch.float64): 41.383,
33.604, 49.883). That gap is the rotation's float32 rounding, about 1e-7,
amplified by a 4-bit model. DH, its signs another draw, is within the seed
spread above: DH32 on MXFP8 +0.183/+0.483/+0.185, on MXFP4
+14.310/+14.910/+15.063, NVFP4's DH16 +9.296/+9.128/+8.188; DH128 on MXFP4
+12.426 and on MXFP8 +0.252, NVFP4's DH64 +8.484.

mlp, the MNIST MLP after 3 epochs, float32 97.49% (the tutorial's epoch 3):
MXFP8 97.52%, MXFP4 96.98% and NVFP4 97.15% unrotated, and a rotation moves
them by -0.09 to +0.23 points, against a binomial standard deviation of 0.16
points on 10,000 images. NVFP4 rises under all six rotations (+0.04 to
+0.22), the only consistent sign, and small. The MLP's activations have
no outliers to spread, and its first layer's K = 784 is a multiple of 16 only,
so only NVFP4's H16 and DH16 rotate that layer.
"""

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mptorch import MXFP4_E2M1, MXFP8_E4M3, NVFP4, BlockFormat
from mptorch.quant import (
    BlockMac,
    GemmFormats,
    Hadamard,
    Recipe,
    block_matmul,
    block_pack,
    block_quantize,
    convert,
    simulate,
)

FORMATS = {"MXFP8": MXFP8_E4M3, "MXFP4": MXFP4_E2M1, "NVFP4": NVFP4}
MULTIPLES = (1, 2, 4)


# --- the rotations -------------------------------------------------------------------------


def hadamard(r: int, device) -> torch.Tensor:
    """The orthonormal Sylvester Walsh-Hadamard matrix of order ``r``, in float64."""
    if r < 1 or r & (r - 1):
        raise ValueError(f"a Walsh-Hadamard matrix has a power-of-two order, got {r}")
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < r:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(r)


@dataclass(frozen=True)
class Rotation:
    """``x -> (x D) blockdiag(H_r)`` along the last axis, ``D`` the identity
    or, ``signed``, a seeded diagonal of +1 and -1 over the K channels."""

    r: int
    signed: bool = False
    seed: int = 0

    def label(self) -> str:
        return f"{'DH' if self.signed else 'H'}{self.r}"

    def _signs(self, k: int, device) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + k)
        return (torch.randint(0, 2, (k,), generator=g) * 2 - 1).to(device, torch.float64)

    def applies(self, k: int) -> bool:
        return k % self.r == 0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        k = t.shape[-1]
        if not self.applies(k):
            return t
        x = t.to(torch.float64)
        if self.signed:
            x = x * self._signs(k, t.device)
        x = x.reshape(*t.shape[:-1], k // self.r, self.r) @ hadamard(self.r, t.device)
        return x.reshape(t.shape).to(t.dtype)

    def inverse(self, t: torch.Tensor) -> torch.Tensor:
        k = t.shape[-1]
        if not self.applies(k):
            return t
        x = t.to(torch.float64).reshape(*t.shape[:-1], k // self.r, self.r)
        x = (x @ hadamard(self.r, t.device)).reshape(t.shape)  # H is symmetric and orthonormal
        if self.signed:
            x = x * self._signs(k, t.device)
        return x.to(t.dtype)


class _Rotate:
    """A stateless quantizer slot that rotates its operand along K."""

    def __init__(self, rotation: Rotation):
        self.rotation = rotation
        self.__name__ = f"rotate {rotation.label()}"

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.rotation.forward(t)


def rotations(fmt: BlockFormat, multiples=MULTIPLES, seed: int = 0) -> list[Rotation | None]:
    """No rotation, then H and D H at each multiple of the format's block size."""
    out: list[Rotation | None] = [None]
    for m in multiples:
        for signed in (False, True):
            out.append(Rotation(fmt.block_size * m, signed, seed))
    return out


def _label(rotation: Rotation | None) -> str:
    return "none" if rotation is None else rotation.label()


def block_recipe(
    fmt: BlockFormat, rotation: Rotation | None, lm_head: bool = False, library: bool = False
) -> Recipe:
    """Every linear product a block GEMM in ``fmt``, its operands rotated
    along K first; the forward pass only. With ``library``, the rotation is
    mptorch's own (``GemmFormats.rotation``: a float32 butterfly, and signs
    from an integer hash) instead of this script's float64 matmul by H."""
    if library:
        rot = None if rotation is None else Hadamard(rotation.r, rotation.signed, rotation.seed)
        gf = GemmFormats(fwd_mac=BlockMac(fmt), rotation=rot)
    else:
        q = None if rotation is None else _Rotate(rotation)
        gf = GemmFormats(input_quant=q, weight_quant=q, fwd_mac=BlockMac(fmt))
    recipe = Recipe(linear=gf)
    return recipe.override("lm_head", linear=None) if lm_head else recipe


# --- errors on tensors -----------------------------------------------------------------------


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def errors(x, w, fmt, rotation) -> tuple[float, float]:
    """The relative errors of ``x``'s block quantization (back in the
    original basis) and of the block GEMM ``x w^T``, under ``rotation``."""
    xr = x if rotation is None else rotation.forward(x)
    wr = w if rotation is None else rotation.forward(w)
    q = block_quantize(xr, fmt)
    back = q if rotation is None else rotation.inverse(q)
    out = block_matmul(block_pack(xr, fmt), block_pack(wr.T.contiguous(), fmt, axis=0))
    return _relative(back, x), _relative(out, x.double() @ w.double().T)


def check(device) -> None:
    """The rotation's own error, with nothing quantized."""
    torch.manual_seed(0)
    for r in (16, 32, 64, 128, 256):
        h = hadamard(r, device)
        eye = torch.eye(r, dtype=torch.float64, device=device)
        assert torch.allclose(h @ h.T, eye, atol=1e-12), r
    x = torch.randn(64, 768, device=device)
    w = torch.randn(96, 768, device=device)
    worst_round_trip = worst_product = 0.0
    for r in (16, 32, 64, 128, 256):
        for signed in (False, True):
            rot = Rotation(r, signed)
            worst_round_trip = max(worst_round_trip, _relative(rot.inverse(rot.forward(x)), x))
            got = rot.forward(x).double() @ rot.forward(w).double().T
            worst_product = max(worst_product, _relative(got, x.double() @ w.double().T))
    print("H H^T = I to 1e-12 at r = 16..256")
    print(f"largest relative error of a round trip: {worst_round_trip:.2e}")
    print(f"largest relative error of (xR)(WR)^T against x W^T: {worst_product:.2e}")


def _gpt2_operands(device, tokens: int = 1024):
    """GPT-2 small's inputs and weights at each Conv1D, from one window of
    WikiText-2's test split, by layer kind."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from transformers.pytorch_utils import Conv1D

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    assert tokenizer is not None
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, :tokens]
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
    captured: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, Conv1D):
            kind = ".".join(name.split(".")[-2:])  # attn.c_attn, mlp.c_proj, ...

            def pre(module, args, kind=kind):
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                captured.append((kind, x, module.weight.detach().T.float()))  # W as [N, K]

            handles.append(module.register_forward_pre_hook(pre))
    with torch.no_grad():
        model(ids.to(device))
    for h in handles:
        h.remove()
    return captured


def _outlier_ratio(x: torch.Tensor) -> float:
    rms = x.double().pow(2).mean(0).sqrt()
    return float(rms.max() / rms.median())


def tensors(device) -> None:
    torch.manual_seed(0)
    K = 768
    gauss = torch.randn(1024, K, device=device)
    outliers = gauss.clone()
    outliers[:, [7, 100, 300, 700]] *= 100
    w = torch.randn(K, K, device=device) / math.sqrt(K)
    groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "N(0,1)": [(gauss, w)],
        "N(0,1), 4 channels x100": [(outliers, w)],
    }
    for kind, x, wt in _gpt2_operands(device):
        groups.setdefault(f"GPT-2 {kind}", []).append((x, wt))
    mlp, (x_te, _) = _train_mlp(device, epochs=3)
    for name, (x, wt) in _mlp_operands(mlp, x_te).items():
        groups[f"MLP {name}"] = [(x, wt)]

    print("outlier ratio (largest channel RMS over the median), mean over a kind's layers:")
    for name, pairs in groups.items():
        ratio = sum(_outlier_ratio(x) for x, _ in pairs) / len(pairs)
        print(f"  {name:<28} {ratio:8.1f}   K = {pairs[0][0].shape[-1]}")
    for fmt_name, fmt in FORMATS.items():
        rots = rotations(fmt)
        print(f"\n{fmt_name}: relative error of the input's quantization / of the GEMM result")
        print(f"  {'data':<28}" + "".join(f"{_label(r):>16}" for r in rots))
        for name, pairs in groups.items():
            cells = []
            for rot in rots:
                pairs_err = [errors(x, wt, fmt, rot) for x, wt in pairs]
                q = sum(e[0] for e in pairs_err) / len(pairs_err)
                g = sum(e[1] for e in pairs_err) / len(pairs_err)
                skipped = rot is not None and not rot.applies(pairs[0][0].shape[-1])
                cells.append(f"{'(K)' if skipped else ''}{q:.4f}/{g:.4f}".rjust(16))
            print(f"  {name:<28}" + "".join(cells))
    print("\n(K): the rotation does not divide K, so the operands are not rotated")


# --- GPT-2's perplexity --------------------------------------------------------------------


def _windows(ids: torch.Tensor, length: int = 1024, stride: int = 512):
    scored = 0
    for begin in range(0, ids.shape[1], stride):
        end = min(begin + length, ids.shape[1])
        window = ids[:, begin:end]
        labels = window.clone()
        labels[:, : scored - begin] = -100
        yield window, labels
        scored = end
        if end == ids.shape[1]:
            break


def _perplexity(model, ids, device, sim=None) -> float:
    nll, count = 0.0, 0
    with torch.no_grad():
        for window, labels in _windows(ids):
            window, labels = window.to(device), labels.to(device)
            if sim is None:
                loss = model(window, labels=labels).loss
            else:
                with sim:
                    loss = model(window, labels=labels).loss
            n = (labels[:, 1:] != -100).sum().item()
            nll += loss.item() * n
            count += n
    return math.exp(nll / count)


def perplexity(device, tokens: int, multiples, formats, seed: int, library: bool) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    assert tokenizer is not None
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, :tokens]
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device).eval()
    print(f"GPT-2 small, WikiText-2 test, {ids.shape[1]:,} tokens, on {device}")
    start = time.perf_counter()
    base = _perplexity(model, ids, device)
    print(f"  float32: {base:.3f}  ({time.perf_counter() - start:.0f} s)", flush=True)
    for fmt_name in formats:
        fmt = FORMATS[fmt_name]
        for rot in rotations(fmt, multiples, seed):
            start = time.perf_counter()
            sim = simulate(block_recipe(fmt, rot, lm_head=True, library=library), model)
            ppl = _perplexity(model, ids, device, sim)
            elapsed = time.perf_counter() - start
            print(
                f"  {fmt_name} {_label(rot):>6}: {ppl:8.3f}  (+{ppl - base:.3f}, {elapsed:.0f} s)",
                flush=True,
            )


# --- the tutorial MLP ------------------------------------------------------------------------


def _mnist(device):
    from torchvision import datasets

    out = []
    for train in (True, False):
        ds = datasets.MNIST("data", train=train, download=True)
        x = (ds.data.float() / 255 - 0.1307) / 0.3081
        out += [x.reshape(len(ds), -1).to(device), ds.targets.to(device)]
    return out


def _train_mlp(device, epochs: int, batch: int = 64, lr: float = 0.1):
    """The tutorial's 784-128-96-10 MLP, trained in float32 as it trains it,
    and the test set."""
    x_tr, y_tr, x_te, y_te = _mnist(device)
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 96), nn.ReLU(), nn.Linear(96, 10)
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(0)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=g, device=device)
        for i in range(0, len(x_tr), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            F.cross_entropy(model(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
    model.eval()
    return model, (x_te, y_te)


def _accuracy(model, x, y, batch: int = 1000) -> float:
    with torch.no_grad():
        correct = sum(
            (model(x[i : i + batch]).argmax(1) == y[i : i + batch]).sum().item()
            for i in range(0, len(x), batch)
        )
    return correct / len(x)


def _mlp_operands(model, x_te):
    """Each layer's input on 1,000 test images, and its weight."""
    inputs: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):

            def pre(module, args, name=name):
                inputs[name] = args[0].detach()

            handles.append(module.register_forward_pre_hook(pre))
    with torch.no_grad():
        model(x_te[:1000])
    for h in handles:
        h.remove()
    layers = dict(model.named_modules())
    return {
        f"layer {n} (K={inputs[n].shape[-1]})": (inputs[n], layers[n].weight.detach())
        for n in inputs
    }


def mlp(device, epochs: int) -> None:
    model, (x_te, y_te) = _train_mlp(device, epochs)
    print(f"MNIST MLP 784-128-96-10, {epochs} epochs of float32 SGD, on {device}")
    print(f"  float32: {_accuracy(model, x_te, y_te):.2%}")
    for fmt_name, fmt in FORMATS.items():
        for rot in rotations(fmt):
            converted, _ = convert(model, block_recipe(fmt, rot))
            rotated = (
                ""
                if rot is None
                else "  (rotated: layers "
                + ", ".join(
                    n
                    for n, m in model.named_modules()
                    if isinstance(m, nn.Linear) and rot.applies(m.in_features)
                )
                + ")"
            )
            print(f"  {fmt_name} {_label(rot):>6}: {_accuracy(converted, x_te, y_te):.2%}{rotated}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["check", "tensors", "perplexity", "mlp"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--multiples", default="1,2,4", help="rotation sizes, x the block size")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--formats", default=",".join(FORMATS), help="perplexity's formats")
    parser.add_argument("--seed", type=int, default=0, help="the seed of D H's signs")
    parser.add_argument(
        "--library", action="store_true", help="perplexity through GemmFormats.rotation"
    )
    args = parser.parse_args()
    multiples = tuple(int(m) for m in args.multiples.split(","))
    if args.stage == "check":
        check(args.device)
    elif args.stage == "tensors":
        tensors(args.device)
    elif args.stage == "perplexity":
        perplexity(
            args.device, args.tokens, multiples, args.formats.split(","), args.seed, args.library
        )
    else:
        mlp(args.device, args.epochs)


if __name__ == "__main__":
    main()
