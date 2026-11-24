"""WikiText-2 perplexity of a pretrained GPT-2 in mptorch's arithmetic.

Hugging Face's own ``GPT2LMHeadModel`` (``openai-community/gpt2``, 124M
parameters) runs as it is, under ``simulate``: its ``Conv1D`` projections
are ``torch.addmm`` calls over their weights, which the ``linear`` kind
reaches, and its attention is ``F.scaled_dot_product_attention``, which the
``attention`` kind reaches. Four recipes:

* ``float32``: the model as it is;
* ``fp8``: ``presets.fp8()`` on every projection but the tied ``lm_head``
  (E4M3 operands scaled per tensor, the sums in the carrier);
* ``fp8+attention``: the attention's two products in E4M3 too, and its
  softmax in a format of bfloat16's precision;
* ``fp8+blocks``: the blocks' own arithmetic (the residual sums, the GELU's
  steps) and their norms in that format as well.

The text is WikiText-2's raw test split (downloaded once by ``datasets``),
scored as transformers' perplexity guide does: windows of 1,024 tokens
moved by 512, each scoring its last 512. ``--tokens`` keeps the text's
first tokens.

Before the recipes run, the SDPA model is held to its eager twin
(``attn_implementation="eager"``) under ``Recipe.attention_parts``: the
largest difference of their logits must be 0. The coverage report of the
last recipe lists what was not simulated.

    uv run --group hf python examples/instrument/gpt2_perplexity.py --device cuda
"""

import argparse
import math
import time

import torch

from mptorch import E4M3, BinaryK
from mptorch.quant import (
    ActivationFormats,
    AttentionFormats,
    FusedMac,
    GemmFormats,
    NormFormats,
    PointwiseFormats,
    Quant,
    SoftmaxFormats,
    presets,
    profile,
    simulate,
)

MODEL = "openai-community/gpt2"
BF = BinaryK(14, 8)  # bfloat16's 8 bits of precision, in a range binary32 holds
ATTENTION = "transformer.h.*.attn"


def attention_formats() -> AttentionFormats:
    """The attention's two products on E4M3 operands, its softmax in ``BF``."""
    e4m3 = GemmFormats(input_quant=Quant(E4M3), weight_quant=Quant(E4M3), fwd_mac=FusedMac(None))
    softmax = SoftmaxFormats.uniform(FusedMac(BF), Quant(BF))
    return AttentionFormats(qk=e4m3, softmax=softmax, pv=e4m3)


def recipes() -> dict:
    fp8 = presets.fp8().override("lm_head", linear=None)
    with_attention = fp8.override(ATTENTION, attention=attention_formats())
    blocks = with_attention.override(
        "transformer.h.*",
        elementwise=PointwiseFormats(quant=Quant(BF)),
        activation=ActivationFormats(out=Quant(BF)),
        norm=NormFormats.uniform(FusedMac(BF), Quant(BF)),
    )
    return {
        "float32": None,
        "fp8": fp8,
        "fp8+attention": with_attention,
        "fp8+blocks": blocks,
    }


def windows(ids: torch.Tensor, length: int = 1024, stride: int = 512):
    """transformers' strided perplexity: each window's input and its labels,
    the tokens an earlier window scored masked with -100."""
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


def perplexity(model, ids, device, sim=None) -> tuple[float, float]:
    """The perplexity of ``ids`` and the tokens computed per second."""
    nll, count, computed = 0.0, 0, 0
    start = time.perf_counter()
    with torch.no_grad():
        for window, labels in windows(ids):
            window, labels = window.to(device), labels.to(device)
            if sim is None:
                loss = model(window, labels=labels).loss
            else:
                with sim:
                    loss = model(window, labels=labels).loss
            # the loss is a mean over the predicted tokens, the labels shifted by one
            n = (labels[:, 1:] != -100).sum().item()
            nll += loss.item() * n
            count += n
            computed += window.shape[1]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return math.exp(nll / count), computed / (time.perf_counter() - start)


def twin_check(device: str, ids: torch.Tensor) -> float:
    """The largest difference of the logits of the SDPA model and its eager
    twin, under one attention arithmetic given to both spellings."""
    from transformers import GPT2LMHeadModel

    attention = attention_formats()
    base = presets.fp8().override("lm_head", linear=None)
    out = {}
    for attn in ("sdpa", "eager"):
        model = GPT2LMHeadModel.from_pretrained(MODEL, attn_implementation=attn).to(device).eval()
        recipe = (
            base.override(ATTENTION, attention=attention)
            if attn == "sdpa"
            else base.attention_parts(ATTENTION, attention)
        )
        with torch.no_grad(), simulate(recipe, model):
            out[attn] = model(ids.to(device)).logits
    return (out["sdpa"] - out["eager"]).abs().max().item()


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPT2LMHeadModel

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=32768, help="the text's first tokens")
    parser.add_argument("--recipes", nargs="+", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    assert tokenizer is not None
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, : args.tokens]
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(args.device).eval()
    print(f"GPT-2 small, WikiText-2 test, {ids.shape[1]:,} tokens, on {args.device}")
    print(f"SDPA against eager, the largest difference: {twin_check(args.device, ids[:, :256])}")

    sim = None
    for name, recipe in recipes().items():
        if args.recipes and name not in args.recipes:
            continue
        sim = None if recipe is None else simulate(recipe, model)
        ppl, rate = perplexity(model, ids, args.device, sim)
        line = f"{name:>14}  perplexity {ppl:8.3f}  {rate:8.1f} tokens/s"
        if recipe is not None:
            window = ids[:, :1024].to(args.device)
            counted = profile(model, window, recipe=recipe, run=False)
            share = 1 - counted.by_format().get(None, 0) / counted.macs
            line += f"  {share:6.1%} of the MACs in the recipe's arithmetic"
        print(line, flush=True)
    if sim is not None:
        uncovered = sim.report().not_covered
        print(f"not covered under the last recipe: {len(uncovered)} kinds of call")
        for row in uncovered:
            print(f"  {row.scope}: {row.call} ({row.action})")


if __name__ == "__main__":
    main()
