"""What simulating a pretrained GPT-2 costs: its speed, its memory, where the
time goes, and compiling it.

GPT-2 small (Hugging Face's ``GPT2LMHeadModel``, ``openai-community/gpt2``,
124M parameters, downloaded once) runs its forward pass on windows of
``--length`` tokens of random ids, without gradients, in two variants:

    float32   the model as it is
    fp8       under simulate: presets.fp8() on every projection but the tied
              lm_head (E4M3 operands scaled per tensor, the sums in the
              carrier), the attention's two products on E4M3 operands and
              its softmax in Binary14p8, bfloat16's precision

For each it reports the tokens per second (the median of ``--rounds``
windows, after one to warm up) and the peak memory (on CUDA the allocator's
for each variant, on the CPU the process's peak resident set so far). For
``fp8`` it then breaks one window's time down into mptorch's GEMM operators,
its quantizers, torch's own operators and the rest (Python, and on CUDA the
GPU's wait for launches), as ``torch.profiler`` attributes it, and with
``--compile`` reports what ``torch.compile(sim.wrap())`` costs: the first
call, the graphs, and the compiled windows' tokens per second.

Run from the repository root, with the ``hf`` group (check the CPU clock
first: a capped clock scales every number):

    uv run --group hf python dev/benchmarks/timing/gpt2_simulated.py --device cuda --compile
    uv run --group hf python dev/benchmarks/timing/gpt2_simulated.py --device cpu --threads 16
    uv run --group hf python dev/benchmarks/timing/gpt2_simulated.py --device cpu --threads 1 \
        --rounds 2

About two minutes on CUDA with ``--compile``, three on sixteen CPU threads
and ten on one.

Results (Ryzen 9 7940HS at 4.6-4.8 GHz, RTX 4060 laptop GPU, torch 2.15,
transformers 5.24.0, 2026-11-24), tokens per second, and where one ``fp8``
window's time goes:

                         float32   fp8      fp8's time   GEMMs  quantize  torch  rest
    cuda, 1024 tokens    22,067    2,985    7.4x         70%    6%        16%    8%
    cpu x16, 1024        1,220     60.3     20x          88%    2%        6%     5%
    cpu x1, 256          333       8.9      37x          97%    1%        2%     0%

Compiled on CUDA, ``torch.compile(sim.wrap())`` is one graph: the first call
takes 30 s, and the compiled windows run at 3,435 tokens/s, 1.15x eager's.
CUDA's peak is 756 MiB under float32 and 762 MiB under ``fp8``: the
simulation adds a few MiB of temporaries to the weights and a window's
activations. All of WikiText-2's test split (about 287k tokens, each
computed twice in windows of 1,024 moved by 512) would take about three
minutes on CUDA in ``fp8`` and two and a half hours on sixteen CPU threads.
"""

import argparse
import resource
import statistics
import time

import torch
from torch.profiler import ProfilerActivity

from mptorch import E4M3, BinaryK
from mptorch.quant import (
    AttentionFormats,
    FusedMac,
    GemmFormats,
    Quant,
    SoftmaxFormats,
    presets,
    simulate,
)

MODEL = "openai-community/gpt2"
BF = BinaryK(14, 8)


def recipe():
    e4m3 = GemmFormats(input_quant=Quant(E4M3), weight_quant=Quant(E4M3), fwd_mac=FusedMac(None))
    softmax = SoftmaxFormats.uniform(FusedMac(BF), Quant(BF))
    attention = AttentionFormats(qk=e4m3, softmax=softmax, pv=e4m3)
    fp8 = presets.fp8().override("lm_head", linear=None)
    return fp8.override("transformer.h.*.attn", attention=attention)


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def rate(forward, ids, device, rounds) -> float:
    """Tokens per second: the median over ``rounds`` windows, after one more."""
    times = []
    with torch.no_grad():
        for _ in range(rounds + 1):
            sync(device)
            start = time.perf_counter()
            forward(ids)
            sync(device)
            times.append(time.perf_counter() - start)
    return ids.numel() / statistics.median(times[1:])


def peak_memory(device: str) -> str:
    if device == "cuda":
        return f"{torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB allocated"
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10:,.0f} MiB resident"


def category(name: str) -> str:
    if name.startswith(("mptorch::custom_matmul", "mptorch::custom_conv")):
        return "mptorch GEMM"
    if name.startswith("mptorch::"):
        return "mptorch quantize"
    if name.startswith("aten::"):
        return "torch"
    return ""


def breakdown(forward, ids, device) -> None:
    """One window's time by kind of operator, as the profiler attributes it:
    each operator's own time on the CPU, and on CUDA the time of the kernels
    it launched itself. The rest is everything else: Python, and on CUDA the
    time the GPU waits for the next launch."""
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    with torch.no_grad(), torch.profiler.profile(activities=activities) as prof:
        sync(device)
        start = time.perf_counter()
        forward(ids)
        sync(device)
        wall = (time.perf_counter() - start) * 1e6
    totals: dict[str, float] = {}
    for e in prof.key_averages():
        kind = category(e.key)
        if kind:
            t = e.self_device_time_total if device == "cuda" else e.self_cpu_time_total
            totals[kind] = totals.get(kind, 0.0) + t
    print(f"  one window: {wall / 1e3:,.1f} ms")
    for kind in ("mptorch GEMM", "mptorch quantize", "torch"):
        t = totals.get(kind, 0.0)
        print(f"    {kind:<17} {t / 1e3:9.1f} ms  {t / wall:6.1%}")
    rest = max(wall - sum(totals.values()), 0.0)
    print(f"    {'the rest':<17} {rest / 1e3:9.1f} ms  {rest / wall:6.1%}")


def main() -> None:
    from transformers import GPT2LMHeadModel

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = args.device
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(device).eval()
    ids = torch.randint(0, model.config.vocab_size, (1, args.length), device=device)
    threads = f", {torch.get_num_threads()} threads" if device == "cpu" else ""
    print(f"GPT-2 small, {args.length} tokens a window, on {device}{threads}")

    plain = rate(lambda x: model(x), ids, device, args.rounds)
    print(f"  float32: {plain:9.1f} tokens/s, {peak_memory(device)}")
    sim = simulate(recipe(), model)

    def simulated(x):
        with sim:
            return model(x)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fp8 = rate(simulated, ids, device, args.rounds)
    print(
        f"  fp8:     {fp8:9.1f} tokens/s, {peak_memory(device)}, {plain / fp8:.1f}x float32's time"
    )
    breakdown(simulated, ids, device)

    if args.compile:
        import torch._dynamo as dynamo

        dynamo.reset()
        compiled = torch.compile(sim.wrap())
        with torch.no_grad():
            sync(device)
            start = time.perf_counter()
            compiled(ids)
            sync(device)
            first = time.perf_counter() - start
        graphs = dynamo.utils.counters["stats"]["unique_graphs"]
        fast = rate(compiled, ids, device, args.rounds)
        print(
            f"  compiled: first call {first:.1f} s, {graphs} graph(s); {fast:9.1f} tokens/s, "
            f"{fp8 and fast / fp8:.2f}x eager's speed"
        )


if __name__ == "__main__":
    main()
