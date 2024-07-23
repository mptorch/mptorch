from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import binary8_quantize
import argparse


# Select matrix sizes and float format (command line arguments)
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, default=1000)
parser.add_argument("-n", type=int, default=1000)
parser.add_argument("-p", type=int, default=3)
parser.add_argument("--rounding", type=str, default="nearest", choices=["nearest", "stochastic", "truncate"])
parser.add_argument("--saturation", type=str, choices=["saturate", "overlow", "no_overflow"], default="saturate")
parser.add_argument("--subnormals", action="store_true", default=False)
parser.add_argument("--unsigned", action="store_true", default=False)
args = parser.parse_args()


if torch.cuda.is_available():
    x = torch.rand(args.m, args.n, device="cuda")*10

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("binary8_quantize"):
            binary8_quantize(
                x,
                P=args.p,
                rounding=args.rounding,
                saturation_mode=args.saturation,
                is_signed=not args.unsigned,
                subnormals=args.subnormals,
                prng_bits=23
            )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
else:
    raise NotImplementedError("No CUDA-capable device found. Stopping script.")