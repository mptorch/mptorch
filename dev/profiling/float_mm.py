from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import float_mm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, default=1000)
parser.add_argument("-k", type=int, default=500)
parser.add_argument("-n", type=int, default=2500)
parser.add_argument("--man-add", type=int, default=23)
parser.add_argument("--exp-add", type=int, default=8)
parser.add_argument("--man-mul", type=int, default=23)
parser.add_argument("--exp-mul", type=int, default=8)
parser.add_argument("--rounding", type=str, default="nearest")
parser.add_argument("--fma", action="store_true")
args = parser.parse_args()


m, k, n = args.m, args.k, args.n
a = torch.rand(m, k).cuda()
b = torch.rand(k, n).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("float_mm"):
        float_mm(
            a,
            b,
            man_add=args.man_add,
            exp_add=args.exp_add,
            man_mul=args.man_mul,
            exp_mul=args.exp_mul,
            rounding=args.rounding,
            fma=args.fma,
            subnormals=True,
            saturate=True
        )

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))