from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import float_bmm
import argparse


# Select matrix sizes and float format (command line arguments)
parser = argparse.ArgumentParser()
parser.add_argument("-b", type=int, default=10)
parser.add_argument("-m", type=int, default=10000)
parser.add_argument("-k", type=int, default=5000)
parser.add_argument("-n", type=int, default=2500)
parser.add_argument("--man-add", type=int, default=10)
parser.add_argument("--exp-add", type=int, default=5)
parser.add_argument("--man-mul", type=int, default=10)
parser.add_argument("--exp-mul", type=int, default=5)
parser.add_argument("--rounding", type=str, default="nearest")
parser.add_argument("--fma", action="store_true")
args = parser.parse_args()


b, m, k, n = args.b, args.m, args.k, args.n
a = torch.rand(b, m, k).cuda()
b = torch.rand(b, k, n).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("float_batched_matrix_multiply"):
        float_bmm(
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