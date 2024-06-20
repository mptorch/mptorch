from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import float_mm
import argparse


# Select matrix sizes and float format (command line arguments)
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, default=10000)
parser.add_argument("-k", type=int, default=5000)
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

# Example adapted from: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
# Setup a profiler recording the following activites:
# - ProfilerActivity.CPU: PyTorch operators and user-defined labels
# - ProfilerActivity.CUDA: CUDA kernels
# - record_shapes: keep track of the shape of operator input tensors
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    # Record a specific block (in addition to PyTorch functions)
    # it will appear with the "float_matrix_multiply" label 
    # in the summary
    with record_function("float_matrix_multiply"):
        # call the function(s) to profile under that label
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

# Print a summary table, showing average times ordered by GPU execution time
# Limit the result to the 10 most expensive subfunctions.
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))