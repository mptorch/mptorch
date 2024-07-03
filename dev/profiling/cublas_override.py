from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import float_mm, cublas_acceleration
import argparse


# Select matrix sizes and float format (command line arguments)
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, default=10000)
parser.add_argument("-k", type=int, default=5000)
parser.add_argument("-n", type=int, default=25000)
parser.add_argument("--man-add", type=int, default=10)
parser.add_argument("--exp-add", type=int, default=5)
parser.add_argument("--man-mul", type=int, default=10)
parser.add_argument("--exp-mul", type=int, default=5)
args = parser.parse_args()

# generate matrices with random data and offload 
# them to GPU; if no CUDA-enabled GPU is available,
# halt the script
if torch.cuda.is_available():
    m, k, n = args.m, args.k, args.n
    a = torch.rand(m, k).cuda()
    b = torch.rand(k, n).cuda()

    print("Benchmarking float_mm without cublas acceleration on.")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("float_mm"):
            float_mm(
                a,
                b,
                man_add=args.man_add,
                exp_add=args.exp_add,
                man_mul=args.man_mul,
                exp_mul=args.exp_mul,
                saturate=False
            )
    print("Self CPU time total:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("Self CUDA time total:", prof.key_averages().total_average().self_cuda_time_total_str)

    print("Benchmarking float_mm with cublas acceleration on.")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("float_mm_with_cublas"):
            # enable cublas acceleration when type matches
            with cublas_acceleration(True):
                float_mm(
                    a,
                    b,
                    man_add=args.man_add,
                    exp_add=args.exp_add,
                    man_mul=args.man_mul,
                    exp_mul=args.exp_mul,
                    saturate=False
                )
    print("Self CPU time total:", prof.key_averages().total_average().self_cpu_time_total_str)
    print("Self CUDA time total:", prof.key_averages().total_average().self_cuda_time_total_str)
else:
    raise NotImplementedError("No CUDA-capable device found. Stopping script.")