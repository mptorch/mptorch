from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import float_mm, binary8_quantize, cublas_acceleration
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

    print("Benchmarking binary8_quantize followed by float_mm without cublas acceleration.")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    a = torch.rand(m, k).cuda()
    b = torch.rand(k, n).cuda()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("quant_binary8_and_float_mm"):
            binary8_quantize(a, 1, "stochastic", "no_overflow", True, True, 0)
            binary8_quantize(b, 1, "stochastic", "no_overflow", True, True, 0)
            float_mm(
                a,
                b,
                man_add=args.man_add,
                exp_add=args.exp_add,
                man_mul=args.man_mul,
                exp_mul=args.exp_mul,
                saturate=False
            )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("Benchmarking binary8_quantize followed by float_mm with cublas acceleration on.")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    a = torch.rand(m, k).cuda()
    b = torch.rand(k, n).cuda()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("quant_binary8_and_float_mm_with_cublas"):
            # enable cublas acceleration when type matches
            with cublas_acceleration(True):
                binary8_quantize(a, 1, "stochastic", "no_overflow", True, True, 0)
                binary8_quantize(b, 1, "stochastic", "no_overflow", True, True, 0)
                float_mm(
                    a,
                    b,
                    man_add=args.man_add,
                    exp_add=args.exp_add,
                    man_mul=args.man_mul,
                    exp_mul=args.exp_mul,
                    saturate=False
                )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
else:
    raise NotImplementedError("No CUDA-capable device found. Stopping script.")

