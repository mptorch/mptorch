from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import cublas_mm, CUBLASComputeType, CUBLASMatrixType, binary8_quantize
import argparse


# Select matrix sizes and float format (command line arguments)
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, default=10000)
parser.add_argument("-k", type=int, default=5000)
parser.add_argument("-n", type=int, default=2500)
args = parser.parse_args()

# generate matrices with random data and offload 
# them to GPU; if no CUDA-enabled GPU is available,
# halt the script


# convert here
if torch.cuda.is_available(): 
    m, k, n = args.m, args.k, args.n
    a = torch.rand(m, k).cuda()
    b = torch.rand(k, n).cuda()
    binary8_quantize(a, 1, "stochastic", "no_overflow", True, True, 0)
    binary8_quantize(b, 1, "stochastic", "no_overflow", True, True, 0)


    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("cublas_matrix_multiply"):
            cublas_mm(
                a,
                b,
                CUBLASMatrixType.F16,
                CUBLASMatrixType.F16,
                CUBLASComputeType.F16,
                False
            )

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
else:
    raise NotImplementedError("No CUDA-capable device found. Stopping script.")

