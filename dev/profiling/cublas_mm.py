from torch.profiler import profile, record_function, ProfilerActivity
import torch
from mptorch.quant import cublas_mm, CUBLASComputeType, CUBLASMatrixType
import argparse


# Select matrix sizes and float format (command line arguments)
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, default=10000)
parser.add_argument("-k", type=int, default=5000)
parser.add_argument("-n", type=int, default=2500)
args = parser.parse_args()


m, k, n = args.m, args.k, args.n
a = torch.rand(m, k).cuda()
b = torch.rand(k, n).cuda()


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