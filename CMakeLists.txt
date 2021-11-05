cmake_minimum_required(VERSION 3.16)
project(mptorch)

set(CMAKE_CXX_STANDARD 14)


include_directories(mptorch/quant/quant_cpu)
include_directories(mptorch/quant/quant_cpu)
include_directories(mptorch/quant/quant_cuda)

add_executable(mptorch
        mptorch/quant/quant_cpu/bit_helper.cpp
        mptorch/quant/quant_cpu/quant_cpu.cpp
        mptorch/quant/quant_cpu/quant_cpu.h
        mptorch/quant/quant_cpu/sim_helper.cpp
        mptorch/quant/quant_cuda/bit_helper.cu
        mptorch/quant/quant_cuda/block_kernel.cu
        mptorch/quant/quant_cuda/fixed_point_kernel.cu
        mptorch/quant/quant_cuda/float_kernel.cu
        mptorch/quant/quant_cuda/quant.cu
        mptorch/quant/quant_cuda/quant_cuda.cpp
        mptorch/quant/quant_cuda/quant_cuda.h
        mptorch/quant/quant_cuda/quant_kernel.h
        mptorch/quant/quant_cuda/sim_helper.cu)
