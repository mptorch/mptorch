# Makefile for building the dev/cuda kernels
# Each kernel should also have its individual compile and run
# commands in the header comments section.

# Find nvcc (NVIDIA CUDA compiler)
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
	$(error nvcc not found.)
endif

# Compiler flags
CFLAGS = -O3 --use_fast_math
NVCCFLAGS = -lcublas -lcublasLt -std=c++17 --extended-lambda -Wno-deprecated-gpu-targets

# Default rule for the CUDA files
%: %.cu 
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $< -o $@

# Build all targets (new executables go here)
TARGETS = vec_add round_bitwise_nearest fp_nearest cublas_gemm cublas_bgemm cublaslt_matmul custom_matmul custom_matmul_fma compensated_matmul superfp_nearest binary8_nearest binary8_stochastic nv_bfloat16 softmax_forward softmax_backward layernorm_forward layernorm_backward

all: $(TARGETS)

# Individual targets
vec_add: vec_add.cu 
round_bitwise_nearest: round_bitwise_nearest.cu
fp_nearest: fp_nearest.cu
cublas_gemm: cublas_gemm.cu
cublas_bgemm: cublas_bgemm.cu
cublaslt_matmul: cublaslt_matmul.cu
custom_matmul: custom_matmul.cu
custom_matmul_fma: custom_matmul_fma.cu
compensated_matmul: compensated_matmul.cu
binary8_nearest: binary8_nearest.cu
binary8_stochastic: binary8_stochastic.cu
nv_bfloat16: nv_bfloat16.cu
superfp_nearest: superfp_nearest.cu
layernorm_forward: layernorm_forward.cu
layernorm_backward: layernorm_backward.cu
softmax_forward: softmax_forward.cu
softmax_backward: softmax_backward.cu

# Run all targets with the default (first) kernel
run_all: all 
	@for target in $(TARGETS); do \
		echo "\n=================================="; \
		echo "Running $$target ..."; \
		echo "==================================\n"; \
		./$$target; \
	done

# Clean up
clean:
	rm -f $(TARGETS)