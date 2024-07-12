#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cfloat>
#include <iostream>

// --------------------------------------------------------------------------------
// helper functions and formatted printing

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

#define MIN(a, b) ((a) < (b) ? a : b)
#define MAX(a, b) ((a) > (b) ? a : b)

#define NC "\e[0m"
#define RED "\e[0;31m"
#define GRN "\e[0;32m"
#define CYN "\e[0;36m"
#define REDB "\e[41m"

auto print_float = [](float x) {
    uint32_t u = FLOAT_TO_BITS(&x);
    std::cout << CYN << (u >> 31) << " ";
    for (int i{1}; i < 9; ++i)
        std::cout << GRN << ((u << i) >> 31);
    std::cout << " ";
    for (int i{9}; i < 32; ++i) 
        std::cout << RED << ((u << i) >> 31);
    std::cout << NC;
};

auto print_uint32 = [](uint32_t u) {
    std::cout << CYN << (u >> 31) << " ";
    for (int i{1}; i < 9; ++i)
        std::cout << GRN << ((u << i) >> 31);
    std::cout << " ";
    for (int i{9}; i < 32; ++i) 
        std::cout << RED << ((u << i) >> 31);
    std::cout << NC;
};

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

// --------------------------------------------------------------------------------
// checking utils

// CUDA error checking code
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, 
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

// CUBLAS error checking code
void cublas_check(cublasStatus_t error, const char* file, int line) {
    if (error != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[CUBLAS ERROR] at file %s:%d\n%s\n", file, line,
            cublasGetStatusString(error));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

#define cublasCheck(err) cublas_check(err, __FILE__, __LINE__)

// --------------------------------------------------------------------------------
// checking utils
float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 0.f;
    }
    return arr;
}

int cuda_arch_major = 0;
int cuda_arch_minor = 0;
int cuda_num_SMs = 0; // for persistent threads where we want 1 threadblock per SM
int cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU


// testing and benchmarking tools
template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count) {
    // copy from host to device with data type conversion
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);

    // returning here and using the checking macro after, leads to better line
    // info in case of error (instead of checking the status at cudaMemcpy)
    return status;
}

void setup_main() {
    srand(0);   // ensure reproducibility between runs

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_num_SMs = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, 
                std::size_t num_elements, T tolerance=1e-4) {

    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;

    float epsilon = FLT_EPSILON;

    for (int i = 0; i < num_elements; ++i) {
        // skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }

        // actual tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare before to scrub L2 cache between benchmarks
    // just memset a large dummy array. recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void *flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}