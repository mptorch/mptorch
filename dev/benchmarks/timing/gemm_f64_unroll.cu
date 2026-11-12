// The binary64 GEMM kernels at other K-loop unroll factors than the shipped
// one (MPTORCH_GEMM_F64_UNROLL, 2 in cuda/custom_matmul_kernel.cuh), timed as
// the build compiles them: this file instantiates the shipped launchers for
// NAIVE, KAHAN, BLOCK and TREE over a binaryK split mac, so each build of it
// is the kernels the extension would hold at that factor. It was written so
// that the binary64 kernels' factor would be measured rather than inherited
// from NAIVE's 2: at 1024^3, RNE and SR, every one was within 0.3% from 1 to
// 4, bound by the GPU's FP64 rate, so 2 stayed (2026-10-15, RTX 4060 Laptop).
//
// Build & run (from the repo root), once per factor; -Xptxas -v prints each
// kernel's registers and stack frame:
//   for U in 1 2 4; do
//     nvcc -O3 -std=c++20 -arch=sm_89 -DMPTORCH_GEMM_F64_UNROLL=$U \
//          -I mptorch/csrc -I .venv/lib/python3.13/site-packages/torch/include \
//          dev/benchmarks/timing/gemm_f64_unroll.cu -o /tmp/f64_unroll_$U && /tmp/f64_unroll_$U
//   done

#include "cuda/custom_matmul_kernel.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace mptorch::gemm;
using namespace mptorch::gemm_cuda;

namespace mptorch::gemm_cuda
{
    template void CudaBackend::launch_as<double, BinaryKSplitArgs>(const GemmShape &, const BinaryKSplitArgs &,
                                                                   const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, BinaryKSplitArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, BinaryKSplitArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::TREE, BinaryKSplitArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKSplitArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda

#define CHECK(x)                                                                    \
    do                                                                              \
    {                                                                               \
        cudaError_t e_ = (x);                                                       \
        if (e_ != cudaSuccess)                                                      \
        {                                                                           \
            std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

int main()
{
    const int64_t n = 1024;
    std::vector<double> ha(n * n), hb(n * n);
    std::srand(1234);
    for (auto &v : ha)
        v = static_cast<double>(std::rand()) / RAND_MAX * 2 - 1;
    for (auto &v : hb)
        v = static_cast<double>(std::rand()) / RAND_MAX * 2 - 1;
    double *a, *b, *c;
    CHECK(cudaMalloc(&a, 8 * n * n));
    CHECK(cudaMalloc(&b, 8 * n * n));
    CHECK(cudaMalloc(&c, 8 * n * n));
    CHECK(cudaMemcpy(a, ha.data(), 8 * n * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b, hb.data(), 8 * n * n, cudaMemcpyHostToDevice));

    GemmShape s;
    s.a = a;
    s.b = b;
    s.c = c;
    s.dt = mptorch::GemmDtype::Double;
    s.M = s.K = s.N = n;

    // Binary8p4 products into a Binary16p11 accumulator, Binary16p11 outer.
    BinaryKSplitArgs base;
    base.mul = binaryK_widths(8, 4, 8, true);
    base.mul_c = BinaryKCommon{true, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, 0};
    base.accumulate_quant = true;
    base.acc = binaryK_widths(16, 11, 16, true);
    base.acc_c = base.mul_c;
    CudaBackend::LaunchContext ctx{};

    auto time = [&](const char *name, auto &&launch)
    {
        launch(); // warm-up
        CHECK(cudaDeviceSynchronize());
        double best = 1e30;
        for (int r = 0; r < 5; ++r)
        {
            auto t0 = std::chrono::steady_clock::now();
            launch();
            CHECK(cudaDeviceSynchronize());
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            best = ms < best ? ms : best;
        }
        std::printf("U=%d  %-6s %8.2f ms\n", MPTORCH_GEMM_F64_UNROLL, name, best);
    };

    for (RoundMode rm : {RoundMode::RNE, RoundMode::SR})
    {
        s.rm = rm;
        s.use_rng = rm == RoundMode::SR;
        std::printf("-- %s\n", rm == RoundMode::RNE ? "RNE" : "SR");
        time("NAIVE", [&] { CudaBackend::launch_as<double>(s, base, ctx); });
        for (auto [alg, name, bs] : {std::tuple{AccumulateAlgorithm::KAHAN, "KAHAN", 0},
                                     std::tuple{AccumulateAlgorithm::BLOCK, "BLOCK", 32},
                                     std::tuple{AccumulateAlgorithm::TREE, "TREE", 16}})
        {
            AccumulateArgs<BinaryKSplitArgs> args;
            args.base = base;
            args.alg = alg;
            args.block_size = bs;
            args.outer_quant = alg != AccumulateAlgorithm::KAHAN;
            args.outer = base.acc;
            args.outer_c = base.acc_c;
            switch (alg)
            {
            case AccumulateAlgorithm::KAHAN:
                time(name, [&] { CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN>(s, args, ctx); });
                break;
            case AccumulateAlgorithm::BLOCK:
                time(name, [&] { CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK>(s, args, ctx); });
                break;
            default:
                time(name, [&] { CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::TREE>(s, args, ctx); });
                break;
            }
        }
    }
    return 0;
}
