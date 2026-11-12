// The shipped CUDA GEMM kernel under RoundMode::SR, timed outside torch, so a
// change to the stochastic casts can be weighed against the build before it
// without rebuilding the extension (written for cf69807, which made every
// stochastic cast exact, in integer arithmetic on the word, and added
// StochasticB and StochasticC).
//
// It calls CudaBackend::launch_as itself, which instantiates the kernel from
// whatever headers -I names. So one source, compiled against two trees, is an
// A/B of the casts and nothing else: the tree before the exact casts from
// `git archive 4164cee mptorch/csrc | tar -x -C DIR`, and the working tree.
// Alternate the two binaries, several rounds, and take each one's minimum:
//
//   T=.venv/lib/python3.13/site-packages/torch/include
//   for tree in DIR/mptorch/csrc mptorch/csrc; do
//     nvcc -O3 -std=c++20 -arch=sm_89 --expt-relaxed-constexpr -I $tree \
//          -I $T -I $T/torch/csrc/api/include dev/benchmarks/timing/gemm_sr_ab.cu \
//          -o ...; done
//
// Each op is timed at 1024^3 (min over --reps launches), with the output's
// hash printed, so two variants meant to round alike can be compared too.
// The formats are sr_variants.py's: binaryK split Binary8p4 x Binary16p11,
// binaryK fused Binary16p11, superfp split SuperFP(3,4,2,7) x
// SuperFP(10,5,8,15), each with 8 random bits, saturation SAT_NONE and
// SUBNORMALS, on N(0,1) inputs. Note what those inputs reach: about 3% of
// the products are under Binary8p4's 2^-7, so about half of all warps take
// binaryK's subnormal arm on every multiply, and every cast of the superfp
// pair is a supernormal (their normal binades start at 2^7 and 2^9).
//
// What it found (2026-11-11, RTX 4060, ps per mac step, min of 20; the
// stable digits are the first three):
//
//                       binaryK split  binaryK fused  superfp split
//   4164cee (before)          19.6           9.5           18.6
//   first build: every arm    27.0          10.5           25.2
//     inline or a call
//   one range test, inline    21.3           9.6           21.2
//     arms (shipped)
//   the same, StochasticA     20.2           9.05          19.8
//     only (not a build)
//
// Inlined, the exact arms took the K-loop past NVVM's size for unswitching
// (one copy of the loop instead of two, about 6% of the fused kernel); a
// call clobbered the registers the slab's A row sits in (32 scalar LDS a
// slab instead of 20) and cost superfp, whose arm runs every step, 4.5 ps;
// and the hot path tested each region against its own invariant. The last
// row is the word forms without B's and C's constants: what it costs to pick
// StochasticA, B or C at run time, from constants the policy builds, in the
// one SR instantiation each kernel has, 5-7%.

#include "cuda/custom_matmul_kernel.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CHECK(x)                                                                          \
  do                                                                                      \
  {                                                                                       \
    cudaError_t e_ = (x);                                                                 \
    if (e_ != cudaSuccess)                                                                \
    {                                                                                     \
      std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__);       \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

using namespace mptorch::gemm;
using mptorch::gemm_cuda::CudaBackend;

namespace
{
  template <class Args>
  void run(const char *name, const GemmShape &s, const Args &args, int reps)
  {
    CudaBackend::LaunchContext ctx;
    ctx.rng = at::PhiloxCudaState(1234, 0);
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CudaBackend::launch_as<float, Args>(s, args, ctx); // warm
    CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < reps; ++i)
    {
      CHECK(cudaEventRecord(t0));
      CudaBackend::launch_as<float, Args>(s, args, ctx);
      CHECK(cudaEventRecord(t1));
      CHECK(cudaEventSynchronize(t1));
      float ms;
      CHECK(cudaEventElapsedTime(&ms, t0, t1));
      best = ms < best ? ms : best;
    }
    std::vector<uint32_t> out(static_cast<size_t>(s.M * s.N));
    CHECK(cudaMemcpy(out.data(), s.c, out.size() * 4, cudaMemcpyDeviceToHost));
    uint64_t h = 1469598103934665603ull;
    for (uint32_t w : out)
      h = (h ^ w) * 1099511628211ull;
    const double steps = double(s.M) * double(s.K) * double(s.N);
    std::printf("%-20s %8.3f ms  %7.3f ps/step  hash %016llx\n", name, best, best * 1e9 / steps,
                (unsigned long long)h);
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
  }
} // namespace

int main(int argc, char **argv)
{
  const int64_t n = 1024;
  int reps = 20;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--reps"))
      reps = std::atoi(argv[i + 1]);

  // N(0,1) by Box-Muller on a fixed LCG
  std::vector<float> host(static_cast<size_t>(2 * n * n));
  uint64_t st = 0x9E3779B97F4A7C15ull;
  auto u01 = [&]
  {
    st = st * 6364136223846793005ull + 1442695040888963407ull;
    return ((st >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  };
  for (auto &v : host)
    v = static_cast<float>(std::sqrt(-2.0 * std::log(u01())) * std::cos(6.283185307179586 * u01()));
  float *a, *b, *c;
  CHECK(cudaMalloc(&a, n * n * 4));
  CHECK(cudaMalloc(&b, n * n * 4));
  CHECK(cudaMalloc(&c, n * n * 4));
  CHECK(cudaMemcpy(a, host.data(), n * n * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b, host.data() + n * n, n * n * 4, cudaMemcpyHostToDevice));

  GemmShape s;
  s.a = a;
  s.b = b;
  s.c = c;
  s.dt = mptorch::GemmDtype::Float;
  s.M = s.K = s.N = n;
  s.rm = RoundMode::SR;
  s.use_rng = true;

  BinaryKSplitArgs bk;
  bk.mul = {3, 4, 8};
  bk.mul_c = {true, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, 8};
  bk.accumulate_quant = true;
  bk.acc = {10, 5, 16};
  bk.acc_c = {true, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, 8};
  run("binaryK split", s, bk, reps);

  BinaryKFusedArgs bf;
  bf.fma_quant = true;
  bf.fma = {10, 5, 16};
  bf.fma_c = {true, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, 8};
  run("binaryK fused", s, bf, reps);

  SuperfpSplitArgs sf;
  sf.mul = {3, 4, 2, 7};
  sf.mul_c = {true, SaturationMode::SAT_NONE, 8};
  sf.accumulate_quant = true;
  sf.acc = {10, 5, 8, 15};
  sf.acc_c = {true, SaturationMode::SAT_NONE, 8};
  run("superfp split", s, sf, reps);

  CHECK(cudaFree(a));
  CHECK(cudaFree(b));
  CHECK(cudaFree(c));
  return 0;
}
