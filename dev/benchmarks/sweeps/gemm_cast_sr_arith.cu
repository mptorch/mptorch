// Is there a fast path for RoundMode::SR?
//
// SR has no hardware analogue -- binary32 rounds to nearest, so nothing in
// the FPU reproduces "add random bits below the retained significand, then
// truncate". What it does share with RNE is everything *after* the rounding:
// clip_normal_range_exponent, which the RNE fast path (cast_binaryK_rne_fast,
// dd03614) replaced with a single magnitude compare. This program prototypes
// that half on its own, before committing it to cast_binaryK.h /
// cast_superfp.h:
//
//   original  cast_binaryK_stochastic(x, r, rb, man, exp, bias, ...)
//   integer   the BinaryKParams / SuperfpParams overload, gate forced off
//   fast      the same overload as shipped -- clip_normal_range_exponent
//             replaced by `|y| > max_finite ? copysign(ovf, x) : y`
//
//  (a) bit-exactness over all 2^32 float inputs x 4 random draws per input,
//      three-way against the original *_bits-taking overload as well;
//  (b) GEMM timing of the two in the shipped kernel's shape.
//
// Build & run (from the repo root):
//   nvcc -O3 -std=c++17 -arch=sm_89 --expt-relaxed-constexpr -I mptorch/csrc \
//        dev/benchmarks/sweeps/gemm_cast_sr_arith.cu -o /tmp/gemm_cast_sr_arith \
//     && /tmp/gemm_cast_sr_arith
// Pass `spot`, `sweep`, `gemm` or nothing (= gemm + sweep).
//
// Since a0407f9 (setup.py's -ffp-contract=off and -DMPTORCH_FAST_CAST=1) both
// of these paths are compiled into the host build as well, so the same
// three-way check builds as plain C++ too:
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc -x c++ dev/benchmarks/sweeps/gemm_cast_sr_arith.cu \
//       -o /tmp/gemm_cast_sr_arith_host && /tmp/gemm_cast_sr_arith_host spot
// `gemm` is CUDA-only. Unlike the two RNE programs there is no arithmetic
// identity at stake here -- the change is a magnitude compare standing in for
// clip_normal_range_exponent, integer either way -- so the host has nothing
// new to prove; `spot` runs anyway because "nothing new to prove" is a claim
// worth a measurement. See gemm_cast_float_arith.cu's header for why the
// format grid is the GPU's job. A configuration costs four draws per input,
// so roughly 20 s of 16 host threads against 0.5 s of a GPU.
#include "common/cast_binaryK.h"
#include "common/cast_superfp.h"
#include "common/modes.h"
#include "reference_casts.h" // the "original" arm: the format-taking casts
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if defined(__CUDACC__)
#define CHECK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){printf("ERR %s %d\n",cudaGetErrorString(e_),__LINE__);exit(1);} }while(0)
#endif

// ---------------------------------------------------------------- exhaustive
struct Counts { unsigned long long fast_vs_int, fast_vs_orig, int_vs_orig; unsigned first_bad; unsigned first_r; };

CUDA_HOST_DEVICE_INLINE uint32_t bits_of(float f) { return FLOAT_TO_BITS(&f); }

CUDA_HOST_DEVICE_INLINE uint32_t mix(uint32_t v)
{
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return v;
}

#if defined(__CUDACC__)
constexpr int PER_THREAD = 16;

__global__ void exhaustive_bk(int man, int expb, int bias, int rbits, bool is_signed,
                              SaturationMode sat, SubnormalsMode subm,
                              BinaryKParams pfast, BinaryKParams pslow, Counts *c)
{
    uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long nfi = 0, nfo = 0, nio = 0;
    uint32_t first = 0, firstr = 0;
    for (int k = 0; k < PER_THREAD; ++k)
    {
        uint32_t bits = (uint32_t)(base + k);
        float x = bits_to_float(bits);
        // 0 and all-ones bracket the draw; the tie value and a hash cover the
        // interior, and the hash differs for every one of the 2^32 inputs.
        uint32_t rs[4] = {0u, 0x007FFFFFu, 0x00400000u, mix(bits)};
        for (int j = 0; j < 4; ++j)
        {
            uint32_t o = bits_of(cast_binaryK_stochastic(x, rs[j], rbits, man, expb, bias, is_signed, sat, subm));
            uint32_t i = bits_of(cast_binaryK_stochastic(x, rs[j], rbits, is_signed, subm, pslow));
            uint32_t f = bits_of(cast_binaryK_stochastic(x, rs[j], rbits, is_signed, subm, pfast));
            if (f != i) { if (nfi++ == 0) { first = bits; firstr = rs[j]; } }
            if (f != o) ++nfo;
            if (i != o) ++nio;
        }
    }
    if (nfi) { if (atomicAdd(&c->fast_vs_int, nfi) == 0) { c->first_bad = first; c->first_r = firstr; } }
    if (nfo) atomicAdd(&c->fast_vs_orig, nfo);
    if (nio) atomicAdd(&c->int_vs_orig, nio);
}

__global__ void exhaustive_sfp(int man, int expb, int nb, int bias, int rbits, bool is_signed,
                               SaturationMode sat, SuperfpParams pfast, SuperfpParams pslow, Counts *c)
{
    uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long nfi = 0, nfo = 0, nio = 0;
    uint32_t first = 0, firstr = 0;
    for (int k = 0; k < PER_THREAD; ++k)
    {
        uint32_t bits = (uint32_t)(base + k);
        float x = bits_to_float(bits);
        uint32_t rs[4] = {0u, 0x007FFFFFu, 0x00400000u, mix(bits)};
        for (int j = 0; j < 4; ++j)
        {
            uint32_t o = bits_of(cast_superfp_stochastic(x, rs[j], rbits, man, expb, nb, bias, is_signed, sat));
            uint32_t i = bits_of(cast_superfp_stochastic(x, rs[j], rbits, is_signed, pslow));
            uint32_t f = bits_of(cast_superfp_stochastic(x, rs[j], rbits, is_signed, pfast));
            if (f != i) { if (nfi++ == 0) { first = bits; firstr = rs[j]; } }
            if (f != o) ++nfo;
            if (i != o) ++nio;
        }
    }
    if (nfi) { if (atomicAdd(&c->fast_vs_int, nfi) == 0) { c->first_bad = first; c->first_r = firstr; } }
    if (nfo) atomicAdd(&c->fast_vs_orig, nfo);
    if (nio) atomicAdd(&c->int_vs_orig, nio);
}

static void run_exhaustive_bk(int man, int expb, int bias, int rbits, bool sgn, SaturationMode sat,
                              SubnormalsMode subm, const BinaryKParams &pfast, const BinaryKParams &pslow,
                              Counts *h)
{
    static Counts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(Counts)));
    CHECK(cudaMemset(dc, 0, sizeof(Counts)));
    exhaustive_bk<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(man, expb, bias, rbits, sgn, sat, subm, pfast, pslow, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
}

static void run_exhaustive_sfp(int man, int expb, int nb, int bias, int rbits, bool sgn, SaturationMode sat,
                               const SuperfpParams &pfast, const SuperfpParams &pslow, Counts *h)
{
    static Counts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(Counts)));
    CHECK(cudaMemset(dc, 0, sizeof(Counts)));
    exhaustive_sfp<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(man, expb, nb, bias, rbits, sgn, sat, pfast, pslow, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
}
#else
// The host arm of both kernels: the same patterns and the same four draws per
// input, across OpenMP threads instead of CUDA blocks. `first_bad` is the
// smallest mismatching pattern rather than an arbitrary one, since a reduction
// has no "first"; `first_r` is the draw that goes with it.
static void run_exhaustive_bk(int man, int expb, int bias, int rbits, bool sgn, SaturationMode sat,
                              SubnormalsMode subm, const BinaryKParams &pfast, const BinaryKParams &pslow,
                              Counts *h)
{
    unsigned long long nfi = 0, nfo = 0, nio = 0;
    uint64_t first = 0xFFFFFFFFFFFFFFFFull; // (bits << 24) | draw, so min picks both
#pragma omp parallel for schedule(static) reduction(+ : nfi, nfo, nio) reduction(min : first)
    for (int64_t hi = 0; hi < (1 << 16); ++hi)
        for (uint32_t lo = 0; lo < (1u << 16); ++lo)
        {
            uint32_t bits = (uint32_t)(((uint64_t)hi << 16) | lo);
            float x = bits_to_float(bits);
            uint32_t rs[4] = {0u, 0x007FFFFFu, 0x00400000u, mix(bits)};
            for (int j = 0; j < 4; ++j)
            {
                uint32_t o = bits_of(cast_binaryK_stochastic(x, rs[j], rbits, man, expb, bias, sgn, sat, subm));
                uint32_t i = bits_of(cast_binaryK_stochastic(x, rs[j], rbits, sgn, subm, pslow));
                uint32_t f = bits_of(cast_binaryK_stochastic(x, rs[j], rbits, sgn, subm, pfast));
                if (f != i) { ++nfi; uint64_t k = ((uint64_t)bits << 24) | rs[j]; if (k < first) first = k; }
                if (f != o) ++nfo;
                if (i != o) ++nio;
            }
        }
    h->fast_vs_int = nfi; h->fast_vs_orig = nfo; h->int_vs_orig = nio;
    h->first_bad = nfi ? (uint32_t)(first >> 24) : 0;
    h->first_r = nfi ? (uint32_t)(first & 0x00FFFFFFu) : 0;
}

static void run_exhaustive_sfp(int man, int expb, int nb, int bias, int rbits, bool sgn, SaturationMode sat,
                               const SuperfpParams &pfast, const SuperfpParams &pslow, Counts *h)
{
    unsigned long long nfi = 0, nfo = 0, nio = 0;
    uint64_t first = 0xFFFFFFFFFFFFFFFFull;
#pragma omp parallel for schedule(static) reduction(+ : nfi, nfo, nio) reduction(min : first)
    for (int64_t hi = 0; hi < (1 << 16); ++hi)
        for (uint32_t lo = 0; lo < (1u << 16); ++lo)
        {
            uint32_t bits = (uint32_t)(((uint64_t)hi << 16) | lo);
            float x = bits_to_float(bits);
            uint32_t rs[4] = {0u, 0x007FFFFFu, 0x00400000u, mix(bits)};
            for (int j = 0; j < 4; ++j)
            {
                uint32_t o = bits_of(cast_superfp_stochastic(x, rs[j], rbits, man, expb, nb, bias, sgn, sat));
                uint32_t i = bits_of(cast_superfp_stochastic(x, rs[j], rbits, sgn, pslow));
                uint32_t f = bits_of(cast_superfp_stochastic(x, rs[j], rbits, sgn, pfast));
                if (f != i) { ++nfi; uint64_t k = ((uint64_t)bits << 24) | rs[j]; if (k < first) first = k; }
                if (f != o) ++nfo;
                if (i != o) ++nio;
            }
        }
    h->fast_vs_int = nfi; h->fast_vs_orig = nfo; h->int_vs_orig = nio;
    h->first_bad = nfi ? (uint32_t)(first >> 24) : 0;
    h->first_r = nfi ? (uint32_t)(first & 0x00FFFFFFu) : 0;
}
#endif

static const char *satname(SaturationMode s)
{
    return s == SaturationMode::SAT_NONE ? "SAT_NONE"
         : (s == SaturationMode::SAT_FINITE ? "SAT_FINITE" : "SAT_PROPAGATE");
}

static bool report(const char *tag, Counts &h, bool verbose)
{
    bool ok = (h.fast_vs_int == 0 && h.fast_vs_orig == 0 && h.int_vs_orig == 0);
    if (!ok || verbose)
    {
        printf("  %-46s fast!=int %llu  fast!=orig %llu  int!=orig %llu", tag,
               h.fast_vs_int, h.fast_vs_orig, h.int_vs_orig);
        if (!ok) printf("   e.g. x=0x%08x r=0x%06x", h.first_bad, h.first_r);
        printf("\n");
    }
    return ok;
}

static int run_bk(int man, int expb, int bias, int rbits, bool sgn, SaturationMode sat,
                  SubnormalsMode subm, bool verbose)
{
    bool ext = (subm == SubnormalsMode::EXTENDED_NORMALS);
    BinaryKParams pfast = make_binaryK_params(man, expb, bias, sgn, sat, ext);
    BinaryKParams pslow = pfast; pslow.fast_rne = false;
    if (!pfast.fast_rne || subm != SubnormalsMode::SUBNORMALS)
        return -1; // same gate the RNE fast path uses
    Counts h;
    run_exhaustive_bk(man, expb, bias, rbits, sgn, sat, subm, pfast, pslow, &h);
    char tag[128];
    snprintf(tag, sizeof tag, "binaryK man=%2d exp=%d bias=%4d rb=%2d %s %s", man, expb, bias, rbits,
             sgn ? "signed  " : "unsigned", satname(sat));
    return report(tag, h, verbose) ? 1 : 0;
}

static int run_sfp(int man, int expb, int nb, int bias, int rbits, bool sgn, SaturationMode sat,
                   bool verbose)
{
    SuperfpParams pfast = make_superfp_params(man, expb, nb, bias, sat);
    SuperfpParams pslow = pfast; pslow.fast_rne = false;
    if (!pfast.fast_rne)
        return -1;
    Counts h;
    run_exhaustive_sfp(man, expb, nb, bias, rbits, sgn, sat, pfast, pslow, &h);
    char tag[128];
    snprintf(tag, sizeof tag, "superfp man=%2d exp=%d nb=%2d bias=%4d rb=%2d %s %s", man, expb, nb, bias, rbits,
             sgn ? "signed  " : "unsigned", satname(sat));
    return report(tag, h, verbose) ? 1 : 0;
}

struct EB { int exp_bits, bias; };
static std::vector<EB> eb_grid()
{
    std::vector<EB> v;
    for (int e = 1; e <= 9; ++e)
    {
        int d = (1 << (e - 1)) - 1;
        for (int bias : {d, d + 4, d - 4, 1, 60, -8})
            v.push_back({e, bias});
    }
    return v;
}

// The host gate: both formats, both saturation modes, both ends of the
// rand_bits range, each over all 2^32 inputs x 4 draws, in minutes rather
// than hours. See gemm_cast_float_arith.cu's header for the sizing argument.
static void spot()
{
    int checked = 0, gated = 0, failed = 0;
    auto keep = [&](int r) { if (r < 0) ++gated; else { ++checked; if (!r) ++failed; } };
    printf("spot set -- 2^32 inputs x 4 draws per configuration, three-way\n");
    for (int man : {1, 3, 10})
        for (const EB &eb : {EB{4, 7}, EB{8, 127}})
        {
            for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE})
                keep(run_bk(man, eb.exp_bits, eb.bias, 23, true, sat, SubnormalsMode::SUBNORMALS, false));
            // nb=1 leaves ordinary operands supernormal, nb=8 makes them normal
            // -- the normal arm is the only one this change touches.
            for (int nb : {1, 8})
                keep(run_sfp(man, eb.exp_bits, nb, eb.bias, 23, true, SaturationMode::SAT_NONE, false));
        }
    // the narrow end of rand_bits, and the unsigned early return
    keep(run_bk(3, 4, 7, 1, true, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, false));
    keep(run_sfp(3, 4, 1, 7, 1, true, SaturationMode::SAT_NONE, false));
    keep(run_bk(3, 4, 7, 23, false, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, false));
    keep(run_sfp(3, 4, 1, 7, 23, false, SaturationMode::SAT_NONE, false));
    printf("total: %d configurations, each over all 2^32 floats x 4 draws\n", checked);
    printf("       %d gated off\n", gated);
    printf("       %s\n", failed ? "*** MISMATCHES ABOVE ***" : "0 mismatches");
}

static void sweep()
{
    int checked = 0, gated = 0, failed = 0;

    printf("binaryK -- 2^32 inputs x 4 random draws per configuration, three-way\n");
    for (int man = 0; man <= 24; ++man)
    {
        int nc = 0, nf = 0;
        for (const EB &eb : eb_grid())
            // rand_bits only selects which random bits reach the addition; it
            // does not touch the clip this change replaces, so the two ends of
            // its range stand in for the middle. The four draws per input are
            // what actually vary the rounded value.
            for (int rb : {1, 23})
                for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE})
                {
                    int r = run_bk(man, eb.exp_bits, eb.bias, rb, true, sat, SubnormalsMode::SUBNORMALS, false);
                    if (r < 0) ++gated; else { ++checked; ++nc; if (!r) { ++failed; ++nf; } }
                }
        if (nc) { printf("  man_bits=%-2d  %4d configurations verified, %d mismatched\n", man, nc, nf); fflush(stdout); }
    }

    printf("superfp -- same\n");
    int sc = 0, sf = 0;
    for (int man = 1; man <= 23; man += 2)
        for (const EB &eb : eb_grid())
            // normal_binades only moves the region cutoffs, which this change
            // does not touch -- gemm_cast_superfp_arith.cu sweeps it wide.
            for (int nb : {1, 2})
                for (int rb : {1, 23})
                {
                    int r = run_sfp(man, eb.exp_bits, nb, eb.bias, rb, true, SaturationMode::SAT_NONE, false);
                    if (r < 0) ++gated; else { ++sc; if (!r) ++sf; }
                }
    printf("  %d configurations verified, %d mismatched\n", sc, sf);

    printf("unsigned confirmation\n");
    int uc = 0, uf = 0;
    for (int man : {1, 3, 10, 22})
        for (const EB &eb : {EB{4, 7}, EB{5, 15}, EB{8, 127}})
            for (int rb : {3, 23})
            {
                int r = run_bk(man, eb.exp_bits, eb.bias, rb, false, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, false);
                if (r < 0) ++gated; else { ++uc; if (!r) ++uf; }
                r = run_sfp(man, eb.exp_bits, 1, eb.bias, rb, false, SaturationMode::SAT_NONE, false);
                if (r < 0) ++gated; else { ++uc; if (!r) ++uf; }
            }
    printf("  %d configurations verified, %d mismatched\n", uc, uf);

    printf("total: %d configurations, each over all 2^32 floats x 4 draws\n", checked + sc + uc);
    printf("       %d gated off\n", gated);
    printf("       %s\n", (failed + sf + uf) ? "*** MISMATCHES ABOVE ***" : "0 mismatches");
}

// ---------------------------------------------------------------------- GEMM
#if defined(__CUDACC__)
// A stand-in for the shipped Philox stream: SR's cost in the kernel is the
// cast, and a counter-based hash keeps the random draw out of the comparison
// while still varying per element.
template<bool SFP>
struct Mul { bool is_signed; SubnormalsMode sub; int rb; BinaryKParams bp; SuperfpParams sp;
  __device__ __forceinline__ float operator()(float a,float b,uint32_t r) const {
      float v=a*b;
      if (SFP) return cast_superfp_stochastic(v,r,rb,is_signed,sp);
      return cast_binaryK_stochastic(v,r,rb,is_signed,sub,bp); } };
template<bool SFP>
struct Add { bool is_signed; SubnormalsMode sub; int rb; BinaryKParams bp; SuperfpParams sp;
  __device__ __forceinline__ float operator()(float x,uint32_t r) const {
      if (SFP) return cast_superfp_stochastic(x,r,rb,is_signed,sp);
      return cast_binaryK_stochastic(x,r,rb,is_signed,sub,bp); } };
template<bool SFP>
struct Acc { Mul<SFP> mul; Add<SFP> add; float sum=0.f; uint32_t ctr=0;
  __device__ __forceinline__ void accumulate(float a,float b){
      sum=add(sum+mul(a,b,mix(ctr++)),mix(ctr++)); }
  __device__ __forceinline__ float finalize() const {return sum;} };

constexpr int BS=16;
template<int UNROLL,class A>
__global__ __launch_bounds__(BS*BS)
void gemm(const float*__restrict__ A_,const float*__restrict__ B,float*__restrict__ C,
          int64_t M,int64_t K,int64_t N,A acc_proto)
{
    __shared__ float As[2][BS*BS]; __shared__ float Bs[2][BS*BS];
    const int tc=threadIdx.x%BS, tr=threadIdx.x/BS;
    const int64_t rId=(int64_t)blockIdx.y*BS+tr, cId=(int64_t)blockIdx.x*BS+tc;
    const int64_t nT=(K+BS-1)/BS;
    A acc=acc_proto; acc.ctr=(uint32_t)(rId*N+cId);
    As[0][tr*BS+tc]=(rId<M&&tc<K)?A_[rId*K+tc]:0.f;
    Bs[0][tr*BS+tc]=(tr<K&&cId<N)?B[cId*K+tr]:0.f;
    for(int64_t t=0;t<nT;++t){
        __syncthreads();
        float*cA=As[t%2]; float*cB=Bs[t%2];
#pragma unroll UNROLL
        for(int d=0;d<BS;++d) acc.accumulate(cA[tr*BS+d],cB[d*BS+tc]);
        if(t+1<nT){ int64_t nk=(t+1)*BS;
            As[(t+1)%2][tr*BS+tc]=(rId<M&&nk+tc<K)?A_[rId*K+nk+tc]:0.f;
            Bs[(t+1)%2][tr*BS+tc]=(nk+tr<K&&cId<N)?B[cId*K+nk+tr]:0.f; }
        __syncthreads();
    }
    if(rId<M&&cId<N) C[rId*N+cId]=acc.finalize();
}

static int M,K,N; static float*dA,*dB,*dC; static float*ref=nullptr;
static void chk(const char*n){ std::vector<float> h(M*N); CHECK(cudaMemcpy(h.data(),dC,4ull*M*N,cudaMemcpyDeviceToHost));
  if(!ref){ref=(float*)malloc(4ull*M*N);memcpy(ref,h.data(),4ull*M*N);printf("      [%s = reference]\n",n);return;}
  int bad=0; for(int i=0;i<M*N;i++) if(ref[i]!=h[i]) bad++;
  printf(bad?"      [%s MISMATCH %d/%d]\n":"      [%s bit-exact]\n",n,bad,M*N); }

static void gemm_bench()
{
    const int man=3,expb=4,bias=7,nb=1,rb=8;
    BinaryKParams bpf=make_binaryK_params(man,expb,bias,true,SaturationMode::SAT_NONE,false);
    BinaryKParams bps=bpf; bps.fast_rne=false;
    SuperfpParams spf=make_superfp_params(man,expb,nb,bias,SaturationMode::SAT_NONE);
    SuperfpParams sps=spf; sps.fast_rne=false;
    // nb=8 drops superfp's normal cutoff to |x| >= 2: operands and products
    // stay supernormal but the running sum crosses into the normal arm, which
    // is the only arm this change touches. With nb=1 the cutoff is 2^8 and
    // nothing reaches it.
    SuperfpParams wpf=make_superfp_params(man,expb,8,bias,SaturationMode::SAT_NONE);
    SuperfpParams wps=wpf; wps.fast_rne=false;
    M=K=N=1024; int iters=5;
    std::vector<float> hA(M*K),hB((size_t)N*K); srand(1234);
    for(auto&v:hA)v=(float)rand()/RAND_MAX*2-1; for(auto&v:hB)v=(float)rand()/RAND_MAX*2-1;
    CHECK(cudaMalloc(&dA,4ull*M*K));CHECK(cudaMalloc(&dB,4ull*(size_t)N*K));CHECK(cudaMalloc(&dC,4ull*M*N));
    CHECK(cudaMemcpy(dA,hA.data(),4ull*M*K,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,hB.data(),4ull*(size_t)N*K,cudaMemcpyHostToDevice));
    dim3 g((N+BS-1)/BS,(M+BS-1)/BS), b(BS*BS);
    const SubnormalsMode sm=SubnormalsMode::SUBNORMALS;
    auto run=[&](const char*name,auto lam){ for(int i=0;i<2;i++)lam(); CHECK(cudaDeviceSynchronize());
        cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
        for(int i=0;i<iters;i++)lam(); cudaEventRecord(e);CHECK(cudaEventSynchronize(e));
        float ms;cudaEventElapsedTime(&ms,s,e);ms/=iters;
        printf("%-40s %9.3f ms  %6.2f Gmac/s\n",name,ms,(double)M*N*K/(ms/1e3)/1e9); chk(name); };

    printf("\nGEMM 1024^3, split mac, SR, binaryK man=%d exp=%d bias=%d prng_bits=%d\n",man,expb,bias,rb);
    Acc<false> bS{{true,sm,rb,bps,sps},{true,sm,rb,bps,sps},0.f,0};
    Acc<false> bF{{true,sm,rb,bpf,spf},{true,sm,rb,bpf,spf},0.f,0};
    run("integer clip, unroll 4",  [&]{gemm<4 ><<<g,b>>>(dA,dB,dC,M,K,N,bS);});
    run("integer clip, unroll 16", [&]{gemm<16><<<g,b>>>(dA,dB,dC,M,K,N,bS);});
    run("compare clip, unroll 4",  [&]{gemm<4 ><<<g,b>>>(dA,dB,dC,M,K,N,bF);});
    run("compare clip, unroll 16", [&]{gemm<16><<<g,b>>>(dA,dB,dC,M,K,N,bF);});
    free(ref);ref=nullptr;

    printf("\nGEMM 1024^3, split mac, SR, superfp man=%d exp=%d nb=%d bias=%d prng_bits=%d (supernormal operands)\n",man,expb,nb,bias,rb);
    Acc<true> sS{{true,sm,rb,bps,sps},{true,sm,rb,bps,sps},0.f,0};
    Acc<true> sF{{true,sm,rb,bpf,spf},{true,sm,rb,bpf,spf},0.f,0};
    run("integer clip, unroll 4",  [&]{gemm<4 ><<<g,b>>>(dA,dB,dC,M,K,N,sS);});
    run("integer clip, unroll 16", [&]{gemm<16><<<g,b>>>(dA,dB,dC,M,K,N,sS);});
    run("compare clip, unroll 4",  [&]{gemm<4 ><<<g,b>>>(dA,dB,dC,M,K,N,sF);});
    run("compare clip, unroll 16", [&]{gemm<16><<<g,b>>>(dA,dB,dC,M,K,N,sF);});
    free(ref);ref=nullptr;

    printf("\nGEMM 1024^3, split mac, SR, superfp man=%d exp=%d nb=8 bias=%d prng_bits=%d (normal operands)\n",man,expb,bias,rb);
    Acc<true> wS{{true,sm,rb,bps,wps},{true,sm,rb,bps,wps},0.f,0};
    Acc<true> wF{{true,sm,rb,bpf,wpf},{true,sm,rb,bpf,wpf},0.f,0};
    run("integer clip, unroll 4",  [&]{gemm<4 ><<<g,b>>>(dA,dB,dC,M,K,N,wS);});
    run("integer clip, unroll 16", [&]{gemm<16><<<g,b>>>(dA,dB,dC,M,K,N,wS);});
    run("compare clip, unroll 4",  [&]{gemm<4 ><<<g,b>>>(dA,dB,dC,M,K,N,wF);});
    run("compare clip, unroll 16", [&]{gemm<16><<<g,b>>>(dA,dB,dC,M,K,N,wF);});
    CHECK(cudaFree(dA));CHECK(cudaFree(dB));CHECK(cudaFree(dC));free(ref);ref=nullptr;
}
#else
// The host half of the same question, in the same shape: a plain triple loop,
// single-threaded, with the shipped runtime gate flipped between runs. It is
// not a model of the shipped CPU kernel -- that one is tiled and threaded and
// shares its clock with memory, and dev/benchmarks/timing/cpu_gemm_tile_state.py
// measures it end to end -- but of the cast's own cost, which is what this
// file is about. The two runs are compared bit for bit as well, so a timing
// run is also a (much smaller) correctness check.
static void gemm_bench()
{
    const int man=3,expb=4,bias=7,rb=8,M=192,K=192,N=192;
    const SubnormalsMode sm=SubnormalsMode::SUBNORMALS;
    BinaryKParams bpf=make_binaryK_params(man,expb,bias,true,SaturationMode::SAT_NONE,false);
    BinaryKParams bps=bpf; bps.fast_rne=false;
    SuperfpParams spf=make_superfp_params(man,expb,1,bias,SaturationMode::SAT_NONE);
    SuperfpParams sps=spf; sps.fast_rne=false;
    // nb=8 drops superfp's normal cutoff to |x| >= 2, so the running sum
    // crosses into the normal arm -- the only arm this change touches. With
    // nb=1 the cutoff is 2^8 and nothing reaches it.
    SuperfpParams wpf=make_superfp_params(man,expb,8,bias,SaturationMode::SAT_NONE);
    SuperfpParams wps=wpf; wps.fast_rne=false;
    std::vector<float> A((size_t)M*K),B((size_t)K*N),C((size_t)M*N),ref;
    srand(1234);
    for(auto&v:A)v=(float)rand()/RAND_MAX*2-1; for(auto&v:B)v=(float)rand()/RAND_MAX*2-1;
    uint32_t ctr=0;
    auto run=[&](const char*name,auto step){
        auto once=[&]{ ctr=0; for(int i=0;i<M;++i) for(int j=0;j<N;++j){ float acc=0.f;
            for(int k=0;k<K;++k) acc=step(A[(size_t)i*K+k],B[(size_t)k*N+j],acc,ctr);
            C[(size_t)i*N+j]=acc; } };
        once();
        auto t0=std::chrono::steady_clock::now();
        for(int r=0;r<3;++r) once();
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/3.0;
        printf("%-40s %9.3f ms  %6.2f Gmac/s",name,ms,(double)M*N*K/(ms/1e3)/1e9);
        if(ref.empty()){ref=C;printf("      [reference]\n");}
        else{ size_t bad=0; for(size_t i=0;i<C.size();++i) if(bits_of(ref[i])!=bits_of(C[i])) ++bad;
              if(bad) printf("      [MISMATCH %zu/%zu]\n",bad,C.size()); else printf("      [bit-exact]\n"); } };
    auto bk=[&](const BinaryKParams&p){ return [&p,sm,rb](float a,float b,float acc,uint32_t&c){
        return cast_binaryK_stochastic(acc+cast_binaryK_stochastic(a*b,mix(c++),rb,true,sm,p),mix(c++),rb,true,sm,p); }; };
    auto sfp=[&](const SuperfpParams&p){ return [&p,rb](float a,float b,float acc,uint32_t&c){
        return cast_superfp_stochastic(acc+cast_superfp_stochastic(a*b,mix(c++),rb,true,p),mix(c++),rb,true,p); }; };

    printf("\nGEMM %d^3, split mac, SR, binaryK man=%d exp=%d bias=%d prng_bits=%d, 1 thread\n",M,man,expb,bias,rb);
    run("integer clip",bk(bps)); run("compare clip",bk(bpf)); ref.clear();
    printf("\nGEMM %d^3, split mac, SR, superfp nb=1 (supernormal operands), 1 thread\n",M);
    run("integer clip",sfp(sps)); run("compare clip",sfp(spf)); ref.clear();
    printf("\nGEMM %d^3, split mac, SR, superfp nb=8 (normal operands), 1 thread\n",M);
    run("integer clip",sfp(wps)); run("compare clip",sfp(wpf));
}
#endif

int main(int argc,char**argv)
{
    std::string what = argc > 1 ? argv[1] : "all";
    if (what == "spot") { spot(); return 0; }
    if (what == "all" || what == "gemm") gemm_bench();
    if (what == "all" || what == "sweep") sweep();
    return 0;
}
