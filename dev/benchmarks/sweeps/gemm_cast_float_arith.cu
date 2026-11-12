// The float-arithmetic RNE fast path in cast_binaryK.h (cast_binaryK_rne_fast).
//
//  (a) exhaustive bit-exactness check over all 2^32 float inputs, for every
//      (man_bits, exp_bits, bias, is_signed, saturation, subnormals) in a
//      sweep -- three-way, so a disagreement is attributable:
//        original   cast_binaryK_nearest_even(x, man, exp, bias, ...)
//        integer    the BinaryKParams overload with the gate forced off
//        fast       the BinaryKParams overload as shipped
//      The gate itself is the shipped one (BinaryKParams::fast_rne); this
//      program only reports whether it is honest.
//  (b) GEMM timing of the two cast paths in the shipped kernel's shape.
//
// Build & run (from the repo root):
//   nvcc -O3 -std=c++17 -arch=sm_89 --expt-relaxed-constexpr -I mptorch/csrc \
//        dev/benchmarks/sweeps/gemm_cast_float_arith.cu -o /tmp/gemm_cast_float_arith \
//     && /tmp/gemm_cast_float_arith
// Add -Xptxas -v for the kernels' register and stack-frame figures. Pass
// `spot`, `sweep`, `coarse`, `gemm`, `one` or nothing (= sweep + gemm).
//
// Since a0407f9 (setup.py's -ffp-contract=off and -DMPTORCH_FAST_CAST=1) the
// same fast path is compiled into the host build too, so the same question has
// to be asked of the host compiler -- the identity is the same, but "is `t` a
// separately rounded binary32 value" is now g++'s answer, not nvcc's. The sweep
// therefore also builds as plain C++, with the same grids and the same output,
// over OpenMP threads instead of CUDA blocks:
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc -x c++ dev/benchmarks/sweeps/gemm_cast_float_arith.cu \
//       -o /tmp/gemm_cast_float_arith_host && /tmp/gemm_cast_float_arith_host coarse
// Those are the flags setup.py builds the extension with; changing them is the
// point of the exercise, so pass what you are asking about. `gemm` is CUDA-only
// (dev/benchmarks/timing/cpu_gemm_tile_state.py is the host equivalent).
//
// One configuration costs ~0.1 s of a GPU and ~5 s of 16 host threads, which
// makes `sweep` an overnight run on the host and `spot` the mode to reach for
// there -- and `spot` is the honest host gate, not a shortcut. The full grid
// answers a question about the *formats*: which ones make_binaryK_params may
// admit. That answer is the identity plus integer arithmetic, identical on
// both backends, and the GPU sweep settles it. What is open on the host is a
// question about the *compiler*: whether these operations survive as separately
// rounded binary32 ones. Nothing about that varies with man_bits or bias -- the
// inlined code is the same and the format only reaches it as runtime values --
// so a handful of configurations, each over all 2^32 inputs and between them
// entering every arm, is a complete answer where a thousand more would only
// repeat it.
#include "common/cast_binaryK.h"
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
struct Counts { unsigned long long fast_vs_int, fast_vs_orig, int_vs_orig; unsigned first_bad; };

CUDA_HOST_DEVICE_INLINE uint32_t bits_of(float f) { return FLOAT_TO_BITS(&f); }

#if defined(__CUDACC__)
// PER_THREAD consecutive bit patterns per thread: 2^32 one-value threads
// spend most of their time in block scheduling, not in the casts.
constexpr int PER_THREAD = 64;

__global__ void exhaustive(int man, int expb, int bias, bool is_signed,
                           SaturationMode sat, SubnormalsMode subm,
                           BinaryKParams pfast, BinaryKParams pslow, Counts *c)
{
    uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long nfi = 0, nfo = 0, nio = 0;
    uint32_t first = 0;
    for (int k = 0; k < PER_THREAD; ++k)
    {
        uint32_t bits = (uint32_t)(base + k);
        float x = bits_to_float(bits);
        uint32_t o = bits_of(cast_binaryK_nearest_even(x, man, expb, bias, is_signed, sat, subm));
        uint32_t i = bits_of(cast_binaryK_nearest_even(x, is_signed, subm, pslow));
        uint32_t f = bits_of(cast_binaryK_nearest_even(x, is_signed, subm, pfast));
        if (f != i) { if (nfi++ == 0) first = bits; }
        if (f != o) ++nfo;
        if (i != o) ++nio;
    }
    if (nfi) { if (atomicAdd(&c->fast_vs_int, nfi) == 0) c->first_bad = first; }
    if (nfo) atomicAdd(&c->fast_vs_orig, nfo);
    if (nio) atomicAdd(&c->int_vs_orig, nio);
}

static void run_exhaustive(int man, int expb, int bias, bool is_signed, SaturationMode sat,
                           SubnormalsMode subm, const BinaryKParams &pfast, const BinaryKParams &pslow,
                           Counts *h)
{
    static Counts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(Counts)));
    CHECK(cudaMemset(dc, 0, sizeof(Counts)));
    exhaustive<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(man, expb, bias, is_signed, sat, subm, pfast, pslow, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
}
#else
// The host arm: the same 2^32 patterns over the same three casts, split across
// OpenMP threads by their high half-word. `first_bad` is the numerically
// smallest mismatching pattern rather than an arbitrary one -- a reduction has
// no "first" -- which is if anything the more useful of the two to be handed.
static void run_exhaustive(int man, int expb, int bias, bool is_signed, SaturationMode sat,
                           SubnormalsMode subm, const BinaryKParams &pfast, const BinaryKParams &pslow,
                           Counts *h)
{
    unsigned long long nfi = 0, nfo = 0, nio = 0;
    uint32_t first = 0xFFFFFFFFu;
#pragma omp parallel for schedule(static) reduction(+ : nfi, nfo, nio) reduction(min : first)
    for (int64_t hi = 0; hi < (1 << 16); ++hi)
        for (uint32_t lo = 0; lo < (1u << 16); ++lo)
        {
            uint32_t bits = (uint32_t)(((uint64_t)hi << 16) | lo);
            float x = bits_to_float(bits);
            uint32_t o = bits_of(cast_binaryK_nearest_even(x, man, expb, bias, is_signed, sat, subm));
            uint32_t i = bits_of(cast_binaryK_nearest_even(x, is_signed, subm, pslow));
            uint32_t f = bits_of(cast_binaryK_nearest_even(x, is_signed, subm, pfast));
            if (f != i) { ++nfi; if (bits < first) first = bits; }
            if (f != o) ++nfo;
            if (i != o) ++nio;
        }
    h->fast_vs_int = nfi;
    h->fast_vs_orig = nfo;
    h->int_vs_orig = nio;
    h->first_bad = nfi ? first : 0;
}
#endif

static int run_case(int man, int expb, int bias, bool is_signed, SaturationMode sat, SubnormalsMode subm,
                    bool verbose)
{
    bool ext = (subm == SubnormalsMode::EXTENDED_NORMALS);
    BinaryKParams pfast = make_binaryK_params(man, expb, bias, is_signed, sat, ext);
    BinaryKParams pslow = pfast;
    pslow.fast_rne = false;
    if (!pfast.fast_rne)
        return -1; // gated off: the fast path is never entered, nothing to verify

    Counts h;
    run_exhaustive(man, expb, bias, is_signed, sat, subm, pfast, pslow, &h);
    bool ok = (h.fast_vs_int == 0 && h.fast_vs_orig == 0 && h.int_vs_orig == 0);
    if (!ok || verbose)
    {
        float xv = *reinterpret_cast<float *>(&h.first_bad);
        printf("  man=%2d exp=%d bias=%4d %s %-13s %-16s  fast!=int %llu  fast!=orig %llu  int!=orig %llu",
               man, expb, bias, is_signed ? "signed  " : "unsigned",
               sat == SaturationMode::SAT_NONE ? "SAT_NONE" : (sat == SaturationMode::SAT_FINITE ? "SAT_FINITE" : "SAT_PROPAGATE"),
               subm == SubnormalsMode::SUBNORMALS ? "SUBNORMALS" : (subm == SubnormalsMode::NORMALS ? "NORMALS" : "EXTENDED_NORMALS"),
               h.fast_vs_int, h.fast_vs_orig, h.int_vs_orig);
        if (!ok) printf("   e.g. 0x%08x = %g", h.first_bad, xv);
        printf("\n");
    }
    return ok ? 1 : 0;
}

// exp_bits/bias pairs: the IEEE-style default bias for that width, plus
// shifted, degenerate and negative ones -- the gate has to hold for whatever
// the caller passes, not just the conventional choice.
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

// The host gate: every arm of the fast path, each over all 2^32 inputs, in a
// few minutes rather than a night. See the header comment for why the format
// axes do not need sweeping to answer the host's question.
static void spot()
{
    int checked = 0, gated = 0, failed = 0;
    auto one = [&](int man, int expb, int bias, bool sgn, SaturationMode sat, SubnormalsMode sm) {
        int r = run_case(man, expb, bias, sgn, sat, sm, false);
        if (r < 0) ++gated;
        else { ++checked; if (r == 0) ++failed; }
    };
    printf("spot set -- 2^32 inputs per configuration, three-way\n");
    // man_bits 0 (no mantissa: the round-up-always formula) through 22, over
    // the three conventional widths and both saturation modes the gate admits.
    for (int man : {0, 1, 3, 10, 22})
        for (const EB &eb : {EB{4, 7}, EB{5, 15}, EB{8, 127}})
            for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE,
                                       SaturationMode::SAT_PROPAGATE})
                one(man, eb.exp_bits, eb.bias, true, sat, SubnormalsMode::SUBNORMALS);
    // the two axes that steer past the fast path rather than through it: an
    // unsigned format's early return, and the subnormals modes the gate excludes
    one(3, 4, 7, false, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS);
    one(3, 4, 7, true, SaturationMode::SAT_NONE, SubnormalsMode::NORMALS);
    one(3, 4, 7, true, SaturationMode::SAT_NONE, SubnormalsMode::EXTENDED_NORMALS);
    printf("total: %d configurations entered the fast path, each checked over all 2^32 floats\n", checked);
    printf("       %d gated off (integer path, not exercised here)\n", gated);
    printf("       %s\n", failed ? "*** MISMATCHES ABOVE ***" : "0 mismatches");
}

static void sweep(int man_step)
{
    std::vector<EB> ebs = eb_grid();
    int checked = 0, gated = 0, failed = 0;

    // Main grid. is_signed is fixed true and subnormals to SUBNORMALS: those
    // two axes are covered by the confirmation set below, because is_signed
    // only guards an early return that precedes the fast path (so the signed
    // sweep already covers every input the unsigned one can reach), and the
    // other two subnormals modes are excluded by the gate's second half.
    printf("main grid -- 2^32 inputs per configuration, three-way\n");
    for (int man = 0; man <= 24; man += man_step)
    {
        int man_checked = 0, man_failed = 0;
        for (const EB &eb : ebs)
            for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE,
                                       SaturationMode::SAT_PROPAGATE})
            {
                int r = run_case(man, eb.exp_bits, eb.bias, true, sat, SubnormalsMode::SUBNORMALS, false);
                if (r < 0) ++gated;
                else { ++checked; ++man_checked; if (r == 0) { ++failed; ++man_failed; } }
            }
        printf("  man_bits=%-2d  %3d configurations verified, %d mismatched\n", man, man_checked, man_failed);
        fflush(stdout);
    }

    // Confirmation set: the other two axes, on formats the gate admits.
    printf("confirmation set -- unsigned, and the subnormals modes the gate excludes\n");
    int conf = 0, conf_failed = 0;
    for (int man : {1, 2, 3, 5, 10, 15, 21, 22})
        for (const EB &eb : {EB{4, 7}, EB{5, 15}, EB{8, 127}, EB{3, 1}, EB{6, 60}, EB{2, -8}})
            for (bool sgn : {true, false})
                for (SubnormalsMode sm : {SubnormalsMode::SUBNORMALS, SubnormalsMode::NORMALS,
                                          SubnormalsMode::EXTENDED_NORMALS})
                {
                    int r = run_case(man, eb.exp_bits, eb.bias, sgn, SaturationMode::SAT_NONE, sm, false);
                    if (r < 0) ++gated;
                    else { ++conf; if (r == 0) ++conf_failed; }
                }
    printf("  %d configurations verified, %d mismatched\n", conf, conf_failed);

    printf("total: %d configurations entered the fast path, each checked over all 2^32 floats\n", checked + conf);
    printf("       %d gated off (integer path, not exercised here)\n", gated);
    printf("       %s\n", (failed + conf_failed) ? "*** MISMATCHES ABOVE ***" : "0 mismatches");
}

// ---------------------------------------------------------------------- GEMM
#if defined(__CUDACC__)
struct Mul { bool is_signed; SubnormalsMode sub; BinaryKParams params;
  __device__ __forceinline__ float operator()(float a,float b) const { return cast_binaryK_nearest_even(a*b,is_signed,sub,params);} };
struct Add { bool is_signed; SubnormalsMode sub; BinaryKParams params;
  __device__ __forceinline__ float operator()(float x) const { return cast_binaryK_nearest_even(x,is_signed,sub,params);} };
struct Split { Mul mul; Add add;
  __device__ __forceinline__ float step(float a,float b,float acc) const { return add(acc+mul(a,b)); } };
struct Acc { Split mac{}; float sum=0.f;
  __device__ __forceinline__ void accumulate(float a,float b){sum=mac.step(a,b,sum);}
  __device__ __forceinline__ float finalize() const {return sum;} };

constexpr int BS=16;
template<int UNROLL>
__global__ __launch_bounds__(BS*BS)
void gemm(const float*__restrict__ A,const float*__restrict__ B,float*__restrict__ C,
          int64_t M,int64_t K,int64_t N,Acc acc_proto)
{
    __shared__ float As[2][BS*BS]; __shared__ float Bs[2][BS*BS];
    const int tc=threadIdx.x%BS, tr=threadIdx.x/BS;
    const int64_t rId=(int64_t)blockIdx.y*BS+tr, cId=(int64_t)blockIdx.x*BS+tc;
    const int64_t nT=(K+BS-1)/BS;
    Acc acc=acc_proto;
    As[0][tr*BS+tc]=(rId<M&&tc<K)?A[rId*K+tc]:0.f;
    Bs[0][tr*BS+tc]=(tr<K&&cId<N)?B[cId*K+tr]:0.f;  // trans_b
    for(int64_t t=0;t<nT;++t){
        __syncthreads();
        float*cA=As[t%2]; float*cB=Bs[t%2];
#pragma unroll UNROLL
        for(int d=0;d<BS;++d) acc.accumulate(cA[tr*BS+d],cB[d*BS+tc]);
        if(t+1<nT){ int64_t nk=(t+1)*BS;
            As[(t+1)%2][tr*BS+tc]=(rId<M&&nk+tc<K)?A[rId*K+nk+tc]:0.f;
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

static void gemm_bench(int man,int expb,int bias)
{
    BinaryKParams pf=make_binaryK_params(man,expb,bias,true,SaturationMode::SAT_NONE,false);
    BinaryKParams ps=pf; ps.fast_rne=false;
    printf("\nGEMM 1024^3, split mac, binaryK man=%d exp=%d bias=%d (fast gate: %s)\n",
           man,expb,bias,pf.fast_rne?"on":"OFF");
    M=K=N=1024; int iters=5;
    std::vector<float> hA(M*K),hB((size_t)N*K); srand(1234);
    for(auto&v:hA)v=(float)rand()/RAND_MAX*2-1; for(auto&v:hB)v=(float)rand()/RAND_MAX*2-1;
    CHECK(cudaMalloc(&dA,4ull*M*K));CHECK(cudaMalloc(&dB,4ull*(size_t)N*K));CHECK(cudaMalloc(&dC,4ull*M*N));
    CHECK(cudaMemcpy(dA,hA.data(),4ull*M*K,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,hB.data(),4ull*(size_t)N*K,cudaMemcpyHostToDevice));
    dim3 g((N+BS-1)/BS,(M+BS-1)/BS), b(BS*BS);
    const SubnormalsMode sm=SubnormalsMode::SUBNORMALS;
    Acc aS{{{true,sm,ps},{true,sm,ps}},0.f};
    Acc aF{{{true,sm,pf},{true,sm,pf}},0.f};
    auto run=[&](const char*name,auto lam){ for(int i=0;i<2;i++)lam(); CHECK(cudaDeviceSynchronize());
        cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
        for(int i=0;i<iters;i++)lam(); cudaEventRecord(e);CHECK(cudaEventSynchronize(e));
        float ms;cudaEventElapsedTime(&ms,s,e);ms/=iters;
        printf("%-40s %9.3f ms  %6.2f Gmac/s\n",name,ms,(double)M*N*K/(ms/1e3)/1e9); chk(name); };
    run("integer cast, unroll 16",[&]{gemm<16><<<g,b>>>(dA,dB,dC,M,K,N,aS);});
    run("integer cast, unroll 4", [&]{gemm<4 ><<<g,b>>>(dA,dB,dC,M,K,N,aS);});
    run("float-arith cast, unroll 16",[&]{gemm<16><<<g,b>>>(dA,dB,dC,M,K,N,aF);});
    run("float-arith cast, unroll 4", [&]{gemm<4 ><<<g,b>>>(dA,dB,dC,M,K,N,aF);});
    run("float-arith cast, unroll 8", [&]{gemm<8 ><<<g,b>>>(dA,dB,dC,M,K,N,aF);});
    run("float-arith cast, unroll 2", [&]{gemm<2 ><<<g,b>>>(dA,dB,dC,M,K,N,aF);});
    run("float-arith cast, unroll 1", [&]{gemm<1 ><<<g,b>>>(dA,dB,dC,M,K,N,aF);});
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
static void gemm_bench(int man,int expb,int bias)
{
    BinaryKParams pf=make_binaryK_params(man,expb,bias,true,SaturationMode::SAT_NONE,false);
    BinaryKParams ps=pf; ps.fast_rne=false;
    const int M=256,K=256,N=256; const SubnormalsMode sm=SubnormalsMode::SUBNORMALS;
    std::vector<float> A((size_t)M*K),B((size_t)K*N),C((size_t)M*N),ref;
    srand(1234);
    for(auto&v:A)v=(float)rand()/RAND_MAX*2-1; for(auto&v:B)v=(float)rand()/RAND_MAX*2-1;
    printf("\nGEMM %d^3, split mac, binaryK man=%d exp=%d bias=%d (fast gate: %s), 1 thread\n",
           M,man,expb,bias,pf.fast_rne?"on":"OFF");
    auto run=[&](const char*name,const BinaryKParams&p){
        auto once=[&]{ for(int i=0;i<M;++i) for(int j=0;j<N;++j){ float acc=0.f;
            for(int k=0;k<K;++k) acc=cast_binaryK_nearest_even(acc+cast_binaryK_nearest_even(A[(size_t)i*K+k]*B[(size_t)k*N+j],true,sm,p),true,sm,p);
            C[(size_t)i*N+j]=acc; } };
        once();
        auto t0=std::chrono::steady_clock::now();
        for(int r=0;r<3;++r) once();
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/3.0;
        printf("%-40s %9.3f ms  %6.2f Gmac/s",name,ms,(double)M*N*K/(ms/1e3)/1e9);
        if(ref.empty()){ref=C;printf("      [reference]\n");}
        else{ size_t bad=0; for(size_t i=0;i<C.size();++i) if(bits_of(ref[i])!=bits_of(C[i])) ++bad;
              if(bad) printf("      [MISMATCH %zu/%zu]\n",bad,C.size()); else printf("      [bit-exact]\n"); } };
    run("integer cast",ps);
    run("float-arith cast",pf);
}
#endif

int main(int argc,char**argv)
{
    std::string what = argc > 1 ? argv[1] : "all";
    if (what == "one") { run_case(3,4,7,true,SaturationMode::SAT_NONE,SubnormalsMode::SUBNORMALS,true); return 0; }
    if (what == "spot") { spot(); return 0; }
    if (what == "all" || what == "sweep") sweep(1);
    if (what == "coarse") { sweep(7); return 0; }
    if (what == "all" || what == "gemm")  gemm_bench(3, 4, 7);
    return 0;
}
