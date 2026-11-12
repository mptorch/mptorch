// The float-arithmetic RNE fast path in cast_superfp.h -- the superfp twin of
// gemm_cast_float_arith.cu, and structured the same way:
//
//  (a) exhaustive bit-exactness check over all 2^32 float inputs, for every
//      (man_bits, exp_bits, normal_binades, bias, is_signed, saturation) in a
//      sweep -- three-way, so a disagreement is attributable:
//        original   cast_superfp_nearest_even(x, man, exp, nb, bias, ...)
//        integer    the SuperfpParams overload with the gate forced off
//        fast       the SuperfpParams overload as shipped
//      The gate itself is the shipped one (SuperfpParams::fast_rne); this
//      program only reports whether it is honest.
//  (b) GEMM timing of the two cast paths in the shipped kernel's shape.
//
// Build & run (from the repo root):
//   nvcc -O3 -std=c++17 -arch=sm_89 --expt-relaxed-constexpr -I mptorch/csrc \
//        dev/benchmarks/sweeps/gemm_cast_superfp_arith.cu -o /tmp/gemm_cast_superfp_arith \
//     && /tmp/gemm_cast_superfp_arith
// Pass `spot`, `sweep`, `coarse`, `gemm`, `one` or nothing (= sweep + gemm).
//
// Since a0407f9 (setup.py's -ffp-contract=off and -DMPTORCH_FAST_CAST=1) it
// also builds as plain C++, to ask the host compiler the question nvcc used to
// be the only one asked -- whether the Veltkamp split's operands survive as
// separately rounded binary32 values:
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc -x c++ dev/benchmarks/sweeps/gemm_cast_superfp_arith.cu \
//       -o /tmp/gemm_cast_superfp_arith_host && /tmp/gemm_cast_superfp_arith_host spot
// `gemm` is CUDA-only. `spot` is the host-sized mode and the right one there:
// see gemm_cast_float_arith.cu's header for why the format grid answers a
// question the GPU has already settled, and a handful of configurations over
// all 2^32 inputs answers the one the host adds.
//
// cast_superfp_rne_fast has two spellings, the device's and the host's
// (cast_superfp.h, at the function; the host's since f9a6aa9), so the device
// sweep and the host sweep check different code, each the code its backend
// ships: run `coarse` on the GPU for the device's and `spot` and `coarse` on
// the host for the host's. The host arm calls cast_superfp_rne_fast directly,
// as it did when the host shipped superfp's integer path (from a0407f9 to
// f9a6aa9 the host build left this fast path to the device); since f9a6aa9 the
// shipped entry point reaches the same function.
//
// Do not decide a spelling with `gemm`'s timings. The host half is a plain
// loop, and on a host this cast's cost moves with the code around it by more
// than the differences in question: adding one benchmark loop to a scratch
// copy of this program took the device spelling's host timing from 0.77x to
// 1.16x of the integer path without touching either. The shipped kernels,
// two builds alternated, are the measurement.
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
struct Counts { unsigned long long fast_vs_int, fast_vs_orig, int_vs_orig; unsigned first_bad; };

CUDA_HOST_DEVICE_INLINE uint32_t bits_of(float f) { return FLOAT_TO_BITS(&f); }

#if defined(__CUDACC__)
constexpr int PER_THREAD = 64;

__global__ void exhaustive(int man, int expb, int nb, int bias, bool is_signed,
                           SaturationMode sat,
                           SuperfpParams pfast, SuperfpParams pslow, Counts *c)
{
    uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long nfi = 0, nfo = 0, nio = 0;
    uint32_t first = 0;
    for (int k = 0; k < PER_THREAD; ++k)
    {
        uint32_t bits = (uint32_t)(base + k);
        float x = bits_to_float(bits);
        uint32_t o = bits_of(cast_superfp_nearest_even(x, man, expb, nb, bias, is_signed, sat));
        uint32_t i = bits_of(cast_superfp_nearest_even(x, is_signed, pslow));
        uint32_t f = bits_of(cast_superfp_nearest_even(x, is_signed, pfast));
        if (f != i) { if (nfi++ == 0) first = bits; }
        if (f != o) ++nfo;
        if (i != o) ++nio;
    }
    if (nfi) { if (atomicAdd(&c->fast_vs_int, nfi) == 0) c->first_bad = first; }
    if (nfo) atomicAdd(&c->fast_vs_orig, nfo);
    if (nio) atomicAdd(&c->int_vs_orig, nio);
}

static void run_exhaustive(int man, int expb, int nb, int bias, bool is_signed, SaturationMode sat,
                           const SuperfpParams &pfast, const SuperfpParams &pslow, Counts *h)
{
    static Counts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(Counts)));
    CHECK(cudaMemset(dc, 0, sizeof(Counts)));
    exhaustive<<<(unsigned)((1ull << 32) / (256 * PER_THREAD)), 256>>>(man, expb, nb, bias, is_signed, sat, pfast, pslow, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
}
#else
// The host arm: the same patterns over the same three casts, across OpenMP
// threads instead of CUDA blocks, and `first_bad` is the smallest mismatching
// pattern rather than an arbitrary one (a reduction has no "first"). The fast
// arm is cast_superfp_rne_fast itself, preceded by the unsigned early return
// its caller would have done -- see the header comment.
static void run_exhaustive(int man, int expb, int nb, int bias, bool is_signed, SaturationMode sat,
                           const SuperfpParams &pfast, const SuperfpParams &pslow, Counts *h)
{
    unsigned long long nfi = 0, nfo = 0, nio = 0;
    uint32_t first = 0xFFFFFFFFu;
#pragma omp parallel for schedule(static) reduction(+ : nfi, nfo, nio) reduction(min : first)
    for (int64_t hi = 0; hi < (1 << 16); ++hi)
        for (uint32_t lo = 0; lo < (1u << 16); ++lo)
        {
            uint32_t bits = (uint32_t)(((uint64_t)hi << 16) | lo);
            float x = bits_to_float(bits);
            uint32_t o = bits_of(cast_superfp_nearest_even(x, man, expb, nb, bias, is_signed, sat));
            uint32_t i = bits_of(cast_superfp_nearest_even(x, is_signed, pslow));
            uint32_t f = bits_of((x < 0.0f && !is_signed) ? 0.0f : cast_superfp_rne_fast(x, pfast));
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

static const char *satname(SaturationMode s)
{
    return s == SaturationMode::SAT_NONE ? "SAT_NONE"
         : (s == SaturationMode::SAT_FINITE ? "SAT_FINITE" : "SAT_PROPAGATE");
}

static int run_case(int man, int expb, int nb, int bias, bool is_signed, SaturationMode sat,
                    bool verbose)
{
    SuperfpParams pfast = make_superfp_params(man, expb, nb, bias, sat);
    SuperfpParams pslow = pfast;
    pslow.fast_rne = false;
    if (!pfast.fast_rne)
        return -1; // gated off: the fast path is never entered, nothing to verify

    Counts h;
    run_exhaustive(man, expb, nb, bias, is_signed, sat, pfast, pslow, &h);
    bool ok = (h.fast_vs_int == 0 && h.fast_vs_orig == 0 && h.int_vs_orig == 0);
    if (!ok || verbose)
    {
        float xv = *reinterpret_cast<float *>(&h.first_bad);
        printf("  man=%2d exp=%d nb=%2d bias=%4d %s %-13s  fast!=int %llu  fast!=orig %llu  int!=orig %llu",
               man, expb, nb, bias, is_signed ? "signed  " : "unsigned", satname(sat),
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

// normal_binades is superfp's own axis: how many binades at the top keep a
// full significand before the supernormal region takes over. 1 and 2 are the
// values the library uses, so they carry the main grid; the wider set gets
// its own pass below, because the gate has to hold for anything a caller
// passes -- including values that empty the supernormal region out.
static std::vector<int> nb_grid() { return {1, 2}; }
static std::vector<int> nb_wide() { return {1, 2, 3, 4, 5, 8, 12, 16}; }

// The host gate: every region of the fast path -- normal, supernormal,
// underflow -- each over all 2^32 inputs, in a few minutes rather than a
// night. See gemm_cast_float_arith.cu's header for why the format axes do not
// need sweeping to answer the host's question.
static void spot()
{
    int checked = 0, gated = 0, failed = 0;
    auto one = [&](int man, int expb, int nb, int bias, bool sgn, SaturationMode sat) {
        int r = run_case(man, expb, nb, bias, sgn, sat, false);
        if (r < 0) ++gated;
        else { ++checked; if (r == 0) ++failed; }
    };
    printf("spot set -- 2^32 inputs per configuration, three-way\n");
    // normal_binades 1 and 2 are what the library uses; 8 is wide enough to
    // squeeze the supernormal region, so between them every region is entered.
    for (int man : {0, 1, 3, 10})
        for (const EB &eb : {EB{4, 7}, EB{5, 15}, EB{8, 127}})
            for (int nb : {1, 2, 8})
                for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE})
                    one(man, eb.exp_bits, nb, eb.bias, true, sat);
    // the axes that steer past the fast path rather than through it
    one(3, 4, 1, 7, false, SaturationMode::SAT_NONE);
    one(3, 4, 1, 7, true, SaturationMode::SAT_PROPAGATE);
    printf("total: %d configurations entered the fast path, each checked over all 2^32 floats\n", checked);
    printf("       %d gated off (integer path, not exercised here)\n", gated);
    printf("       %s\n", failed ? "*** MISMATCHES ABOVE ***" : "0 mismatches");
}

static void sweep(int man_step)
{
    std::vector<EB> ebs = eb_grid();
    int checked = 0, gated = 0, failed = 0;

    // Main grid. is_signed is fixed true: it only guards an early return that
    // *precedes* the fast path, so the signed sweep already covers every input
    // the unsigned case can reach. The confirmation set below spot-checks it
    // anyway.
    printf("main grid -- 2^32 inputs per configuration, three-way\n");
    for (int man = 0; man <= 24; man += man_step)
    {
        int man_checked = 0, man_failed = 0;
        for (const EB &eb : ebs)
            for (int nb : nb_grid())
                for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE,
                                           SaturationMode::SAT_PROPAGATE})
                {
                    int r = run_case(man, eb.exp_bits, nb, eb.bias, true, sat, false);
                    if (r < 0) ++gated;
                    else { ++checked; ++man_checked; if (r == 0) { ++failed; ++man_failed; } }
                }
        printf("  man_bits=%-2d  %4d configurations verified, %d mismatched\n", man, man_checked, man_failed);
        fflush(stdout);
    }

    // normal_binades pass: the main grid holds it at the two values the
    // library uses, so sweep it wide here against a cross-section of the rest.
    printf("normal_binades pass\n");
    int wide = 0, wide_failed = 0;
    for (int man : {1, 2, 3, 5, 10, 16, 22})
        for (const EB &eb : {EB{4, 7}, EB{5, 15}, EB{8, 127}, EB{3, 1}, EB{6, 60}, EB{2, -8}})
            for (int nb : nb_wide())
                for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE})
                {
                    int r = run_case(man, eb.exp_bits, nb, eb.bias, true, sat, false);
                    if (r < 0) ++gated;
                    else { ++wide; if (r == 0) ++wide_failed; }
                }
    printf("  %d configurations verified, %d mismatched\n", wide, wide_failed);

    printf("confirmation set -- unsigned\n");
    int conf = 0, conf_failed = 0;
    for (int man : {1, 2, 3, 5, 10, 15, 21, 22})
        for (const EB &eb : {EB{4, 7}, EB{5, 15}, EB{8, 127}, EB{3, 1}, EB{6, 60}, EB{2, -8}})
            for (int nb : {1, 2, 4})
            {
                int r = run_case(man, eb.exp_bits, nb, eb.bias, false, SaturationMode::SAT_NONE, false);
                if (r < 0) ++gated;
                else { ++conf; if (r == 0) ++conf_failed; }
            }
    printf("  %d configurations verified, %d mismatched\n", conf, conf_failed);

    printf("total: %d configurations entered the fast path, each checked over all 2^32 floats\n", checked + wide + conf);
    printf("       %d gated off (integer path, not exercised here)\n", gated);
    printf("       %s\n", (failed + wide_failed + conf_failed) ? "*** MISMATCHES ABOVE ***" : "0 mismatches");
}

// ---------------------------------------------------------------------- GEMM
#if defined(__CUDACC__)
struct Mul { bool is_signed; SuperfpParams params;
  __device__ __forceinline__ float operator()(float a,float b) const { return cast_superfp_nearest_even(a*b,is_signed,params);} };
struct Add { bool is_signed; SuperfpParams params;
  __device__ __forceinline__ float operator()(float x) const { return cast_superfp_nearest_even(x,is_signed,params);} };
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
// Compared as words, not as values: at the scales that put the operands in the
// normal region the products saturate, inf - inf is NaN, and NaN != NaN would
// report every arm -- including an arm against itself -- as a mismatch.
static void chk(const char*n){ std::vector<float> h(M*N); CHECK(cudaMemcpy(h.data(),dC,4ull*M*N,cudaMemcpyDeviceToHost));
  if(!ref){ref=(float*)malloc(4ull*M*N);memcpy(ref,h.data(),4ull*M*N);printf("      [%s = reference]\n",n);return;}
  int bad=0; for(int i=0;i<M*N;i++){ uint32_t a,b; memcpy(&a,&ref[i],4); memcpy(&b,&h[i],4); if(a!=b) bad++; }
  printf(bad?"      [%s MISMATCH %d/%d]\n":"      [%s bit-exact]\n",n,bad,M*N); }

// `scale` decides which of superfp's three regions the operands land in, and
// that turns out to matter more than anything else here: with normal_binades
// small, the normal region starts at 2^normal_cutoff and everything below it
// -- which is where ordinary [-1, 1] data lives -- takes the integer path's
// *cheap* branch (a five-operation round to the nearest power of two, no
// clipping). The expensive branch this fast path replaces is only reached by
// operands at or above 2^normal_cutoff.
static void gemm_bench(int man,int expb,int nb,int bias,float scale,const char *regime)
{
    SuperfpParams pf=make_superfp_params(man,expb,nb,bias,SaturationMode::SAT_NONE);
    SuperfpParams ps=pf; ps.fast_rne=false;
    SuperfpCutoffs co=superfp_region_cutoffs(man,expb,nb,bias);
    printf("\nGEMM 1024^3, split mac, superfp man=%d exp=%d nb=%d bias=%d (fast gate: %s)\n",
           man,expb,nb,bias,pf.fast_rne?"on":"OFF");
    printf("  operands ~ +-%g, %s region (normal_cutoff=%d -> normal from %g)\n",
           scale,regime,co.normal_cutoff,(double)ldexpf(1.f,co.normal_cutoff));
    M=K=N=1024; int iters=5;
    std::vector<float> hA(M*K),hB((size_t)N*K); srand(1234);
    for(auto&v:hA)v=((float)rand()/RAND_MAX*2-1)*scale; for(auto&v:hB)v=((float)rand()/RAND_MAX*2-1)*scale;
    CHECK(cudaMalloc(&dA,4ull*M*K));CHECK(cudaMalloc(&dB,4ull*(size_t)N*K));CHECK(cudaMalloc(&dC,4ull*M*N));
    CHECK(cudaMemcpy(dA,hA.data(),4ull*M*K,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,hB.data(),4ull*(size_t)N*K,cudaMemcpyHostToDevice));
    dim3 g((N+BS-1)/BS,(M+BS-1)/BS), b(BS*BS);
    Acc aS{{{true,ps},{true,ps}},0.f};
    Acc aF{{{true,pf},{true,pf}},0.f};
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
// measures it end to end -- and, as the header says, its timings are not a
// reliable model of the cast's cost either. The two runs are compared bit for
// bit, so a timing run is also a (much smaller) correctness check.
static void gemm_bench(int man,int expb,int nb,int bias,float scale,const char*label)
{
    SuperfpParams pf=make_superfp_params(man,expb,nb,bias,SaturationMode::SAT_NONE);
    SuperfpParams ps=pf; ps.fast_rne=false;
    const int M=192,K=192,N=192;
    std::vector<float> A((size_t)M*K),B((size_t)K*N),C((size_t)M*N),ref;
    srand(1234);
    for(auto&v:A)v=((float)rand()/RAND_MAX*2-1)*scale; for(auto&v:B)v=((float)rand()/RAND_MAX*2-1)*scale;
    printf("\nGEMM %d^3, split mac, superfp man=%d exp=%d nb=%d bias=%d scale=%g (%s operands), 1 thread\n",
           M,man,expb,nb,bias,(double)scale,label);
    auto run=[&](const char*name,bool fast){
        const SuperfpParams&p=fast?pf:ps;
        auto cast=[&](float x){ return fast?cast_superfp_rne_fast(x,p):cast_superfp_nearest_even(x,true,p); };
        auto once=[&]{ for(int i=0;i<M;++i) for(int j=0;j<N;++j){ float acc=0.f;
            for(int k=0;k<K;++k) acc=cast(acc+cast(A[(size_t)i*K+k]*B[(size_t)k*N+j]));
            C[(size_t)i*N+j]=acc; } };
        once();
        auto t0=std::chrono::steady_clock::now();
        for(int r=0;r<3;++r) once();
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/3.0;
        printf("%-40s %9.3f ms  %6.2f Gmac/s",name,ms,(double)M*N*K/(ms/1e3)/1e9);
        if(ref.empty()){ref=C;printf("      [reference]\n");}
        else{ size_t bad=0; for(size_t i=0;i<C.size();++i) if(bits_of(ref[i])!=bits_of(C[i])) ++bad;
              if(bad) printf("      [MISMATCH %zu/%zu]\n",bad,C.size()); else printf("      [bit-exact]\n"); } };
    run("integer cast",false);
    run("float-arith cast",true);
}
#endif

int main(int argc,char**argv)
{
    std::string what = argc > 1 ? argv[1] : "all";
    if (what == "one") { run_case(3,4,1,7,true,SaturationMode::SAT_NONE,true); return 0; }
    if (what == "spot") { spot(); return 0; }
    if (what == "coarse") { sweep(7); return 0; }
    if (what == "all" || what == "sweep") sweep(1);
    if (what == "all" || what == "gemm")
    {
        // Which region the operands land in is what decides this cast, so vary
        // it two ways. nb=1 bias=7 puts the normal region at |x| >= 2^8, so
        // [-1,1] operands are entirely supernormal and scaling them by 512
        // moves everything into the normal arm. nb=8 lowers the cutoff to
        // |x| >= 2, which leaves operands and products supernormal but takes
        // the running sum of ~200 of them across into the normal arm -- so
        // that row is a mix, not a third pure case.
        gemm_bench(3, 4, 1, 7, 1.0f, "supernormal");
        gemm_bench(3, 4, 1, 7, 512.0f, "normal");
        gemm_bench(3, 4, 8, 7, 1.0f, "mixed");
    }
    return 0;
}
