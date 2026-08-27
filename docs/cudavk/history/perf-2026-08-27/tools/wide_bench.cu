
// wide_bench.cu - shape sweep: does a wide launch issue better than many narrow ones?
// Work per item is FIXED (pix_per_item pixels of edge/depth style math with a
// dependent, L1-resident load chain), so total work is constant across the sweep.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// tex: 32 KiB -> L1 resident (real kernels show 92.5% L1 hit, 1% DRAM)
#define TEX_WORDS (8u*1024u)
#define TEX_MASK  (TEX_WORDS-1u)
#define SETUP_ITEMS 4096u
#define SETUP_MASK  (SETUP_ITEMS-1u)
#define OUT_WORDS (1u<<20)
#define OUT_MASK  (OUT_WORDS-1u)

extern "C" __global__ __launch_bounds__(512)
void raster_like(const float4* __restrict__ setup,
                 const uint32_t* __restrict__ tex,
                 float* __restrict__ out,
                 int pix_per_item, int item_base)
{
    const int item = item_base + blockIdx.x;
    const int t  = threadIdx.x;
    const int nt = blockDim.x;

    // per-item setup: 3 edge equations + depth plane (dependent global loads)
    const uint32_t s = ((uint32_t)item) & SETUP_MASK;
    float4 E0 = setup[s*4u + 0u];
    float4 E1 = setup[s*4u + 1u];
    float4 E2 = setup[s*4u + 2u];
    float4 Z  = setup[s*4u + 3u];

    uint32_t idx = (uint32_t)(item * 7919 + t * 131) & TEX_MASK;
    float acc  = 0.0f;
    float zref = 1.0f;

    for (int p = t; p < pix_per_item; p += nt) {
        float x = (float)(p & 63);
        float y = (float)(p >> 6);
        float e0 = fmaf(E0.x, x, fmaf(E0.y, y, E0.z));
        float e1 = fmaf(E1.x, x, fmaf(E1.y, y, E1.z));
        float e2 = fmaf(E2.x, x, fmaf(E2.y, y, E2.z));
        float z  = fmaf(Z.x,  x, fmaf(Z.y,  y, Z.z));
        uint32_t v = tex[idx];                     // L1-resident load
        idx = (v * 1664525u + (uint32_t)p) & TEX_MASK;   // dependent chain
        float cov = fminf(fminf(e0, e1), e2);
        if (cov >= 0.0f && z < zref) {
            acc  = fmaf(z, __uint2float_rn(v & 255u), acc);
            zref = z;
        }
    }
    out[((uint32_t)(item * nt + t)) & OUT_MASK] = acc;
}

extern "C" __global__ void empty_kernel(float* o){ if(threadIdx.x==1u<<30) o[0]=1.f; }

int main(int argc, char** argv)
{
    // argv: mode T P total_items pix reps
    const char* mode = argc>1 ? argv[1] : "timing";
    int T     = argc>2 ? atoi(argv[2]) : 64;
    int P     = argc>3 ? atoi(argv[3]) : 12;
    long TOT  = argc>4 ? atol(argv[4]) : 60000;
    int PIX   = argc>5 ? atoi(argv[5]) : 4096;
    int REPS  = argc>6 ? atoi(argv[6]) : 3;

    float4*   d_setup; uint32_t* d_tex; float* d_out;
    CK(cudaMalloc(&d_setup, SETUP_ITEMS*4*sizeof(float4)));
    CK(cudaMalloc(&d_tex,   TEX_WORDS*sizeof(uint32_t)));
    CK(cudaMalloc(&d_out,   OUT_WORDS*sizeof(float)));
    {
        float4* h = (float4*)malloc(SETUP_ITEMS*4*sizeof(float4));
        for (uint32_t i=0;i<SETUP_ITEMS*4;i++){
            h[i] = make_float4((float)((i*37)%17)/16.f-0.5f,
                               (float)((i*53)%13)/12.f-0.5f,
                               (float)((i*29)%31)/30.f-0.2f, 0.f);
        }
        CK(cudaMemcpy(d_setup,h,SETUP_ITEMS*4*sizeof(float4),cudaMemcpyHostToDevice));
        free(h);
        uint32_t* ht=(uint32_t*)malloc(TEX_WORDS*sizeof(uint32_t));
        for(uint32_t i=0;i<TEX_WORDS;i++) ht[i]= i*2654435761u;
        CK(cudaMemcpy(d_tex,ht,TEX_WORDS*sizeof(uint32_t),cudaMemcpyHostToDevice));
        free(ht);
        CK(cudaMemset(d_out,0,OUT_WORDS*sizeof(float)));
    }

    if (!strcmp(mode,"ncu")) {
        // warm up 3 launches of this exact shape, then ONE measured launch
        for (int i=0;i<3;i++) raster_like<<<P,T>>>(d_setup,d_tex,d_out,PIX,0);
        CK(cudaDeviceSynchronize());
        raster_like<<<P,T>>>(d_setup,d_tex,d_out,PIX,0);
        CK(cudaDeviceSynchronize());
        printf("ncu shape T=%d P=%d PIX=%d\n",T,P,PIX);
        return 0;
    }
    if (!strcmp(mode,"launchcost")) {
        // back-to-back empty launches, to price the launch itself
        cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for(int i=0;i<1000;i++) empty_kernel<<<P,T>>>(d_out);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for(long i=0;i<TOT;i++) empty_kernel<<<P,T>>>(d_out);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b));
        printf("LAUNCHCOST T=%d P=%d n=%ld total_ms=%.3f per_launch_us=%.3f\n",
               T,P,TOT,ms,ms*1000.f/(float)TOT);
        return 0;
    }

    // timing mode: TOT items total, issued P per launch -> TOT/P launches
    long NL = TOT / P;
    // warmup: same total work, wide
    raster_like<<<(int)(TOT>10000?10000:TOT),T>>>(d_setup,d_tex,d_out,PIX,0);
    CK(cudaDeviceSynchronize());

    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double best = 1e30, sum = 0;
    for (int r=0;r<REPS;r++){
        CK(cudaEventRecord(a));
        for (long l=0;l<NL;l++)
            raster_like<<<P,T>>>(d_setup,d_tex,d_out,PIX,(int)(l*P));
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b));
        if (ms<best) best=ms; sum+=ms;
    }
    printf("TIMING T=%d P=%d launches=%ld items=%ld pix=%d best_ms=%.4f mean_ms=%.4f "
           "us_per_launch=%.4f ns_per_item=%.2f\n",
           T,P,NL,TOT,PIX,best,sum/REPS,best*1000.0/(double)NL,best*1e6/(double)TOT);
    return 0;
}
