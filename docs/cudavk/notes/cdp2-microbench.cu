
// CDP2 device-side chain launch latency vs host-issued chain, sm_120 / CUDA 12.8.
// Arms:
//   host  : N empty kernels launched back-to-back by the host on one stream
//   tail  : host launches 1 kernel; each kernel tail-launches its successor
//           (cudaStreamTailLaunch) -- the serial-chain shape of idea F
//   fnf   : same but cudaStreamFireAndForget from the last thread
// Per-link price = wall(chain)/N, minus nothing -- these are the same units
// the project's 0.78-0.81us remove / ~2.0us exposed host prices are quoted in.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void empty_k() {}

__global__ void chain_tail(int depth) {
    if (threadIdx.x == 0 && depth > 0)
        chain_tail<<<1, 32, 0, cudaStreamTailLaunch>>>(depth - 1);
}

__global__ void chain_fnf(int depth) {
    if (threadIdx.x == 0 && depth > 0)
        chain_fnf<<<1, 32, 0, cudaStreamFireAndForget>>>(depth - 1);
}

#define CK(x) do { cudaError_t e=(x); if(e){printf("ERR %s %d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

int main() {
    const int N = 512, REPS = 20;
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    // warmup
    for (int i = 0; i < 64; i++) empty_k<<<1, 32, 0, s>>>();
    chain_tail<<<1, 32, 0, s>>>(64);
    chain_fnf<<<1, 32, 0, s>>>(64);
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());

    float best_host = 1e9f, best_tail = 1e9f, best_fnf = 1e9f;
    float sum_host = 0, sum_tail = 0, sum_fnf = 0;
    for (int r = 0; r < REPS; r++) {
        float ms;
        CK(cudaEventRecord(a, s));
        for (int i = 0; i < N; i++) empty_k<<<1, 32, 0, s>>>();
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best_host) best_host = ms; sum_host += ms;

        CK(cudaEventRecord(a, s));
        chain_tail<<<1, 32, 0, s>>>(N - 1);
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best_tail) best_tail = ms; sum_tail += ms;

        CK(cudaEventRecord(a, s));
        chain_fnf<<<1, 32, 0, s>>>(N - 1);
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best_fnf) best_fnf = ms; sum_fnf += ms;
    }
    // host API cost alone (issue price, device saturated so calls don't wait)
    // measure CPU time of cuLaunchKernel-equivalent while a long chain runs
    printf("chain N=%d reps=%d (per-link us: best / mean)\n", N, REPS);
    printf("host-issued : %.3f / %.3f\n", best_host * 1e3f / N, sum_host / REPS * 1e3f / N);
    printf("tail-launch : %.3f / %.3f\n", best_tail * 1e3f / N, sum_tail / REPS * 1e3f / N);
    printf("fire-forget : %.3f / %.3f\n", best_fnf * 1e3f / N, sum_fnf / REPS * 1e3f / N);

    // host launch API CPU cost (wall clock across N launches, stream busy)
    struct timespec t0, t1;
    chain_tail<<<1, 32, 0, s>>>(2000); // keep device busy
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) empty_k<<<1, 32, 0, s>>>();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    CK(cudaStreamSynchronize(s));
    double api_us = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1e3 / N;
    printf("host cudaLaunchKernel API CPU cost, stream busy: %.3f us/launch\n", api_us);
    return 0;
}
