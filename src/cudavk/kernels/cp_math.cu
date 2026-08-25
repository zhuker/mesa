/* Exact range-reduced math helpers for helper-isolated hardware shaders. */
extern "C" __device__ float cp_sinf(float x) { return sinf(x); }
extern "C" __device__ float cp_cosf(float x) { return cosf(x); }
