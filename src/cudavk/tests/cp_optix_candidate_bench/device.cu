#include <optix.h>
#include "common.h"
extern "C" { __constant__ Params params; }

static __forceinline__ __device__ float3 v3(Vertex v) { return make_float3(v.x,v.y,v.z); }
static __forceinline__ __device__ float3 sub(float3 a,float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
static __forceinline__ __device__ float3 cross3(float3 a,float3 b) { return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
static __forceinline__ __device__ float dot3(float3 a,float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }

extern "C" __global__ void __raygen__rg() {
    uint3 i=optixGetLaunchIndex();
    float3 o=make_float3((i.x+0.5f)/params.width,(i.y+0.5f)/params.height,1.0f);
    float3 d=make_float3(0.0f,0.0f,-1.0f);
    unsigned payload=0;
    optixTrace(params.handle,o,d,0.0f,2.0f,0.0f,OptixVisibilityMask(255),OPTIX_RAY_FLAG_NONE,0,1,0,payload);
    params.output[i.y*params.width+i.x]=payload;
}

extern "C" __global__ void __intersection__tri() {
    unsigned p=optixGetPrimitiveIndex();
    float3 a=v3(params.vertices[3*p+0]);
    float3 b=v3(params.vertices[3*p+1]);
    float3 c=v3(params.vertices[3*p+2]);
    float3 o=optixGetObjectRayOrigin();
    float3 d=optixGetObjectRayDirection();
    // Two-sided Moller-Trumbore. These three bounds are the exact triangle
    // edge tests; an AABB hit alone is never reported as a triangle hit.
    float3 e1=sub(b,a), e2=sub(c,a), q=cross3(d,e2);
    float det=dot3(e1,q);
    if (fabsf(det)<1.0e-12f) return;
    float inv=1.0f/det;
    float3 s=sub(o,a);
    float u=dot3(s,q)*inv;
    if (u<0.0f || u>1.0f) return;
    float3 r=cross3(s,e1);
    float v=dot3(d,r)*inv;
    if (v<0.0f || u+v>1.0f) return;
    float t=dot3(e2,r)*inv;
    if (t>=optixGetRayTmin() && t<=optixGetRayTmax()) optixReportIntersection(t,0,u,v);
}
extern "C" __global__ void __closesthit__ch() { optixSetPayload_0(optixGetPrimitiveIndex()+1); }
extern "C" __global__ void __miss__ms() { optixSetPayload_0(0); }
