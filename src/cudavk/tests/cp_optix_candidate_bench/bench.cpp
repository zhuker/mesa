#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "common.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUCHK(x) do { cudaError_t _err=(x); if(_err!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_err)); exit(1);} } while(0)
#define DCHK(x) do { CUresult _err=(x); if(_err!=CUDA_SUCCESS){fprintf(stderr,"CUDA driver %s:%d: %d\n",__FILE__,__LINE__,(int)_err); exit(1);} } while(0)
#define OCHK(x) do { OptixResult _err=(x); if(_err!=OPTIX_SUCCESS){fprintf(stderr,"OptiX %s:%d: %d\n",__FILE__,__LINE__,(int)_err); exit(1);} } while(0)
static void logcb(unsigned level,const char* tag,const char* msg,void*) { if(level<=2) fprintf(stderr,"[%s] %s\n",tag,msg); }
static std::string readall(const char* p) { std::ifstream f(p,std::ios::binary); if(!f) { perror(p); exit(1); } return {(std::istreambuf_iterator<char>(f)),{}}; }
template<class T> struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Rec { char hdr[OPTIX_SBT_RECORD_HEADER_SIZE]; T data; };
struct Empty{};
struct Episode { Vertex* v{}; OptixAabb* a{}; void* gas{}; size_t gasBytes{}; OptixTraversableHandle handle{}; };
static void* dalloc(size_t n) { CUdeviceptr p=0; DCHK(cuMemAlloc(&p,n)); return (void*)p; }
static float elapsed(cudaEvent_t a,cudaEvent_t b) { float x; CUCHK(cudaEventElapsedTime(&x,a,b)); return x; }

int main(int argc,char**argv) {
  const char* ir=argc>1?argv[1]:"device.optixir";
  int traceIters=argc>2?atoi(argv[2]):5;
  const unsigned E=argc>3?(unsigned)atoi(argv[3]):16;
  const unsigned scale=E==1?4:E==4?2:E==16?1:0;
  if(!scale){fprintf(stderr,"episodes must be 1, 4, or 16\n");return 2;}
  constexpr unsigned W=1280,H=720;
  const unsigned NX=177*scale,NY=125*scale,N=NX*NY; // always 354,000 total
  CUCHK(cudaFree(0)); OCHK(optixInit());
  CUcontext cuCtx=0; if(cuCtxGetCurrent(&cuCtx)!=CUDA_SUCCESS || !cuCtx){fprintf(stderr,"no CUDA current context\n"); return 1;} OptixDeviceContextOptions co{}; co.logCallbackFunction=logcb; co.logCallbackLevel=2;
  OptixDeviceContext ctx; OCHK(optixDeviceContextCreate(cuCtx,&co,&ctx));
  std::string code=readall(ir); char log[8192]; size_t logs=sizeof(log);
  OptixModuleCompileOptions mo{}; mo.optLevel=OPTIX_COMPILE_OPTIMIZATION_LEVEL_3; mo.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  OptixPipelineCompileOptions po{}; po.usesMotionBlur=0; po.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  po.numPayloadValues=1; po.numAttributeValues=2; po.exceptionFlags=OPTIX_EXCEPTION_FLAG_NONE;
  po.pipelineLaunchParamsVariableName="params"; po.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
  OptixModule mod; OCHK(optixModuleCreate(ctx,&mo,&po,code.data(),code.size(),log,&logs,&mod));
  OptixProgramGroupOptions pgo{}; OptixProgramGroupDesc d[3]{};
  d[0].kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d[0].raygen.module=mod; d[0].raygen.entryFunctionName="__raygen__rg";
  d[1].kind=OPTIX_PROGRAM_GROUP_KIND_MISS; d[1].miss.module=mod; d[1].miss.entryFunctionName="__miss__ms";
  d[2].kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d[2].hitgroup.moduleIS=mod; d[2].hitgroup.entryFunctionNameIS="__intersection__tri"; d[2].hitgroup.moduleCH=mod; d[2].hitgroup.entryFunctionNameCH="__closesthit__ch";
  OptixProgramGroup pg[3]; logs=sizeof(log); OCHK(optixProgramGroupCreate(ctx,d,3,&pgo,log,&logs,pg));
  OptixPipelineLinkOptions plo{}; plo.maxTraceDepth=1; OptixPipeline pipe; logs=sizeof(log); OCHK(optixPipelineCreate(ctx,&po,&plo,pg,3,log,&logs,&pipe));
  OCHK(optixPipelineSetStackSize(pipe,2048,2048,2048,1));
  Rec<Empty> hr{},hm{},hh{}; OCHK(optixSbtRecordPackHeader(pg[0],&hr)); OCHK(optixSbtRecordPackHeader(pg[1],&hm)); OCHK(optixSbtRecordPackHeader(pg[2],&hh));
  Rec<Empty> *dr=(Rec<Empty>*)dalloc(sizeof(hr)),*dm=(Rec<Empty>*)dalloc(sizeof(hm)),*dh=(Rec<Empty>*)dalloc(sizeof(hh));
  CUCHK(cudaMemcpy(dr,&hr,sizeof(hr),cudaMemcpyHostToDevice)); CUCHK(cudaMemcpy(dm,&hm,sizeof(hm),cudaMemcpyHostToDevice)); CUCHK(cudaMemcpy(dh,&hh,sizeof(hh),cudaMemcpyHostToDevice));
  OptixShaderBindingTable sbt{}; sbt.raygenRecord=(CUdeviceptr)dr; sbt.missRecordBase=(CUdeviceptr)dm; sbt.missRecordStrideInBytes=sizeof(hm); sbt.missRecordCount=1; sbt.hitgroupRecordBase=(CUdeviceptr)dh; sbt.hitgroupRecordStrideInBytes=sizeof(hh); sbt.hitgroupRecordCount=1;

  std::vector<Episode> ep(E); std::vector<void*> buildTemps,refitTemps; std::vector<Vertex> hv(3*N); std::vector<OptixAabb> ha(N);
  uint32_t geomFlag=OPTIX_GEOMETRY_FLAG_NONE; cudaStream_t stream; CUCHK(cudaStreamCreate(&stream));
  OptixAccelBuildOptions abo{}; abo.motionOptions.numKeys=0; abo.buildFlags=OPTIX_BUILD_FLAG_ALLOW_UPDATE|OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; abo.operation=OPTIX_BUILD_OPERATION_BUILD;
  cudaEvent_t t0,t1; CUCHK(cudaEventCreate(&t0)); CUCHK(cudaEventCreate(&t1));
  CUCHK(cudaEventRecord(t0,stream));
  size_t gasTotal=0;
  for(unsigned e=0;e<E;e++) {
    float z=0.15f+0.04f*e;
    for(unsigned y=0;y<NY;y++) for(unsigned x=0;x<NX;x++) {
      unsigned i=y*NX+x; float cx=(x+0.5f)/NX,cy=(y+0.5f)/NY;
      float sx=0.39f/NX,sy=0.39f/NY;
      // Deterministic sub-cell jitter makes this unlike one giant regular mesh.
      float jx=((i*1664525u+1013904223u)&255u)/255.0f-0.5f;
      cx += jx*0.08f/NX;
      hv[3*i+0]={cx-sx,cy-sy,z}; hv[3*i+1]={cx+sx,cy-sy,z}; hv[3*i+2]={cx,cy+sy,z};
      ha[i]={cx-sx,cy-sy,z-1e-6f,cx+sx,cy+sy,z+1e-6f};
    }
    ep[e].v=(Vertex*)dalloc(hv.size()*sizeof(Vertex)); ep[e].a=(OptixAabb*)dalloc(ha.size()*sizeof(OptixAabb));
    CUCHK(cudaMemcpyAsync(ep[e].v,hv.data(),hv.size()*sizeof(Vertex),cudaMemcpyHostToDevice,stream)); CUCHK(cudaMemcpyAsync(ep[e].a,ha.data(),ha.size()*sizeof(OptixAabb),cudaMemcpyHostToDevice,stream));
    CUdeviceptr ap=(CUdeviceptr)ep[e].a; OptixBuildInput bi{}; bi.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers=&ap; bi.customPrimitiveArray.numPrimitives=N; bi.customPrimitiveArray.strideInBytes=sizeof(OptixAabb); bi.customPrimitiveArray.flags=&geomFlag; bi.customPrimitiveArray.numSbtRecords=1;
    OptixAccelBufferSizes bs{}; OCHK(optixAccelComputeMemoryUsage(ctx,&abo,&bi,1,&bs)); void* tmp=dalloc(bs.tempSizeInBytes); buildTemps.push_back(tmp); ep[e].gas=dalloc(bs.outputSizeInBytes); ep[e].gasBytes=bs.outputSizeInBytes; gasTotal+=bs.outputSizeInBytes;
    OCHK(optixAccelBuild(ctx,stream,&abo,&bi,1,(CUdeviceptr)tmp,bs.tempSizeInBytes,(CUdeviceptr)ep[e].gas,bs.outputSizeInBytes,&ep[e].handle,nullptr,0));
  }
  CUCHK(cudaEventRecord(t1,stream)); CUCHK(cudaEventSynchronize(t1)); float buildMs=elapsed(t0,t1); for(void* p:buildTemps) DCHK(cuMemFree((CUdeviceptr)p));
  // UPDATE the same 354k AABBs. This isolates GAS refit cost from animation generation/copies.
  abo.operation=OPTIX_BUILD_OPERATION_UPDATE; CUCHK(cudaEventRecord(t0,stream));
  for(unsigned e=0;e<E;e++) {
    CUdeviceptr ap=(CUdeviceptr)ep[e].a; OptixBuildInput bi{}; bi.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; bi.customPrimitiveArray.aabbBuffers=&ap; bi.customPrimitiveArray.numPrimitives=N; bi.customPrimitiveArray.strideInBytes=sizeof(OptixAabb); bi.customPrimitiveArray.flags=&geomFlag; bi.customPrimitiveArray.numSbtRecords=1;
    OptixAccelBufferSizes bs{}; OCHK(optixAccelComputeMemoryUsage(ctx,&abo,&bi,1,&bs)); void* tmp=dalloc(bs.tempUpdateSizeInBytes); refitTemps.push_back(tmp);
    OCHK(optixAccelBuild(ctx,stream,&abo,&bi,1,(CUdeviceptr)tmp,bs.tempUpdateSizeInBytes,(CUdeviceptr)ep[e].gas,ep[e].gasBytes,&ep[e].handle,nullptr,0));
  }
  CUCHK(cudaEventRecord(t1,stream)); CUCHK(cudaEventSynchronize(t1)); float refitMs=elapsed(t0,t1); for(void* p:refitTemps) DCHK(cuMemFree((CUdeviceptr)p));
  unsigned* out=(unsigned*)dalloc(size_t(W)*H*sizeof(unsigned)); Params* dp=(Params*)dalloc(sizeof(Params));
  // Warm up all episode launches once, then time trace-only iterations.
  for(unsigned e=0;e<E;e++){ Params p{ep[e].handle,ep[e].v,out,W,H}; CUCHK(cudaMemcpyAsync(dp,&p,sizeof(p),cudaMemcpyHostToDevice,stream)); OCHK(optixLaunch(pipe,stream,(CUdeviceptr)dp,sizeof(p),&sbt,W,H,1)); }
  CUCHK(cudaStreamSynchronize(stream));
  CUCHK(cudaEventRecord(t0,stream));
  for(int it=0;it<traceIters;it++) for(unsigned e=0;e<E;e++){ Params p{ep[e].handle,ep[e].v,out,W,H}; CUCHK(cudaMemcpyAsync(dp,&p,sizeof(p),cudaMemcpyHostToDevice,stream)); OCHK(optixLaunch(pipe,stream,(CUdeviceptr)dp,sizeof(p),&sbt,W,H,1)); }
  CUCHK(cudaEventRecord(t1,stream)); CUCHK(cudaEventSynchronize(t1)); float traceMs=elapsed(t0,t1)/traceIters;
  std::vector<unsigned> hout(size_t(W)*H); CUCHK(cudaMemcpy(hout.data(),out,hout.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
  uint64_t checksum=0,hits=0; for(auto x:hout){checksum+=x;hits+=x!=0;}
  printf("OptiX custom-primitive microbenchmark\n");
  printf("episodes=%u primitives/episode=%u total_primitives=%u rays/episode=%u total_rays=%u\n",E,N,E*N,W*H,E*W*H);
  printf("build_all_ms=%.3f refit_all_ms=%.3f trace_all_episodes_ms=%.3f trace_per_episode_ms=%.3f\n",buildMs,refitMs,traceMs,traceMs/E);
  printf("build_plus_trace_ms=%.3f refit_plus_trace_ms=%.3f\n",buildMs+traceMs,refitMs+traceMs);
  printf("gas_MiB=%.2f trace_iterations=%d last_episode_hits=%llu checksum=%llu\n",gasTotal/1048576.0,traceIters,(unsigned long long)hits,(unsigned long long)checksum);
  return 0;
}
