#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUCHK(x) do { CUresult e=(x); if(e!=CUDA_SUCCESS){const char *s=nullptr;cuGetErrorString(e,&s);fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,s?s:"?");exit(1);} } while(0)
#define OCHK(x) do { OptixResult e=(x); if(e!=OPTIX_SUCCESS){fprintf(stderr,"OptiX %s:%d: %d\n",__FILE__,__LINE__,(int)e);exit(1);} } while(0)
static void cb(unsigned level,const char *tag,const char *msg,void*) { if(level<=2) fprintf(stderr,"optix[%s] %s\n",tag,msg); }
static float median(std::vector<float> v){std::sort(v.begin(),v.end()); return v[v.size()/2];}

static void run(OptixDeviceContext oc, CUstream stream, CUdeviceptr verts,
                unsigned ntri, unsigned flags, bool update, const char *name) {
  CUdeviceptr vp=verts;
  unsigned geom_flags=OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  OptixBuildInput in={}; in.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  in.triangleArray.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;
  in.triangleArray.vertexStrideInBytes=12;
  in.triangleArray.numVertices=ntri*3;
  in.triangleArray.vertexBuffers=&vp;
  in.triangleArray.indexFormat=OPTIX_INDICES_FORMAT_NONE;
  in.triangleArray.numIndexTriplets=0;
  in.triangleArray.indexBuffer=0;
  in.triangleArray.flags=&geom_flags;
  in.triangleArray.numSbtRecords=1;
  OptixAccelBuildOptions bo={}; bo.buildFlags=flags; bo.operation=OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sz={}; OCHK(optixAccelComputeMemoryUsage(oc,&bo,&in,1,&sz));
  size_t tmpbytes=std::max(sz.tempSizeInBytes,sz.tempUpdateSizeInBytes);
  CUdeviceptr tmp=0,out=0; CUCHK(cuMemAlloc(&tmp,tmpbytes)); CUCHK(cuMemAlloc(&out,sz.outputSizeInBytes));
  OptixTraversableHandle handle=0;
  OCHK(optixAccelBuild(oc,stream,&bo,&in,1,tmp,tmpbytes,out,sz.outputSizeInBytes,&handle,nullptr,0)); CUCHK(cuStreamSynchronize(stream));
  if(update) bo.operation=OPTIX_BUILD_OPERATION_UPDATE;
  CUevent a,b; CUCHK(cuEventCreate(&a,CU_EVENT_DEFAULT)); CUCHK(cuEventCreate(&b,CU_EVENT_DEFAULT));
  std::vector<float> ms;
  for(int i=0;i<23;i++){
    CUCHK(cuEventRecord(a,stream));
    OCHK(optixAccelBuild(oc,stream,&bo,&in,1,tmp,update?sz.tempUpdateSizeInBytes:sz.tempSizeInBytes,out,sz.outputSizeInBytes,&handle,nullptr,0));
    CUCHK(cuEventRecord(b,stream)); CUCHK(cuEventSynchronize(b)); float x; CUCHK(cuEventElapsedTime(&x,a,b)); if(i>=3) ms.push_back(x);
  }
  float sum=0; for(float x:ms)sum+=x; printf("%s triangles=%u median_ms=%.6f mean_ms=%.6f min_ms=%.6f output_MiB=%.2f temp_MiB=%.2f\n",name,ntri,median(ms),sum/ms.size(),*std::min_element(ms.begin(),ms.end()),sz.outputSizeInBytes/1048576.,(update?sz.tempUpdateSizeInBytes:sz.tempSizeInBytes)/1048576.);
  cuEventDestroy(a);cuEventDestroy(b);cuMemFree(tmp);cuMemFree(out);
}
static void run_aabb(OptixDeviceContext oc, CUstream stream, CUdeviceptr aabbs,
                unsigned n, unsigned flags, bool update, const char *name) {
  CUdeviceptr ap=aabbs; unsigned geom_flags=OPTIX_GEOMETRY_FLAG_NONE;
  OptixBuildInput in={}; in.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  in.customPrimitiveArray.aabbBuffers=&ap; in.customPrimitiveArray.numPrimitives=n;
  in.customPrimitiveArray.strideInBytes=24; in.customPrimitiveArray.flags=&geom_flags;
  in.customPrimitiveArray.numSbtRecords=1;
  OptixAccelBuildOptions bo={}; bo.buildFlags=flags; bo.operation=OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes sz={}; OCHK(optixAccelComputeMemoryUsage(oc,&bo,&in,1,&sz));
  size_t tmpbytes=std::max(sz.tempSizeInBytes,sz.tempUpdateSizeInBytes);
  CUdeviceptr tmp=0,out=0; CUCHK(cuMemAlloc(&tmp,tmpbytes)); CUCHK(cuMemAlloc(&out,sz.outputSizeInBytes)); OptixTraversableHandle handle=0;
  OCHK(optixAccelBuild(oc,stream,&bo,&in,1,tmp,tmpbytes,out,sz.outputSizeInBytes,&handle,nullptr,0)); CUCHK(cuStreamSynchronize(stream)); if(update)bo.operation=OPTIX_BUILD_OPERATION_UPDATE;
  CUevent a,b;CUCHK(cuEventCreate(&a,CU_EVENT_DEFAULT));CUCHK(cuEventCreate(&b,CU_EVENT_DEFAULT));std::vector<float> ms;
  for(int i=0;i<23;i++){CUCHK(cuEventRecord(a,stream));OCHK(optixAccelBuild(oc,stream,&bo,&in,1,tmp,update?sz.tempUpdateSizeInBytes:sz.tempSizeInBytes,out,sz.outputSizeInBytes,&handle,nullptr,0));CUCHK(cuEventRecord(b,stream));CUCHK(cuEventSynchronize(b));float x;CUCHK(cuEventElapsedTime(&x,a,b));if(i>=3)ms.push_back(x);}
  float sum=0;for(float x:ms)sum+=x;printf("%s primitives=%u median_ms=%.6f mean_ms=%.6f min_ms=%.6f output_MiB=%.2f temp_MiB=%.2f\n",name,n,median(ms),sum/ms.size(),*std::min_element(ms.begin(),ms.end()),sz.outputSizeInBytes/1048576.,(update?sz.tempUpdateSizeInBytes:sz.tempSizeInBytes)/1048576.);
  cuEventDestroy(a);cuEventDestroy(b);cuMemFree(tmp);cuMemFree(out);
}

int main(int argc,char**argv){ unsigned ntri=argc>1?strtoul(argv[1],0,0):354000;
 CUCHK(cuInit(0)); CUdevice dev; CUCHK(cuDeviceGet(&dev,0)); CUcontext ctx; CUCHK(cuCtxCreate(&ctx,0,dev));
 OCHK(optixInit()); OptixDeviceContextOptions oo={};oo.logCallbackFunction=cb;oo.logCallbackLevel=2;OptixDeviceContext oc;OCHK(optixDeviceContextCreate(ctx,&oo,&oc));
 std::vector<float> h((size_t)ntri*9); unsigned rng=1; auto rnd=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return (rng&0xffffff)*(1.f/16777216.f);}; bool random=argc>2;
 for(unsigned i=0;i<ntri;i++){if(random){for(int j=0;j<9;j++)h[i*9+j]=rnd();}else{float x=(i%1024)/1024.f,y=(i/1024)/1024.f,z=(i%997)/997.f; h[i*9]=x;h[i*9+1]=y;h[i*9+2]=z;h[i*9+3]=x+0.0004f;h[i*9+4]=y;h[i*9+5]=z;h[i*9+6]=x;h[i*9+7]=y+0.0004f;h[i*9+8]=z;}}
 CUdeviceptr verts;CUCHK(cuMemAlloc(&verts,h.size()*sizeof(float)));CUCHK(cuMemcpyHtoD(verts,h.data(),h.size()*sizeof(float)));CUstream s;CUCHK(cuStreamCreate(&s,CU_STREAM_NON_BLOCKING));
 run(oc,s,verts,ntri,OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,false,"fast-build");
 run(oc,s,verts,ntri,OPTIX_BUILD_FLAG_ALLOW_UPDATE|OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,true,"fast-update-refit");
 run(oc,s,verts,ntri,OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,false,"fast-trace-build");
 std::vector<float> ah((size_t)ntri*6); for(unsigned i=0;i<ntri;i++){float *v=&h[i*9],*a=&ah[i*6];a[0]=std::min(v[0],std::min(v[3],v[6]));a[1]=std::min(v[1],std::min(v[4],v[7]));a[2]=std::min(v[2],std::min(v[5],v[8]));a[3]=std::max(v[0],std::max(v[3],v[6]));a[4]=std::max(v[1],std::max(v[4],v[7]));a[5]=std::max(v[2],std::max(v[5],v[8]));}
 CUdeviceptr aabbs;CUCHK(cuMemAlloc(&aabbs,ah.size()*sizeof(float)));CUCHK(cuMemcpyHtoD(aabbs,ah.data(),ah.size()*sizeof(float)));
 run_aabb(oc,s,aabbs,ntri,OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,false,"aabb-fast-build");
 run_aabb(oc,s,aabbs,ntri,OPTIX_BUILD_FLAG_ALLOW_UPDATE|OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,true,"aabb-update-refit"); cuMemFree(aabbs);
 optixDeviceContextDestroy(oc);cuStreamDestroy(s);cuMemFree(verts);cuCtxDestroy(ctx); }
