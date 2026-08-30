/*
 * The device a renderer runs on.
 *
 * A CUDA context, the compiled kernels and the compute capability:
 * everything the draw pipeline needs of the device it runs on.
 */

#ifndef CP_DEVICE_H
#define CP_DEVICE_H

#include <cuda.h>
#include <stdint.h>
#include <stdatomic.h>

#include "cp_kernels.h"

enum cp_texture_cache_purge_result {
   CP_TEXTURE_CACHE_PURGE_NONE,
   CP_TEXTURE_CACHE_PURGE_RECLAIMED,
   CP_TEXTURE_CACHE_PURGE_FATAL,
};

enum cp_texture_cache_result {
   CP_TEXTURE_CACHE_READY,
   CP_TEXTURE_CACHE_SOFT_FALLBACK,
   CP_TEXTURE_CACHE_FATAL,
};

struct cp_device {
   CUdevice cuda_device;
   CUcontext cuda_ctx;
   int sm_major;
   int sm_minor;

   struct cp_kernels kernels;

   /* Optional native-front-end bridge. Gallium leaves these null. The
    * renderer passes only immutable descriptor cookies and its exact ordered
    * stream; CUDA-array ownership stays in the native Vulkan driver. */
   void *texture_cache_private;
   atomic_uint_fast64_t next_texture_stream_serial;
   /* View cookies are small indices, so their top bit carries a per-site
    * fact the resolver needs: the sampling shader reads only .r, which is
    * what makes a one-channel (depth) texture safe to serve from the
    * hardware path -- its .gba fill differs from Vulkan's expansion, and a
    * site that never reads those lanes cannot tell. */
#define CP_TEXTURE_COOKIE_R_ONLY (1ull << 63)
   enum cp_texture_cache_result
      (*texture_cache_resolve)(void *private_data,
                               uint64_t view_cookie,
                               uint64_t sampler_cookie,
                               CUstream stream, uint64_t stream_serial,
                               CUtexObject *object);
   enum cp_texture_cache_result
      (*texture_cache_resolve_batch)(void *private_data,
                                     const uint64_t *view_cookies,
                                     const uint64_t *sampler_cookies,
                                     size_t count, CUstream stream,
                                     uint64_t stream_serial,
                                     CUtexObject *objects);
   void (*texture_cache_written)(void *private_data, uint64_t image_cookie,
                                 CUstream stream);
   void (*texture_cache_fatal)(void *private_data);
   enum cp_texture_cache_purge_result
      (*texture_cache_purge)(void *private_data);
   void (*texture_cache_use_begin)(void *private_data);
   void (*texture_cache_use_end)(void *private_data);

};

#endif /* CP_DEVICE_H */
