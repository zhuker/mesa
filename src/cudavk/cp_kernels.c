#include "cp_kernels.h"
#include "cp_debug.h"
#include "kernels/cp_rast_types.h"
#include "util/u_memory.h"
#include "util/macros.h"
#include "util/blob.h"
#include "util/disk_cache.h"

#include <nvrtc.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Embedded kernel sources, stringified from the .cu files at build time. */
static const char cp_clear_src[] =
#include "cp_clear.cu.inc"
;

static const char cp_rasterize_src[] =
#include "cp_rasterize.cu.inc"
;

static const char cp_fs_src[] =
#include "cp_fs.cu.inc"
;

static const char cp_sampler_src[] =
#include "cp_sampler.cu.inc"
;
static const char cp_math_src[] =
#include "cp_math.cu.inc"
;

static const char cp_vertex_fetch_src[] =
#include "cp_vertex_fetch.cu.inc"
;


/* The physical device owns this cache.  Native exposes one CUDA device per
 * process today; keeping the pointer here also covers lazily compiled sampler
 * variants without threading cache state through every helper. */
static struct disk_cache *cp_nvrtc_cache;

/* The kernels share this header; NVRTC has no filesystem, so hand it over
 * in memory rather than pointing it at an include directory. */
static const char cp_rast_types_src[] =
#include "cp_rast_types.h.inc"
;

static const char cp_fs_interp_src[] =
#include "cp_fs_interp.h.inc"
;

static const char cp_vf_lane_src[] =
#include "cp_vf_lane.h.inc"
;

/*
 * Whether the debug instrumentation is compiled into the kernels at all.
 *
 * This is the fragment census and the machinery that compares the A-buffer
 * against the peel loop — not the A-buffer itself, which is now always built
 * and is selected by launching a different kernel rather than by a branch.
 * What is left here is what genuinely has to sit inside emit_fragment and
 * cp_fs_interpolate, both of which run hundreds of millions of times a frame,
 * so it must not exist unless something is going to read it. NVRTC compiles at
 * run time, so it simply does not, the same way CP_SMALL_THRESHOLD and friends
 * are handed over below.
 */
bool
cp_kernels_instrumented(void)
{
   /* CUDAVK_ABUF_COMPILE=1 compiles the branches in without turning any
    * feature on, which is the only way to measure what they cost; =0 refuses
    * them outright. */
   if (cp_debug->abuf_compile.set)
      return cp_debug->abuf_compile.value;

   /*
    * Otherwise they are compiled in when something needs them. This used to
    * test CUDAVK_ABUFFER_VERIFY for presence while cp_abuf_enabled() tested
    * it for value, so CUDAVK_ABUFFER_VERIFY=0 switched the verification off
    * and still paid to compile the kernels it would have used. Nobody was
    * bitten because everybody sets =1. It now means off in both places.
    */
   return cp_debug->abuffer_verify || cp_debug->frag_census;
}

/*
 * Compile a kernel source with NVRTC.
 *
 * When `relocatable` is set the result is device-relocatable PTX suitable for
 * cuLinkAddData(), which is how the sampler gets linked into shader PTX.
 */
static char *
compile_cuda_source(const char *source, const char *name, int sm_major,
                    int sm_minor, bool relocatable)
{
   const char *header_srcs[] = { cp_rast_types_src, cp_fs_interp_src,
                                 cp_vf_lane_src };
   const char *header_names[] = { "cp_rast_types.h", "cp_fs_interp.h",
                                  "cp_vf_lane.h" };

   nvrtcProgram prog;
   nvrtcResult res = nvrtcCreateProgram(&prog, source, name,
                                        ARRAY_SIZE(header_srcs),
                                        header_srcs, header_names);
   if (res != NVRTC_SUCCESS) {
      fprintf(stderr, "cudavk: nvrtcCreateProgram failed: %d\n", res);
      return NULL;
   }

   char arch_opt[32];
   /* A virtual architecture keeps the output as PTX for the linker to JIT,
    * rather than baking in SASS for one chip. */
   snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d",
            sm_major, sm_minor);

   /*
    * The rasterizer thresholds decide how much of the machine a triangle
    * gets, and the right values are a property of the workload rather than
    * something to be reasoned out once. NVRTC compiles at run time, so they
    * can be swept from the environment without a rebuild, which is what makes
    * a sweep of them repeatable rather than a series of builds nobody can
    * reproduce. Unset means the header's default.
    */
   char small_opt[64], medium_opt[64], point_opt[64], tilebound_opt[64];
   const char *opts[8];
   unsigned num_opts = 0;
   opts[num_opts++] = arch_opt;
   opts[num_opts++] = "--std=c++14";
   if (relocatable)
      opts[num_opts++] = "--relocatable-device-code=true";
   if (cp_kernels_instrumented())
      opts[num_opts++] = "-DCP_ABUF_INSTRUMENT=1";

   if (cp_debug->small_threshold.set) {
      snprintf(small_opt, sizeof(small_opt), "-DCP_SMALL_THRESHOLD=%d",
               cp_debug->small_threshold.value);
      opts[num_opts++] = small_opt;
   }
   if (cp_debug->medium_threshold.set) {
      snprintf(medium_opt, sizeof(medium_opt), "-DCP_MEDIUM_THRESHOLD=%d",
               cp_debug->medium_threshold.value);
      opts[num_opts++] = medium_opt;
   }
   if (cp_debug->point_threshold.set) {
      snprintf(point_opt, sizeof(point_opt), "-DCP_POINT_THRESHOLD=%d",
               cp_debug->point_threshold.value);
      opts[num_opts++] = point_opt;
   }
   /* Not a threshold but the same kind of knob: whether stage 3 walks the
    * whole tile or only the part of it the primitive's bounding box reaches.
    * CUDAVK_TILE_BOUND=0 compiles the full-tile walk back, so the two can be
    * compared without a rebuild. */
   if (cp_debug->tile_bound.set) {
      snprintf(tilebound_opt, sizeof(tilebound_opt), "-DCP_TILE_BOUND=%d",
               cp_debug->tile_bound.value);
      opts[num_opts++] = tilebound_opt;
   }

   assert(num_opts <= ARRAY_SIZE(opts));

   cache_key key;
   bool have_key = false;
   if (cp_nvrtc_cache) {
      struct blob key_blob;
      blob_init(&key_blob);
      blob_write_string(&key_blob, "cudavk NVRTC PTX v1");
      int nvrtc_major = 0, nvrtc_minor = 0;
      nvrtcVersion(&nvrtc_major, &nvrtc_minor);
      blob_write_uint32(&key_blob, (uint32_t)nvrtc_major);
      blob_write_uint32(&key_blob, (uint32_t)nvrtc_minor);
      blob_write_string(&key_blob, name);
      blob_write_string(&key_blob, source);
      blob_write_string(&key_blob, cp_rast_types_src);
      blob_write_string(&key_blob, cp_fs_interp_src);
      blob_write_string(&key_blob, cp_vf_lane_src);
      for (unsigned i = 0; i < num_opts; i++)
         blob_write_string(&key_blob, opts[i]);
      if (!key_blob.out_of_memory) {
         disk_cache_compute_key(cp_nvrtc_cache, key_blob.data, key_blob.size,
                                key);
         have_key = true;
         size_t cached_size = 0;
         char *cached = disk_cache_get(cp_nvrtc_cache, key, &cached_size);
         if (cached && cached_size && cached[cached_size - 1] == '\0') {
            blob_finish(&key_blob);
            nvrtcDestroyProgram(&prog);
            return cached;
         }
         free(cached);
      }
      blob_finish(&key_blob);
   }

   res = nvrtcCompileProgram(prog, num_opts, opts);
   if (res != NVRTC_SUCCESS) {
      size_t log_size;
      nvrtcGetProgramLogSize(prog, &log_size);
      char *log = malloc(log_size);
      nvrtcGetProgramLog(prog, log);
      fprintf(stderr, "cudavk: NVRTC compile failed for %s:\n%s\n", name, log);
      free(log);
      nvrtcDestroyProgram(&prog);
      return NULL;
   }

   size_t ptx_size;
   nvrtcGetPTXSize(prog, &ptx_size);
   char *ptx = malloc(ptx_size);
   nvrtcGetPTX(prog, ptx);
   nvrtcDestroyProgram(&prog);

   if (have_key)
      disk_cache_put(cp_nvrtc_cache, key, ptx, ptx_size, NULL);
   return ptx;
}

static char *
compile_sampler_source(const char *name, int sm_major, int sm_minor,
                       bool enable_3d)
{
   if (!enable_3d)
      return compile_cuda_source(cp_sampler_src, name, sm_major, sm_minor,
                                 true);

   static const char define[] = "#define CP_ENABLE_3D_SAMPLER 1\n";
   size_t source_len = strlen(cp_sampler_src);
   char *source = malloc(sizeof(define) - 1 + source_len + 1);
   if (!source)
      return NULL;
   memcpy(source, define, sizeof(define) - 1);
   memcpy(source + sizeof(define) - 1, cp_sampler_src, source_len + 1);
   char *ptx = compile_cuda_source(source, name, sm_major, sm_minor, true);
   free(source);
   return ptx;
}

char *
cp_compile_sampler_3d(int sm_major, int sm_minor)
{
   return compile_sampler_source("cp_sampler_3d.cu", sm_major, sm_minor,
                                 true);
}

char *
cp_compile_sampler_variant(int sm_major, int sm_minor,
                           const struct cp_sampler_info *info,
                           bool enable_3d)
{
   static_assert(sizeof(float) == sizeof(uint32_t), "32-bit float required");
   uint32_t bits[8];
   memcpy(&bits[0], &info->min_lod, 4);
   memcpy(&bits[1], &info->max_lod, 4);
   memcpy(&bits[2], &info->lod_bias, 4);
   memcpy(&bits[3], &info->max_anisotropy, 4);
   for (unsigned i = 0; i < 4; i++)
      memcpy(&bits[4 + i], &info->border_color[i], 4);

   char defines[2048];
   int len = snprintf(defines, sizeof(defines),
      "%s#define CP_SPECIALIZED_SAMPLER 1\n"
      "#define CP_SPEC_WRAP_S %u\n#define CP_SPEC_WRAP_T %u\n"
      "#define CP_SPEC_WRAP_R %u\n#define CP_SPEC_MIN_IMG %u\n"
      "#define CP_SPEC_MAG_IMG %u\n#define CP_SPEC_MIP %u\n"
      "#define CP_SPEC_UNNORM %u\n"
      "#define CP_SPEC_MIN_LOD (__int_as_float((int)0x%08xU))\n"
      "#define CP_SPEC_MAX_LOD (__int_as_float((int)0x%08xU))\n"
      "#define CP_SPEC_LOD_BIAS (__int_as_float((int)0x%08xU))\n"
      "#define CP_SPEC_MAX_ANISO (__int_as_float((int)0x%08xU))\n"
      "#define CP_SPEC_BORDER_R (__int_as_float((int)0x%08xU))\n"
      "#define CP_SPEC_BORDER_G (__int_as_float((int)0x%08xU))\n"
      "#define CP_SPEC_BORDER_B (__int_as_float((int)0x%08xU))\n"
      "#define CP_SPEC_BORDER_A (__int_as_float((int)0x%08xU))\n",
      enable_3d ? "#define CP_ENABLE_3D_SAMPLER 1\n" : "",
      info->wrap_s, info->wrap_t, info->wrap_r, info->min_img_filter,
      info->mag_img_filter, info->min_mip_filter, info->unnormalized_coords,
      bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7]);
   if (len < 0 || (size_t)len >= sizeof(defines))
      return NULL;

   size_t source_len = strlen(cp_sampler_src);
   char *source = malloc((size_t)len + source_len + 1);
   if (!source)
      return NULL;
   memcpy(source, defines, (size_t)len);
   memcpy(source + len, cp_sampler_src, source_len + 1);
   char *ptx = compile_cuda_source(source, "cp_sampler_variant.cu",
                                   sm_major, sm_minor, true);
   free(source);
   return ptx;
}

/* Compile one kernel source and load it as a module. */
static bool
build_module(CUmodule *out, const char *src, const char *name,
             int sm_major, int sm_minor)
{
   char *ptx = compile_cuda_source(src, name, sm_major,
                                   sm_minor, false);
   if (!ptx) {
      fprintf(stderr, "cudavk: failed to compile %s\n", name);
      return false;
   }

   CUresult err = cuModuleLoadData(out, ptx);
   free(ptx);
   if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cudavk: cuModuleLoadData(%s) failed: %d\n", name, err);
      return false;
   }
   return true;
}

bool
cp_kernels_init(struct cp_kernels *k, int sm_major, int sm_minor,
                struct disk_cache *disk_cache)
{
   memset(k, 0, sizeof(*k));
   cp_nvrtc_cache = disk_cache;

   if (!build_module(&k->clear_module, cp_clear_src, "cp_clear.cu", sm_major, sm_minor))
      return false;
   cuModuleGetFunction(&k->clear_kernel, k->clear_module, "cp_clear_kernel");
   cuModuleGetFunction(&k->clear_depth_kernel, k->clear_module, "cp_clear_depth_kernel");
   cuModuleGetFunction(&k->depth_attachment_load, k->clear_module,
                       "cp_depth_attachment_load");
   cuModuleGetFunction(&k->depth_attachment_store, k->clear_module,
                       "cp_depth_attachment_store");
   cuModuleGetFunction(&k->cache_convert, k->clear_module,
                       "cp_cache_convert");

   if (!build_module(&k->module, cp_rasterize_src, "cp_rasterize.cu", sm_major, sm_minor))
      goto fail;
   cuModuleGetFunction(&k->rasterize_stage1, k->module, "cp_rasterize_stage1");
   cuModuleGetFunction(&k->rasterize_stage2, k->module, "cp_rasterize_stage2");
   cuModuleGetFunction(&k->rasterize_stage3, k->module, "cp_rasterize_stage3");
   cuModuleGetFunction(&k->clip_triangles, k->module, "cp_clip_triangles");
   k->rasterize_triangles = k->rasterize_stage1;
   cuModuleGetFunction(&k->clear_visbuf, k->module, "cp_clear_visbuf");
   cuModuleGetFunction(&k->peel_advance, k->module, "cp_peel_advance");
   cuModuleGetFunction(&k->resolve_visbuf, k->module, "cp_resolve_visbuf");

   /*
    * The A-buffer's own specialisation of the three rasterizer stages. Same
    * source, compiled a second time with the count-and-fill branch live, so
    * that the branch is chosen by which kernel the host launches rather than
    * tested on the device once per coverage event. See emit_fragment().
    */
   cuModuleGetFunction(&k->rasterize_stage1_abuf, k->module,
                       "cp_rasterize_stage1_abuf");
   cuModuleGetFunction(&k->rasterize_stage2_abuf, k->module,
                       "cp_rasterize_stage2_abuf");
   cuModuleGetFunction(&k->rasterize_stage3_abuf, k->module,
                       "cp_rasterize_stage3_abuf");

   cuModuleGetFunction(&k->clip_rast_fused, k->module, "cp_clip_rast_fused");
   cuModuleGetFunction(&k->clip_rast_fused_abuf, k->module,
                       "cp_clip_rast_fused_abuf");

   cuModuleGetFunction(&k->abuf_scan_block, k->module, "cp_abuf_scan_block");
   cuModuleGetFunction(&k->abuf_scan_add, k->module, "cp_abuf_scan_add");
   cuModuleGetFunction(&k->abuf_scan_reduce, k->module, "cp_abuf_scan_reduce");
   cuModuleGetFunction(&k->abuf_scan_finish, k->module, "cp_abuf_scan_finish");
   cuModuleGetFunction(&k->abuf_quad_count_all, k->module,
                       "cp_abuf_quad_count_all");
   cuModuleGetFunction(&k->abuf_quad_fill_all, k->module,
                       "cp_abuf_quad_fill_all");
   cuModuleGetFunction(&k->abuf_fuse_cmp, k->module, "cp_abuf_fuse_cmp");
   cuModuleGetFunction(&k->abuf_fuse_cover, k->module, "cp_abuf_fuse_cover");
   cuModuleGetFunction(&k->abuf_worklist, k->module, "cp_abuf_worklist");
   cuModuleGetFunction(&k->abuf_sort, k->module, "cp_abuf_sort");
   cuModuleGetFunction(&k->abuf_sort_short, k->module,
                       "cp_abuf_sort_short");
   cuModuleGetFunction(&k->abuf_block_worklist, k->module,
                       "cp_abuf_block_worklist");
   cuModuleGetFunction(&k->abuf_quad_count, k->module, "cp_abuf_quad_count");
   cuModuleGetFunction(&k->abuf_clear_slots, k->module, "cp_abuf_clear_slots");
   cuModuleGetFunction(&k->abuf_clamp_runs, k->module, "cp_abuf_clamp_runs");
   cuModuleGetFunction(&k->abuf_fill_recs, k->module, "cp_abuf_fill_recs");
   cuModuleGetFunction(&k->abuf_quad_fill, k->module, "cp_abuf_quad_fill");
   cuModuleGetFunction(&k->abuf_seg_count, k->module, "cp_abuf_seg_count");
   cuModuleGetFunction(&k->tile_census_mark, k->module, "cp_tile_census_mark");
   cuModuleGetFunction(&k->tile_census_mark_vis, k->module,
                       "cp_tile_census_mark_vis");
   cuModuleGetFunction(&k->tile_census_refs, k->module, "cp_tile_census_refs");
   cuModuleGetFunction(&k->tile_census_reduce, k->module,
                       "cp_tile_census_reduce");
   cuModuleGetFunction(&k->abuf_seg_scatter, k->module, "cp_abuf_seg_scatter");
   cuModuleGetFunction(&k->abuf_seg_prefix, k->module, "cp_abuf_seg_prefix");
   cuModuleGetFunction(&k->abuf_prepare_shade_count, k->module,
                       "cp_abuf_prepare_shade_count");
   cuModuleGetFunction(&k->opaque_tile_count, k->module,
                       "cp_opaque_tile_count");
   cuModuleGetFunction(&k->opaque_tile_fill, k->module,
                       "cp_opaque_tile_fill");
   cuModuleGetFunction(&k->opaque_tile_raster, k->module,
                       "cp_opaque_tile_raster");
   /* Only present when the instrumentation was compiled in, so only looked
    * up then; the draw path checks the pointers before launching. */
   if (cp_kernels_instrumented()) {
      cuModuleGetFunction(&k->abuf_peel_log, k->module, "cp_abuf_peel_log");
      cuModuleGetFunction(&k->abuf_peel_log_list, k->module,
                          "cp_abuf_peel_log_list");
   }

   if (!build_module(&k->fs_module, cp_fs_src, "cp_fs.cu", sm_major, sm_minor))
      goto fail;
   cuModuleGetFunction(&k->fs_interpolate, k->fs_module, "cp_fs_interpolate");
   cuModuleGetFunction(&k->fs_compact, k->fs_module, "cp_fs_compact");
   cuModuleGetFunction(&k->fs_writeback, k->fs_module, "cp_fs_writeback");
   cuModuleGetFunction(&k->resolve_samples, k->fs_module, "cp_resolve_samples");
   cuModuleGetFunction(&k->blit_linear, k->fs_module, "cp_blit_linear");
   /* The quad-stream interpolator lives beside the peel one so both call the
    * same cp_interp_pixel; see cp_fs.cu. */
   cuModuleGetFunction(&k->abuf_interpolate, k->fs_module,
                       "cp_abuf_interpolate");
   cuModuleGetFunction(&k->abuf_interpolate_ranges, k->fs_module,
                       "cp_abuf_interpolate_ranges");
   cuModuleGetFunction(&k->abuf_composite, k->fs_module, "cp_abuf_composite");
   if (cp_kernels_instrumented())
      cuModuleGetFunction(&k->abuf_scatter_colors, k->fs_module,
                          "cp_abuf_scatter_colors");

   if (!build_module(&k->vfetch_module, cp_vertex_fetch_src, "cp_vertex_fetch.cu", sm_major, sm_minor))
      goto fail;
   cuModuleGetFunction(&k->vertex_fetch, k->vfetch_module, "cp_vertex_fetch");

   /* Kept as relocatable PTX rather than a module: it is linked into each
    * shader that samples textures, not launched on its own. */
   k->sampler_ptx = compile_sampler_source("cp_sampler.cu", sm_major,
                                           sm_minor, false);
   if (cp_debug->texture_cache)
      k->math_ptx = compile_cuda_source(cp_math_src, "cp_math.cu", sm_major,
                                        sm_minor, true);
   if (!k->sampler_ptx) {
      fprintf(stderr, "cudavk: failed to compile texture sampler\n");
      goto fail;
   }
   if (cp_debug->texture_cache && !k->math_ptx)
      fprintf(stderr, "cudavk: exact math helper unavailable; sin/cos "
              "hardware texture modules will fall back\n");
   k->fs_helper_ptx = compile_cuda_source(cp_fs_src, "cp_fs_helper.cu",
                                          sm_major, sm_minor,
                                          true);
   if (!k->fs_helper_ptx) {
      fprintf(stderr, "cudavk: failed to compile fragment helpers\n");
      goto fail;
   }

   k->initialized = true;
   return true;

fail:
   cp_kernels_destroy(k);
   return false;
}

void
cp_kernels_destroy(struct cp_kernels *k)
{
   if (k->module)
      cuModuleUnload(k->module);
   if (k->clear_module)
      cuModuleUnload(k->clear_module);
   if (k->fs_module)
      cuModuleUnload(k->fs_module);
   if (k->vfetch_module)
      cuModuleUnload(k->vfetch_module);
   free(k->sampler_ptx);
   free(k->math_ptx);
   free(k->sampler_3d_ptx);
   free(k->fs_helper_ptx);
   memset(k, 0, sizeof(*k));
}
