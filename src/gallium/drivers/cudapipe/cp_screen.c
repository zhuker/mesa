#include "cp_screen.h"
#include "cp_context.h"
#include "cp_resource.h"
#include "cp_public.h"
#include "kernels/cp_rast_types.h"

#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"

#include "util/u_memory.h"
#include "util/u_screen.h"
#include "util/format/u_format.h"
#include "util/u_debug.h"
#include "compiler/nir/nir.h"

#include "frontend/sw_winsys.h"

#include <cuda.h>

static const char *
cp_get_vendor(struct pipe_screen *screen)
{
   return "Mesa";
}

static const char *
cp_get_name(struct pipe_screen *screen)
{
   struct cp_screen *cp = cp_screen(screen);
   return cp->renderer_string;
}

static const char *
cp_get_device_vendor(struct pipe_screen *screen)
{
   return "NVIDIA (cudapipe)";
}

static void
cp_init_screen_caps(struct pipe_screen *screen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&screen->caps;

   u_init_pipe_screen_caps(screen, 0);

   /*
    * Only claim what the kernels implement. Anything advertised here that the
    * driver can't actually do turns into a crash later rather than a clean
    * refusal, so this list stays deliberately short.
    */
   caps->npot_textures = true;
   caps->mixed_framebuffer_sizes = true;
   caps->mixed_color_depth_bits = true;
   caps->texture_mirror_clamp_to_edge = true;
   caps->texture_swizzle = true;
   caps->blend_equation_separate = true;
   caps->depth_clip_disable = true;
   caps->fragment_shader_texture_lod = true;
   caps->seamless_cube_map = true;

   /* Not implemented:
    *  - occlusion_query / conditional_render: no query support
    *  - texture_barrier: no explicit texture barriers
    *  - primitive_restart: the index walk has no restart handling
    *  - fragment_shader_derivatives: ddx/ddy return zero, since the fragment
    *    stage runs one thread per pixel with no quad neighbours
    *  - seamless_cube_map_per_texture: cube faces don't filter across seams
    */
   /* Advertised but falls back to trilinear — sufficient for validation. */
   caps->anisotropic_filter = true;
   caps->occlusion_query = false;
   caps->conditional_render = false;
   caps->texture_barrier = false;
   caps->primitive_restart = false;
   caps->fragment_shader_derivatives = false;
   caps->seamless_cube_map_per_texture = false;

   /* The fragment writeback resolves a single colour attachment, and blending
    * uses the state of render target zero. */
   caps->max_render_targets = 1;
   caps->max_dual_source_render_targets = 0;
   caps->indep_blend_enable = false;
   caps->indep_blend_func = false;
   caps->max_texture_2d_size = 16384;
   caps->max_texture_3d_levels = 12;
   caps->max_texture_cube_levels = 14;
   caps->max_texture_array_layers = 2048;
   caps->max_stream_output_buffers = 0;
   caps->max_vertex_streams = 1;
   caps->max_vertex_attrib_stride = 2048;
   caps->max_vertex_element_src_offset = 2047;
   caps->glsl_feature_level = 450;
   caps->glsl_feature_level_compatibility = 450;
   caps->constant_buffer_offset_alignment = 256;
   caps->min_map_buffer_alignment = 64;
   caps->endianness = PIPE_ENDIAN_LITTLE;
   caps->max_viewports = 1;
   caps->vendor_id = 0x10DE;
   caps->device_id = 0;
   caps->video_memory = 0;
   caps->uma = false;
}

static void
cp_init_shader_caps(struct pipe_screen *screen)
{
   for (unsigned i = 0; i < ARRAY_SIZE(screen->shader_caps); i++) {
      struct pipe_shader_caps *caps =
         (struct pipe_shader_caps *)&screen->shader_caps[i];

      if (i != MESA_SHADER_VERTEX &&
          i != MESA_SHADER_FRAGMENT &&
          i != MESA_SHADER_COMPUTE)
         continue;

      caps->max_instructions = 1 << 23;
      caps->max_alu_instructions = 1 << 23;
      caps->max_tex_instructions = 1 << 23;
      caps->max_tex_indirections = 1 << 23;
      caps->max_control_flow_depth = 1024;
      caps->max_inputs = 32;
      caps->max_outputs = 32;
      caps->max_const_buffer0_size = 65536;
      caps->max_const_buffers = 16;
      caps->max_temps = 4096;
      caps->cont_supported = true;
      caps->indirect_temp_addr = true;
      caps->indirect_const_addr = true;
      caps->integers = true;
      /* 64-bit integers aren't supported (caps.int64 is left false), and
       * 64-bit atomics without them is a contradiction the API rejects. */
      caps->int64_atomics = false;
      caps->max_texture_samplers = 32;
      caps->max_sampler_views = 32;
      caps->supported_irs = (1 << PIPE_SHADER_IR_NIR);
      caps->max_shader_buffers = 16;
      caps->max_shader_images = 16;
   }
}

static void
cp_init_compute_caps(struct pipe_screen *screen)
{
   struct pipe_compute_caps *caps =
      (struct pipe_compute_caps *)&screen->compute_caps;

   caps->max_grid_size[0] = 65535;
   caps->max_grid_size[1] = 65535;
   caps->max_grid_size[2] = 65535;
   caps->max_block_size[0] = 1024;
   caps->max_block_size[1] = 1024;
   caps->max_block_size[2] = 64;
   caps->max_threads_per_block = 1024;
   caps->max_local_size = 49152;
   caps->grid_dimension = 3;
   caps->max_global_size = 1ull << 32;
   caps->max_mem_alloc_size = 1ull << 32;
   caps->max_compute_units = 80;
   caps->max_clock_frequency = 1500;
   caps->address_bits = 64;
   caps->subgroup_sizes = 32;
   caps->max_subgroups = 32;
}

/*
 * Report format support.
 *
 * This answers for the kernels rather than for the API: a format is supported
 * for sampling only if the sampler can decode it, and for rendering only if
 * the fragment writeback can encode it. Claiming more than that doesn't make
 * anything work — it invites applications down paths the driver then crashes
 * on, instead of letting them pick something else.
 */
static bool
cp_is_format_supported(struct pipe_screen *screen, enum pipe_format format,
                       enum pipe_texture_target target, unsigned sample_count,
                       unsigned storage_sample_count, unsigned bind)
{
   /* Support 1x and 4x multisampling. */
   if (sample_count > 4)
      return false;
   if (sample_count == 3 || sample_count == 2)
      return false;
   if (MAX2(1, sample_count) != MAX2(1, storage_sample_count))
      return false;

   switch (target) {
   case PIPE_BUFFER:
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_3D:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
   case PIPE_TEXTURE_RECT:
      break;
   default:
      return false;
   }

   /*
    * A compressed format is unusable unless the sampler can decode its blocks:
    * there is no path that treats it as opaque bytes end to end, since even
    * clearing and blitting have to understand the block layout. Rejecting them
    * here rather than per-bind matters, because image creation, copies and
    * blits arrive with bind flags naming none of the paths below — which is
    * how ASTC and ETC2 were getting through and crashing.
    *
    * Uncompressed formats deliberately fall through: copying one is a byte
    * move that works whether or not the sampler understands the format, so the
    * decode and encode requirements below are applied per bind instead.
    */
   if (util_format_is_compressed(format) &&
       cp_texel_encoding_from_format(format) == CP_TEXEL_UNSUPPORTED)
      return false;

   if (bind & PIPE_BIND_RENDER_TARGET) {
      if (cp_color_encoding_from_format(format) < 0)
         return false;
   }

   if (bind & PIPE_BIND_DEPTH_STENCIL) {
      /* Depth only: there is no stencil buffer and no stencil test, so a
       * combined format would silently drop the stencil aspect. */
      switch (format) {
      case PIPE_FORMAT_Z16_UNORM:
      case PIPE_FORMAT_Z32_FLOAT:
      case PIPE_FORMAT_Z24X8_UNORM:
         break;
      default:
         return false;
      }
   }

   if (bind & PIPE_BIND_SAMPLER_VIEW) {
      if (util_format_is_depth_or_stencil(format)) {
         /* Sampling depth returns floats through the normal path; stencil
          * would need integer returns, which the sampler doesn't do. */
         if (util_format_has_stencil(util_format_description(format)))
            return false;
      } else if (cp_texel_encoding_from_format(format) == CP_TEXEL_UNSUPPORTED) {
         return false;
      }
   }

   /* Images are loaded and stored by the shader backend, which handles the
    * same uncompressed formats the sampler does and no compressed ones. */
   if (bind & PIPE_BIND_SHADER_IMAGE) {
      if (util_format_is_compressed(format) ||
          util_format_is_depth_or_stencil(format) ||
          cp_texel_encoding_from_format(format) == CP_TEXEL_UNSUPPORTED)
         return false;
   }

   /* Vertex attributes are fetched as raw bytes and reinterpreted by the
    * shader, so anything uncompressed works. */
   if (bind & PIPE_BIND_VERTEX_BUFFER) {
      if (util_format_is_compressed(format))
         return false;
   }

   return true;
}

static const struct nir_shader_compiler_options cp_nir_options = {
   .lower_scmp = true,
   .lower_flrp32 = true,
   .lower_flrp64 = true,
   .lower_flrp16 = true,
   .lower_fsat = true,
   .lower_bitfield_insert = true,
   .lower_bitfield_extract = true,
   .lower_fdph = true,
   .lower_fmod = true,
   /* NVPTX has no libcall for pow or the trig functions, so llvm.pow and
    * llvm.sin fail instruction selection and take the backend down with them.
    * Let NIR express them in terms of exp2/log2 and the fractional-turn
    * reductions, which do select. */
   .lower_fpow = true,
   /* Keep fsin/fcos: the backend routes them to the CUDA library versions,
    * which are far more accurate than NIR's lowered polynomial. That accuracy
    * matters where a shader rotates a position rather than a direction — the
    * instancing sample's asteroids orbit at radius 7 with vertices spanning
    * 0.06, an 80x lever that turned the polynomial's error into a visible
    * displacement of every rock. llvmpipe leaves this off for the same reason. */
   .lower_sincos = false,
   .lower_hadd = true,
   .lower_uadd_sat = true,
   .lower_usub_sat = true,
   .lower_iadd_sat = true,
   .lower_pack_snorm_2x16 = true,
   .lower_pack_snorm_4x8 = true,
   .lower_pack_unorm_2x16 = true,
   .lower_pack_unorm_4x8 = true,
   .lower_pack_half_2x16 = true,
   .lower_unpack_snorm_2x16 = true,
   .lower_unpack_snorm_4x8 = true,
   .lower_unpack_unorm_2x16 = true,
   .lower_unpack_unorm_4x8 = true,
   .lower_unpack_half_2x16 = true,
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_mul_2x32_64 = true,
   .lower_ifind_msb = true,
   .max_unroll_iterations = 32,
   .lower_to_scalar = true,
   .lower_uniforms_to_ubo = true,
   .lower_device_index_to_zero = true,
   .support_16bit_alu = false,
};

static void
cp_finalize_nir(struct pipe_screen *screen, struct nir_shader *nir, bool optimize)
{
   /* TODO: apply cudapipe-specific NIR lowering passes */
}

static void
cp_destroy_screen(struct pipe_screen *screen)
{
   struct cp_screen *cp = cp_screen(screen);
   cp_kernels_destroy(&cp->kernels);
   cuCtxDestroy(cp->cuda_ctx);
   FREE(cp);
}

static void
cp_fence_reference(struct pipe_screen *screen,
                   struct pipe_fence_handle **dst,
                   struct pipe_fence_handle *src)
{
   *dst = src;
}

static bool
cp_fence_finish(struct pipe_screen *screen, struct pipe_context *ctx,
                struct pipe_fence_handle *fence, uint64_t timeout)
{
   if (!fence)
      return true;
   struct cp_screen *cp = (struct cp_screen *)screen;
   cuCtxSetCurrent(cp->cuda_ctx);
   cuCtxSynchronize();
   return true;
}

static void
cp_query_memory_info(struct pipe_screen *screen,
                     struct pipe_memory_info *info)
{
   memset(info, 0, sizeof(*info));
}

static void
cp_get_device_uuid(struct pipe_screen *screen, char *uuid)
{
   memset(uuid, 0, 16);
   memcpy(uuid, "cudapipe", 8);
}

static void
cp_get_driver_uuid(struct pipe_screen *screen, char *uuid)
{
   memset(uuid, 0, 16);
   memcpy(uuid, "cudapipe", 8);
}

static uint64_t
cp_get_timestamp(struct pipe_screen *screen)
{
   return 0;
}

struct pipe_screen *
cudapipe_create_screen(struct sw_winsys *winsys)
{
   struct cp_screen *screen;

   screen = CALLOC_STRUCT(cp_screen);
   if (!screen)
      return NULL;

   screen->winsys = winsys;

   if (cuInit(0) != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: cuInit failed\n");
      FREE(screen);
      return NULL;
   }

   if (cuDeviceGet(&screen->cuda_device, 0) != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: cuDeviceGet failed\n");
      FREE(screen);
      return NULL;
   }

   {
      /* CUDA 13 added a ctx-params argument; 12.x takes (ctx, flags, dev). */
      CUresult err = cuCtxCreate(&screen->cuda_ctx, 0, screen->cuda_device);
      if (err != CUDA_SUCCESS) {
         fprintf(stderr, "cudapipe: cuCtxCreate failed (%d)\n", err);
         FREE(screen);
         return NULL;
      }
   }

   cuDeviceGetAttribute(&screen->sm_major,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        screen->cuda_device);
   cuDeviceGetAttribute(&screen->sm_minor,
                        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        screen->cuda_device);

   snprintf(screen->renderer_string, sizeof(screen->renderer_string),
            "cudapipe (sm_%d%d)", screen->sm_major, screen->sm_minor);

   if (!cp_kernels_init(&screen->kernels, screen)) {
      fprintf(stderr, "cudapipe: warning: rasterization kernels failed to compile\n");
      /* Non-fatal — compute still works, just no draw support */
   }

   screen->base.destroy = cp_destroy_screen;
   screen->base.get_name = cp_get_name;
   screen->base.get_vendor = cp_get_vendor;
   screen->base.get_device_vendor = cp_get_device_vendor;
   screen->base.is_format_supported = cp_is_format_supported;
   screen->base.context_create = cudapipe_create_context;
   screen->base.fence_reference = cp_fence_reference;
   screen->base.fence_finish = cp_fence_finish;
   screen->base.query_memory_info = cp_query_memory_info;
   screen->base.get_device_uuid = cp_get_device_uuid;
   screen->base.get_driver_uuid = cp_get_driver_uuid;
   screen->base.get_timestamp = cp_get_timestamp;
   screen->base.finalize_nir = cp_finalize_nir;

   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++)
      screen->base.nir_options[i] = &cp_nir_options;

   cp_init_screen_caps(&screen->base);
   cp_init_shader_caps(&screen->base);
   cp_init_compute_caps(&screen->base);
   cudapipe_init_screen_resource_funcs(&screen->base);

   return &screen->base;
}
