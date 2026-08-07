#include "cp_screen.h"
#include "cp_context.h"
#include "cp_resource.h"
#include "cp_public.h"

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

   caps->npot_textures = true;
   caps->mixed_framebuffer_sizes = true;
   caps->mixed_color_depth_bits = true;
   caps->anisotropic_filter = true;
   caps->occlusion_query = true;
   caps->texture_mirror_clamp_to_edge = true;
   caps->texture_swizzle = true;
   caps->blend_equation_separate = true;
   caps->indep_blend_enable = true;
   caps->indep_blend_func = true;
   caps->depth_clip_disable = true;
   caps->fragment_shader_texture_lod = true;
   caps->fragment_shader_derivatives = true;
   caps->primitive_restart = true;
   caps->conditional_render = true;
   caps->texture_barrier = true;
   caps->seamless_cube_map = true;
   caps->seamless_cube_map_per_texture = true;
   caps->max_dual_source_render_targets = 1;
   caps->max_render_targets = 4;
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
      caps->int64_atomics = true;
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

static bool
cp_is_format_supported(struct pipe_screen *screen, enum pipe_format format,
                       enum pipe_texture_target target, unsigned sample_count,
                       unsigned storage_sample_count, unsigned bind)
{
   if (sample_count > 1)
      return false;

   if (MAX2(1, sample_count) != MAX2(1, storage_sample_count))
      return false;

   if (bind & PIPE_BIND_RENDER_TARGET) {
      switch (format) {
      case PIPE_FORMAT_R8G8B8A8_UNORM:
      case PIPE_FORMAT_B8G8R8A8_UNORM:
      case PIPE_FORMAT_R8G8B8X8_UNORM:
      case PIPE_FORMAT_B8G8R8X8_UNORM:
      case PIPE_FORMAT_R16G16B16A16_FLOAT:
      case PIPE_FORMAT_R16G16_FLOAT:
      case PIPE_FORMAT_R32G32B32A32_FLOAT:
      case PIPE_FORMAT_R8_UNORM:
      case PIPE_FORMAT_R8G8_UNORM:
      case PIPE_FORMAT_R11G11B10_FLOAT:
      case PIPE_FORMAT_R10G10B10A2_UNORM:
         break;
      default:
         return false;
      }
   }

   if (bind & PIPE_BIND_DEPTH_STENCIL) {
      switch (format) {
      case PIPE_FORMAT_Z16_UNORM:
      case PIPE_FORMAT_Z32_FLOAT:
      case PIPE_FORMAT_Z24X8_UNORM:
      case PIPE_FORMAT_Z24_UNORM_S8_UINT:
      case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
         break;
      default:
         return false;
      }
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
   .support_16bit_alu = true,
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
   /* TODO: cuEventSynchronize */
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
      CUctxCreateParams params = {0};
      CUresult err = cuCtxCreate(&screen->cuda_ctx, &params, 0, screen->cuda_device);
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
