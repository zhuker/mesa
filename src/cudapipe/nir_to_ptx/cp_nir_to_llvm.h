#ifndef CP_NIR_TO_LLVM_H
#define CP_NIR_TO_LLVM_H

#include <cuda.h>
#include <stdbool.h>
#include <stdint.h>

struct nir_shader;
struct cp_sampler_info;

#define CP_MAX_IO_SLOTS 32
#define CP_MAX_TEX_DESCS 8
#define CP_MAX_HW_TEX_SITES 64
#define CP_MAX_SAMPLER_VARIANTS 4

/* Existing deduplicated software sampler-specialisation reference. */
struct cp_tex_desc_ref {
   uint16_t ubo_slot;
   uint16_t reserved;
   uint32_t sampler_offset;
   int32_t flags;
};

struct cp_hw_tex_ref {
   uint16_t ubo_slot;
   uint16_t reserved;
   uint32_t offset;
};

/* One static sampled NIR instruction, in immutable hardware table order. */
struct cp_hw_tex_site {
   struct cp_hw_tex_ref image;
   struct cp_hw_tex_ref sampler;
   int32_t flags;
   uint8_t op;
   uint8_t dim;
   uint8_t coord_components;
   uint8_t reserved;
};

/* Launches timed per phase of a shader's register-cap trial, and how many are
 * thrown away at the start of one. See cp_tune_before() in cp_context.c. */
#define CP_TUNE_SAMPLES 24
#define CP_TUNE_SKIP    16

/*
 * A fragment shader's register-cap trial: the driver runs the shader as the
 * JIT built it and then capped, times both on the device, and keeps the
 * faster. Lives on the shader because trials overlap — one per shader, all
 * running against the same frames — and because the decision is the shader's.
 */
struct cp_shader_tune {
   int phase;              /* 0 capped, 1 as the JIT built it */
   unsigned seen;          /* launches this phase */
   unsigned timed;         /* of those, ones with events on them */
   int regs_capped;
   bool swap_pending;      /* switch builds after the launch in flight */
   bool reading;           /* a phase's events are recorded, not yet read */
   bool events_made;
   CUevent start[CP_TUNE_SAMPLES];
   CUevent stop[CP_TUNE_SAMPLES];
   float us[2][CP_TUNE_SAMPLES];
};

enum cp_shader_exec_mode {
   CP_SHADER_EXEC_CLASSIC = 0,
   CP_SHADER_EXEC_FUSED = 1,
   CP_SHADER_EXEC_INLINE = 2,
   /* Resource-isolated same-LLVM interpolation plus direct NVVM texture ops. */
   CP_SHADER_EXEC_HW_INLINE = 3,
   /* Resource-isolated texture ops with the proven fused interpolation helper. */
   CP_SHADER_EXEC_HW_FUSED = 4,
   CP_SHADER_EXEC_COUNT = 5,
};

/*
 * One independently linked execution of a shader.  A fragment shader owns a
 * classic execution whose PTX has no fused-interpolation reference and a
 * fused execution which links cp_fs.cu.  Module resources, register-cap
 * alternatives, tuning state and module globals all belong here: none of
 * those facts may transfer between the two binaries.
 */
struct cp_shader_exec {
   char *ptx_text;
   size_t ptx_size;
   CUmodule module;
   CUfunction kernel;
   int num_regs;
   int spill_bytes;
   int blocks_per_sm;
   int reg_cap;

   CUmodule alt_module;
   CUfunction alt_kernel;
   int alt_regs;
   int alt_spill_bytes;
   int alt_blocks_per_sm;
   int alt_reg_cap;

   int tune_cap;
   bool tune_done;
   struct cp_shader_tune tune;

   bool globals_resolved;
   CUdeviceptr sym_sampler_table;
   CUdeviceptr sym_quad_derivs;
   uint64_t last_sampler_table;
   int last_quad_derivs;
   bool alt_globals_resolved;
   CUdeviceptr alt_sym_sampler_table;
   CUdeviceptr alt_sym_quad_derivs;
   uint64_t alt_last_sampler_table;
   int alt_last_quad_derivs;

   /* Borrowed from screen-owned kernel PTX. */
   const char *sampler_ptx;
   const char *fs_helper_ptx;
};

struct cp_sampler_variant {
   struct cp_sampler_info *states;
   unsigned num_states;
   struct cp_shader_exec exec[CP_SHADER_EXEC_COUNT];
};

void cp_shader_exec_swap_build(struct cp_shader_exec *exec);

enum cp_hw_compile_failure {
   CP_HW_COMPILE_NONE = 0,
   CP_HW_COMPILE_INELIGIBLE,
   CP_HW_COMPILE_INLINE_FOOTPRINT,
   CP_HW_COMPILE_SURVIVING_HELPER,
   CP_HW_COMPILE_LOCAL_MEMORY,
   CP_HW_COMPILE_BAD_TEXTURE_PTX,
   CP_HW_COMPILE_JIT,
   CP_HW_COMPILE_OTHER,
   CP_HW_COMPILE_FAILURE_COUNT,
};

struct cp_shader_binary {
   struct cp_shader_exec exec[CP_SHADER_EXEC_COUNT];
   int sm_major;
   int sm_minor;
   unsigned shared_size;
   unsigned nir_num_outputs;
   unsigned nir_num_inputs;
   bool is_fragment;
   bool classic_fallback_reported;
   bool fused_fallback_reported;
   bool inline_fallback_reported;
   bool hw_inline_fallback_reported;

   bool uses_tex_3d;

   /* Whether the shader reads gl.VertexIndex. Only then does a non-indexed
    * draw have to materialise the vertex id array the shader reads from. */
   bool reads_vertex_id;

   /* Whether the shader reads gl.InstanceIndex. Same bargain: the array is one
    * uint32 per assembled vertex, which is megabytes for an instanced draw,
    * and a shader that ignores the id does not need it written at all. */
   bool reads_instance_id;

   /* Whether the shader can discard. An alpha-tested draw needs several passes
    * because visibility is resolved before shading. */
   bool uses_discard;

   /* Whether the shader writes globally visible memory. */
   bool writes_memory;

   /* Whether the fragment shader reads gl.FrontFacing. */
   bool reads_front_face;

   /* Whether the shader reads gl.BaseVertex, gl.BaseInstance or gl.DrawID. */
   bool reads_draw_params;

   /* Whether the shader reads any constant buffer. */
   bool reads_const_bufs;

   /* Statically traceable sampler descriptors used by this shader. */
   unsigned num_tex_descs;
   struct cp_tex_desc_ref tex_descs[CP_MAX_TEX_DESCS];
   bool tex_descs_dynamic;
   bool tex_descs_reported;
   unsigned num_tex_instrs;
   bool spec_counted;
   bool spec_rejected_reported;

   /* Independent all-or-nothing hardware texture metadata. */
   unsigned num_hw_tex_sites;
   struct cp_hw_tex_site hw_tex_sites[CP_MAX_HW_TEX_SITES];
   bool hw_tex_dynamic;
   uint8_t hw_compile_failure;
   bool hw_failure_counted;

   struct cp_sampler_variant sampler_variants[CP_MAX_SAMPLER_VARIANTS];
   unsigned num_sampler_variants;

   /* Which VARYING_SLOT_* each I/O slot carries. */
   unsigned in_location[CP_MAX_IO_SLOTS];
   unsigned out_location[CP_MAX_IO_SLOTS];
};

static inline struct cp_shader_exec *
cp_shader_exec(struct cp_shader_binary *bin, enum cp_shader_exec_mode mode)
{
   return bin ? &bin->exec[mode] : NULL;
}

static inline const struct cp_shader_exec *
cp_shader_exec_const(const struct cp_shader_binary *bin,
                     enum cp_shader_exec_mode mode)
{
   return bin ? &bin->exec[mode] : NULL;
}

static inline bool
cp_shader_has_any_exec(const struct cp_shader_binary *bin)
{
   return bin && (bin->exec[CP_SHADER_EXEC_CLASSIC].kernel ||
                  bin->exec[CP_SHADER_EXEC_FUSED].kernel ||
                  bin->exec[CP_SHADER_EXEC_INLINE].kernel ||
                  bin->exec[CP_SHADER_EXEC_HW_INLINE].kernel ||
                  bin->exec[CP_SHADER_EXEC_HW_FUSED].kernel);
}

/* Inline execution needs interpolation scratch. Classic and fused can both
 * run after the standalone interpolator, so either is a correctness fallback. */
static inline bool
cp_shader_has_standalone_exec(const struct cp_shader_binary *bin)
{
   return bin && (bin->exec[CP_SHADER_EXEC_CLASSIC].kernel ||
                  bin->exec[CP_SHADER_EXEC_FUSED].kernel);
}

/* `sampler_ptx` is the relocatable PTX of the texture sampler, linked in when
 * the shader samples textures. May be NULL for shaders that cannot. */
struct cp_shader_binary *
cp_compile_nir_to_ptx(struct nir_shader *nir, int sm_major, int sm_minor,
                      const char *sampler_ptx, const char *math_ptx,
                      const char *fs_helper_ptx,
                      bool no_inline_fs, bool inline_fs, bool force_fused_fs,
                      bool hw_texture);

void
cp_shader_binary_destroy(struct cp_shader_binary *bin);

bool cp_shader_build_sampler_variant(struct cp_shader_binary *bin,
                                     const char *sampler_ptx,
                                     const struct cp_sampler_info *states,
                                     unsigned num_states,
                                     enum cp_shader_exec_mode mode);

struct cp_sampler_variant *
cp_shader_find_sampler_variant(struct cp_shader_binary *bin,
                               const struct cp_sampler_info *states,
                               unsigned num_states);

#endif
