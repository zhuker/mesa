#ifndef CP_NIR_TO_LLVM_H
#define CP_NIR_TO_LLVM_H

#include <cuda.h>
#include <stdbool.h>
#include <stdint.h>

struct nir_shader;
struct cp_sampler_info;

#define CP_MAX_IO_SLOTS 32
#define CP_MAX_TEX_DESCS 8
#define CP_MAX_SAMPLER_VARIANTS 4

struct cp_tex_desc_ref {
   uint16_t ubo_slot;
   uint16_t reserved;
   uint32_t sampler_offset;
   int32_t flags;
};

struct cp_sampler_variant {
   CUmodule module;
   CUfunction kernel;
   struct cp_sampler_info *states;
   unsigned num_states;
   int regs;
   int spill_bytes;
   bool globals_resolved;
   CUdeviceptr sym_sampler_table;
   CUdeviceptr sym_quad_derivs;
   uint64_t last_sampler_table;
   int last_quad_derivs;
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

struct cp_shader_binary {
   char *ptx_text;
   size_t ptx_size;
   CUmodule module;
   CUfunction kernel;
   int sm_major;
   int sm_minor;
   unsigned shared_size;
   unsigned nir_num_outputs;
   unsigned nir_num_inputs;

   /*
    * What the built kernel cost in registers, what it spills, and how many
    * 256-thread blocks of it fit on an SM — read back from the driver after
    * the module is loaded, which is the only place the number exists: the JIT
    * decides it and nothing upstream of it can be asked. `reg_cap` is the cap
    * that was applied, or 0 for a shader left as the JIT built it.
    */
   int num_regs;
   int spill_bytes;
   int blocks_per_sm;
   int reg_cap;

   /*
    * The register cap this shader is a candidate for, or 0 for one that is
    * not. Non-zero means the shader is a fragment shader whose own register
    * count is holding its occupancy below the target, so capping it *might*
    * pay — which of the two builds is actually faster is settled by timing
    * both on this workload, in cp_context.c. `tune_done` is set once that has
    * happened, so it happens once per shader.
    */
   int tune_cap;
   bool tune_done;
   struct cp_shader_tune tune;

   /*
    * The other build of a candidate shader: the same PTX with the cap on, or
    * off, whichever the fields above are not currently using. Both are built
    * when the pipeline is created and both stay loaded, and switching between
    * them is a pointer swap.
    *
    * Both are built up front so that the choice, when the driver makes it,
    * costs nothing on a hot path: building the second one during the run
    * would put a JIT link and the drain it needs between two draws, and the
    * cost of that would land on whichever build happened to be rebuilt —
    * which is the one thing a comparison between them cannot afford.
    */
   CUmodule alt_module;
   CUfunction alt_kernel;
   int alt_regs;
   int alt_spill_bytes;
   int alt_blocks_per_sm;
   int alt_reg_cap;
   bool alt_globals_resolved;
   CUdeviceptr alt_sym_sampler_table;
   CUdeviceptr alt_sym_quad_derivs;
   uint64_t alt_last_sampler_table;
   int alt_last_quad_derivs;

   /* Borrowed from the screen, which owns it for the life of the driver: the
    * sampler PTX this shader was linked against, kept so the shader can be
    * rebuilt with a different cap after the fact. NULL for a shader that
    * links nothing. */
   const char *sampler_ptx;
   const char *fs_helper_ptx;
   bool uses_tex_3d;

   /* Whether the shader reads gl_VertexIndex. Only then does a non-indexed
    * draw have to materialise the vertex id array the shader reads from. */
   bool reads_vertex_id;

   /* Whether the shader reads gl_InstanceIndex. Same bargain: the array is one
    * uint32 per assembled vertex, which is megabytes for an instanced draw,
    * and a shader that ignores the id does not need it written at all. */
   bool reads_instance_id;

   /* Whether the shader can discard. An alpha-tested draw needs several passes
    * because visibility is resolved before shading. */
   bool uses_discard;

   /*
    * Whether the shader writes globally visible memory — a storage image, an
    * SSBO, a global address — by store or by atomic. Vulkan lets a fragment
    * shader exist purely for those writes, with no colour attachment at all
    * (the `oit` sample builds its per-pixel linked lists that way), so the
    * fragment stage cannot be skipped just because there is nowhere to put a
    * colour. Gathered from the intrinsics rather than read out of
    * nir_shader_info, so that it describes the NIR this PTX was generated
    * from and cannot go stale behind a lowering pass.
    */
   bool writes_memory;

   /*
    * Whether the fragment shader reads gl_FrontFacing. Only then does the
    * interpolator write the per-slot facing byte the shader reads, which is
    * one byte per shaded pixel and so worth not writing for the shaders — the
    * large majority — that never ask.
    */
   bool reads_front_face;

   /*
    * Whether the shader reads gl_BaseVertex, gl_BaseInstance or gl_DrawID.
    * Those come from the per-draw parameter table at args[7], one
    * CP_ARG_DRAW_PARAM_STRIDE row per merged draw indexed by the batch row —
    * so draws that disagree in them may still merge. The host needs the flag
    * to know the shader wants the batch-row array at all; see
    * cp_draw_execute_batch()'s batch_rows gating.
    */
   bool reads_draw_params;

   /*
    * Whether the shader reads any constant buffer — a uniform block, a push
    * constant range, or the descriptor behind a sampler, all of which reach it
    * through the same argument slots. False means what is bound in those slots
    * cannot change what the shader computes, which is what lets consecutive
    * draws be batched across a binding the fragment stage never looks at. It
    * is recorded by the one code generator function that reads those slots, so
    * it cannot drift from what the PTX actually does.
    */
   bool reads_const_bufs;

   /* Statically traceable sampler descriptors used by this shader. Each
    * sampler handle is a fixed byte offset from a constant-buffer base. */
   unsigned num_tex_descs;
   struct cp_tex_desc_ref tex_descs[CP_MAX_TEX_DESCS];
   bool tex_descs_dynamic;
   bool tex_descs_reported;

   struct cp_sampler_variant sampler_variants[CP_MAX_SAMPLER_VARIANTS];
   unsigned num_sampler_variants;

   /* Which VARYING_SLOT_* each I/O slot carries, so a fragment shader's inputs
    * can be matched to the vertex shader's outputs by location rather than by
    * position. Slots with no variable are VARYING_SLOT_MAX. */
   unsigned in_location[CP_MAX_IO_SLOTS];
   unsigned out_location[CP_MAX_IO_SLOTS];

   /*
    * The linked sampler reads two things through module globals rather than
    * through the argument block: the sampler table, and whether fragment
    * threads are laid out four to a quad so derivatives can be shuffled
    * between them. Both were resolved and rewritten on every draw, which is
    * two cuModuleGetGlobal and two cuMemcpyHtoD per draw to store bytes that
    * almost never change — and a draw of a dozen triangles is host-bound.
    *
    * The addresses are resolved once and the last value written is kept, so
    * the copy happens only when the value actually differs. `globals_resolved`
    * distinguishes "not looked up yet" from "this module has no sampler", the
    * latter leaving both symbols zero.
    */
   bool globals_resolved;
   CUdeviceptr sym_sampler_table;
   CUdeviceptr sym_quad_derivs;
   uint64_t last_sampler_table;
   int last_quad_derivs;
};

/* `sampler_ptx` is the relocatable PTX of the texture sampler, linked in when
 * the shader samples textures. May be NULL for shaders that cannot. */
struct cp_shader_binary *
cp_compile_nir_to_ptx(struct nir_shader *nir, int sm_major, int sm_minor,
                      const char *sampler_ptx, const char *fs_helper_ptx);

void
cp_shader_binary_destroy(struct cp_shader_binary *bin);

bool cp_shader_build_sampler_variant(struct cp_shader_binary *bin,
                                     const char *sampler_ptx,
                                     const struct cp_sampler_info *states,
                                     unsigned num_states);

struct cp_sampler_variant *
cp_shader_find_sampler_variant(struct cp_shader_binary *bin,
                               const struct cp_sampler_info *states,
                               unsigned num_states);

/* Rebuild a loaded shader with a register cap (0 for none), in place. The
 * caller must have drained the stream first. False leaves it untouched. */
bool
cp_shader_set_reg_cap(struct cp_shader_binary *bin, int max_regs);

/* Swap the shader's two builds over, so the one that was inactive is the one
 * launched. Costs nothing on the device: both are already loaded. */
void
cp_shader_swap_build(struct cp_shader_binary *bin);

#endif
