#include "cp_nir_to_llvm.h"
#include "cp_debug.h"

#include "compiler/nir/nir.h"
#include "util/u_memory.h"
#include "kernels/cp_rast_types.h"
#include "cp_shader_abi.h"

#include <llvm-c/Core.h>
#include <llvm/Config/llvm-config.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/Analysis.h>
#ifdef CP_HAVE_FS_INLINE_BC
#include <llvm-c/BitReader.h>
#include <llvm-c/Linker.h>
#include <llvm-c/Error.h>
#include <llvm-c/Transforms/PassBuilder.h>
#endif

#include <stdio.h>
#include <string.h>

/* Highest SM architecture we ask LLVM's NVPTX backend to target, as major*10 +
 * minor. Kept at the conservative end of what every LLVM we build against
 * understands: the generated PTX is JIT-compiled by the driver, so a newer GPU
 * runs it fine, while naming an architecture LLVM does not know makes it fall
 * back to a default that emits unloadable PTX. Raise this only alongside the
 * PTX ISA version in the target machine's feature string below. */
#define CP_MAX_PTX_SM 86

#ifdef CP_HAVE_FS_INLINE_BC
static const unsigned char cp_fs_inline_bc[] = {
#include "cp_fs_inline.bc.inc"
};
#else
static once_flag cp_inline_missing_once = ONCE_FLAG_INIT;
static void
cp_warn_inline_missing(void)
{
   fprintf(stderr, "cudapipe: inline FS unavailable: matching clang "
           "bitcode was not built; using fused helper binary\n");
}
#endif

/* Argument slots per launch; the host fills this array — see cp_context.c. */
#define CP_MAX_ARG_SLOTS 64

struct ntl_context {
   LLVMContextRef llvm_ctx;
   LLVMModuleRef module;
   LLVMBuilderRef builder;
   LLVMValueRef function;

   LLVMValueRef *ssa_defs;
   unsigned num_ssa_defs;

   LLVMValueRef *regs;
   unsigned num_regs;

   LLVMValueRef *kernel_args;
   unsigned num_kernel_args;

   /* One load per argument slot, in the entry block; see cp_arg_slot(). */
   LLVMValueRef arg_slots[CP_MAX_ARG_SLOTS];
   LLVMBasicBlockRef entry_block;
   unsigned md_invariant_load;

   /* The grid-stride loop's induction value, standing in for ctaid.x inside
    * a drawing stage's body; null outside one. See emit_workgroup_id(). */
   LLVMValueRef virtual_bid;

   /* Set when the shader samples a texture, so the sampler PTX gets linked in. */
   bool uses_tex;
   /*
    * Set the first time emit_const_buf_base() runs. That function is the only
    * place the generated code touches args[18..], so a shader it was never
    * called for cannot read a constant buffer at all — descriptors, samplers
    * and push constants included, since every one of them arrives through it.
    * Draw batching uses that to stop caring what is bound where the shader
    * does not look.
    */
   bool reads_const_bufs;
   bool needs_link;   /* pull in cp_sampler.cu for its device helpers */
   /* This module emits direct texture-object NVVM operations and must never
    * acquire a cp_sampler.cu declaration or link dependency. */
   bool hardware_texture;
   unsigned hardware_texture_site;
   unsigned hardware_texture_site_count;

   /*
    * Set when the shader writes globally visible memory. Gathered from the
    * intrinsics before code generation starts, so that the fragment prologue
    * and the flag the driver records afterwards are the same answer about the
    * same NIR. See cp_nir_writes_memory().
    */
   bool writes_memory;
   bool fused_interp;
   bool inline_interp;
   LLVMValueRef inline_fs_inputs;
   uint32_t inline_live_slots;
   int32_t inline_pntc_input;
   int32_t inline_pos_input;

   LLVMBasicBlockRef break_block;
   LLVMBasicBlockRef continue_block;

   struct nir_shader *nir;
   int sm_major;
   int sm_minor;
};

/*
 * The pointer in argument slot `index`, loaded once.
 *
 * Every stage reads its buffers out of a 64-entry array of pointers the host
 * fills in before the launch, and each site that wanted one used to load it
 * again. LLVM cannot merge those by itself: any store through any of the
 * pointers might alias the array, so a vertex shader that reads an input and
 * then writes an output reloaded the input pointer and the stride afterwards,
 * and each reload is a dependent global load in front of the real work.
 *
 * Two things fix it. The load is marked invariant, which is true -- the block
 * is written before the launch and nothing in the kernel writes it -- and it
 * is emitted in the entry block and cached, so it dominates every use and one
 * load serves them all.
 */
static LLVMValueRef
cp_arg_slot(struct ntl_context *ctx, unsigned index)
{
   assert(index < CP_MAX_ARG_SLOTS);
   if (ctx->arg_slots[index])
      return ctx->arg_slots[index];

   LLVMTypeRef i8 = LLVMInt8TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef ptr_type = LLVMPointerType(i8, 0);
   LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);

   /* Emit into the entry block so the value dominates wherever it is used. */
   LLVMBasicBlockRef saved = LLVMGetInsertBlock(ctx->builder);
   LLVMValueRef term = LLVMGetBasicBlockTerminator(ctx->entry_block);
   if (term)
      LLVMPositionBuilderBefore(ctx->builder, term);
   else
      LLVMPositionBuilderAtEnd(ctx->builder, ctx->entry_block);

   LLVMValueRef args_pp = LLVMBuildBitCast(ctx->builder, ctx->kernel_args[0],
                                           LLVMPointerType(ptr_type, 0), "");
   LLVMValueRef slot = LLVMBuildLoad2(ctx->builder, ptr_type,
      LLVMBuildGEP2(ctx->builder, ptr_type, args_pp,
         &(LLVMValueRef){LLVMConstInt(i64, index, false)}, 1, ""), "arg_slot");
   LLVMSetMetadata(slot, ctx->md_invariant_load,
                   LLVMMDNodeInContext(ctx->llvm_ctx, NULL, 0));

   LLVMPositionBuilderAtEnd(ctx->builder, saved);
   ctx->arg_slots[index] = slot;
   return slot;
}

static LLVMTypeRef
get_llvm_type(struct ntl_context *ctx, unsigned bit_size, unsigned num_components)
{
   LLVMTypeRef base;
   switch (bit_size) {
   case 1:  base = LLVMInt1TypeInContext(ctx->llvm_ctx); break;
   case 8:  base = LLVMInt8TypeInContext(ctx->llvm_ctx); break;
   case 16: base = LLVMInt16TypeInContext(ctx->llvm_ctx); break;
   case 32: base = LLVMInt32TypeInContext(ctx->llvm_ctx); break;
   case 64: base = LLVMInt64TypeInContext(ctx->llvm_ctx); break;
   default: base = LLVMInt32TypeInContext(ctx->llvm_ctx); break;
   }
   if (num_components == 1)
      return base;
   return LLVMVectorType(base, num_components);
}

static LLVMTypeRef
get_float_type(struct ntl_context *ctx, unsigned bit_size)
{
   switch (bit_size) {
   case 16: return LLVMHalfTypeInContext(ctx->llvm_ctx);
   case 32: return LLVMFloatTypeInContext(ctx->llvm_ctx);
   case 64: return LLVMDoubleTypeInContext(ctx->llvm_ctx);
   default: return LLVMFloatTypeInContext(ctx->llvm_ctx);
   }
}

/*
 * Reinterpret a scalar as `type`.
 *
 * Values here carry no int/float tag — a float can arrive as the integer its
 * bits spell — so same-width conversions are bitcasts, not numeric casts.
 * Booleans are the one real width change, and widen by zero-extension.
 */
static LLVMValueRef
coerce_to_type(struct ntl_context *ctx, LLVMValueRef value, LLVMTypeRef type)
{
   LLVMTypeRef from = LLVMTypeOf(value);
   if (from == type)
      return value;

   unsigned from_bits = LLVMGetTypeKind(from) == LLVMIntegerTypeKind
      ? LLVMGetIntTypeWidth(from) : (unsigned)LLVMSizeOfTypeInBits(
           LLVMGetModuleDataLayout(ctx->module), from);
   unsigned to_bits = LLVMGetTypeKind(type) == LLVMIntegerTypeKind
      ? LLVMGetIntTypeWidth(type) : (unsigned)LLVMSizeOfTypeInBits(
           LLVMGetModuleDataLayout(ctx->module), type);

   if (from_bits != to_bits) {
      LLVMTypeRef int_type = LLVMIntTypeInContext(ctx->llvm_ctx, to_bits);
      value = from_bits < to_bits
         ? LLVMBuildZExt(ctx->builder, value, int_type, "")
         : LLVMBuildTrunc(ctx->builder, value, int_type, "");
      if (LLVMTypeOf(value) == type)
         return value;
   }

   return LLVMBuildBitCast(ctx->builder, value, type, "");
}

static LLVMValueRef
get_ssa_def(struct ntl_context *ctx, nir_def *def)
{
   assert(def->index < ctx->num_ssa_defs);
   return ctx->ssa_defs[def->index];
}

static void
set_ssa_def(struct ntl_context *ctx, nir_def *def, LLVMValueRef val)
{
   assert(def->index < ctx->num_ssa_defs);
   ctx->ssa_defs[def->index] = val;
}

static LLVMValueRef
get_src(struct ntl_context *ctx, nir_src *src)
{
   LLVMValueRef val = get_ssa_def(ctx, src->ssa);
   if (!val) {
      unsigned bit_size = src->ssa->bit_size;
      unsigned num_comp = src->ssa->num_components;
      val = LLVMConstNull(get_llvm_type(ctx, bit_size, num_comp));
   }
   return val;
}

static LLVMValueRef
emit_nvptx_read_sreg(struct ntl_context *ctx, const char *name)
{
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef fn_type = LLVMFunctionType(i32, NULL, 0, false);
   LLVMValueRef fn = LLVMGetNamedFunction(ctx->module, name);
   if (!fn) {
      fn = LLVMAddFunction(ctx->module, name, fn_type);
      LLVMSetLinkage(fn, LLVMExternalLinkage);
   }
   return LLVMBuildCall2(ctx->builder, fn_type, fn, NULL, 0, "");
}

static LLVMValueRef
emit_workgroup_id(struct ntl_context *ctx, unsigned component)
{
   const char *names[] = {
      "llvm.nvvm.read.ptx.sreg.ctaid.x",
      "llvm.nvvm.read.ptx.sreg.ctaid.y",
      "llvm.nvvm.read.ptx.sreg.ctaid.z",
   };
   /* Inside a drawing stage's grid-stride loop the block id is virtual — the
    * loop's induction value — so every slot-derived address follows the loop.
    * See the kernel skeleton in emit_function_body(). */
   if (component == 0 && ctx->virtual_bid)
      return ctx->virtual_bid;
   return emit_nvptx_read_sreg(ctx, names[component]);
}

static LLVMValueRef
emit_local_invocation_id(struct ntl_context *ctx, unsigned component)
{
   const char *names[] = {
      "llvm.nvvm.read.ptx.sreg.tid.x",
      "llvm.nvvm.read.ptx.sreg.tid.y",
      "llvm.nvvm.read.ptx.sreg.tid.z",
   };
   return emit_nvptx_read_sreg(ctx, names[component]);
}

/*
 * The base pointer of the stage's constant buffer number `slot`.
 *
 * Every read of args[18..] in the generated code goes through here — the index
 * form of load_ubo/load_ssbo below, and load_const_buf_base_addr_cudapipe,
 * which is how a lavapipe-lowered shader actually reaches a uniform block. That
 * is what makes it one function rather than a line in each of them, and what
 * makes `reads_const_bufs` trustworthy.
 *
 * Both drawing stages read through the batch table, so that consecutive draws
 * differing only in their uniforms can share one launch; see
 * CP_ARG_SLOT_UBO_TABLE. A single draw points the table at args[18], the mask
 * at a zero and the row array at one word holding zero, so the row index is
 * zero and this computes the very address the plain form does — there is no
 * second variant of the shader and no branch.
 *
 * The two stages differ only in who fills the row array. The vertex stage's is
 * written by cp_vertex_fetch, which searches the batch's slices to know what
 * to gather and writes the answer down rather than have the shader repeat it.
 * The fragment stage's is written by the interpolator, which is where a shaded
 * slot and the primitive that won it are both in hand — a pixel has no draw of
 * its own. Compute reads args[18 + i] directly; it has no batch.
 */
/*
 * Which draw of the batch this thread belongs to. The vertex stage's row
 * array is written by cp_vertex_fetch, which searches the batch's slices to
 * know what to gather and writes the answer down rather than have the shader
 * repeat it; the fragment stage's by the interpolator, which is where a
 * shaded slot and the primitive that won it are both in hand. A single draw
 * points the row array at one word holding zero and masks the index to zero,
 * so this is row zero there without a branch.
 */
static LLVMValueRef
emit_batch_row(struct ntl_context *ctx)
{
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);

   assert(ctx->nir->info.stage == MESA_SHADER_VERTEX ||
          ctx->nir->info.stage == MESA_SHADER_FRAGMENT);

   LLVMValueRef mask = LLVMBuildLoad2(ctx->builder, i32,
      LLVMBuildBitCast(ctx->builder, cp_arg_slot(ctx, CP_ARG_SLOT_BATCH_MASK),
                       LLVMPointerType(i32, 0), ""), "batch_mask");
   LLVMSetMetadata(mask, ctx->md_invariant_load,
                   LLVMMDNodeInContext(ctx->llvm_ctx, NULL, 0));

   LLVMValueRef tid = LLVMBuildAdd(ctx->builder,
      LLVMBuildMul(ctx->builder, emit_workgroup_id(ctx, 0),
                   LLVMConstInt(i32, 256, false), ""),
      emit_local_invocation_id(ctx, 0), "");

   LLVMValueRef rows = LLVMBuildBitCast(ctx->builder,
      cp_arg_slot(ctx, CP_ARG_SLOT_BATCH_ROWS), LLVMPointerType(i32, 0),
      "batch_rows");
   LLVMValueRef ridx = LLVMBuildZExt(ctx->builder,
      LLVMBuildAnd(ctx->builder, tid, mask, ""), i64, "");
   LLVMValueRef row = LLVMBuildLoad2(ctx->builder, i32,
      LLVMBuildGEP2(ctx->builder, i32, rows, &ridx, 1, ""), "batch_draw");
   LLVMSetMetadata(row, ctx->md_invariant_load,
                   LLVMMDNodeInContext(ctx->llvm_ctx, NULL, 0));
   return row;
}

static LLVMValueRef
emit_const_buf_base(struct ntl_context *ctx, LLVMValueRef slot)
{
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
   LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);

   ctx->reads_const_bufs = true;

   if (ctx->nir->info.stage == MESA_SHADER_VERTEX ||
       ctx->nir->info.stage == MESA_SHADER_FRAGMENT) {
      LLVMValueRef table = LLVMBuildBitCast(ctx->builder,
         cp_arg_slot(ctx, CP_ARG_SLOT_UBO_TABLE), ptr_ptr_type, "ubo_table");

      /* Which draw of the batch this vertex came from, decided by
       * cp_vertex_fetch and read back rather than recomputed. Masked to zero
       * for a single draw, which is the one-word array below it. */
      LLVMValueRef row = emit_batch_row(ctx);

      LLVMValueRef off = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, row,
                      LLVMConstInt(i32, CP_ARG_UBO_STRIDE, false), ""),
         slot, "");
      off = LLVMBuildZExt(ctx->builder, off, i64, "");
      return LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, table, &off, 1, ""), "buf_base");
   }

   LLVMValueRef args = LLVMBuildBitCast(ctx->builder, ctx->kernel_args[0],
                                        ptr_ptr_type, "");
   LLVMValueRef idx = LLVMBuildAdd(ctx->builder, slot,
                                   LLVMConstInt(i32, CP_ARG_UBO_BASE, false), "");
   idx = LLVMBuildZExt(ctx->builder, idx, i64, "");
   return LLVMBuildLoad2(ctx->builder, ptr_type,
      LLVMBuildGEP2(ctx->builder, ptr_type, args, &idx, 1, ""), "buf_base");
}

/*
 * The base pointer of the buffer a load_ubo/load_ssbo addresses.
 *
 * Two forms reach us, distinguished by the width of the source — the same
 * split lp_llvm_buffer_member() makes:
 *
 *   64 bit: the address of an lp_jit_buffer descriptor, whose first member is
 *           the base pointer. One dereference.
 *   32 bit: an index into the stage's constant buffer array, resolved above.
 *
 * Push constants arrive as the index form with index 0, which is why missing
 * this case faulted every shader that used them.
 */
static LLVMValueRef
emit_buffer_base(struct ntl_context *ctx, nir_src *src)
{
   LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
   LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
   LLVMValueRef val = get_src(ctx, src);

   if (nir_src_bit_size(*src) == 64) {
      LLVMValueRef desc_ptr = LLVMBuildIntToPtr(ctx->builder, val, ptr_ptr_type, "");
      return LLVMBuildLoad2(ctx->builder, ptr_type, desc_ptr, "buf_base");
   }

   return emit_const_buf_base(ctx, val);
}

/*
 * `discard`. One thread shades one covered pixel, so a discarded fragment is
 * recorded in a mask that cp_fs_writeback consults instead of being killed
 * outright — the shader has already been laid out to run to completion.
 *
 * Visibility was resolved before the shader ran, so a discarded fragment can
 * still have occluded another fragment of the same draw. Alpha-tested geometry
 * that overlaps itself is therefore not exact; separate draws are fine.
 */
static void
emit_terminate(struct ntl_context *ctx, LLVMValueRef cond)
{
   LLVMTypeRef i8 = LLVMInt8TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef ptr_type = LLVMPointerType(i8, 0);
   LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);

   LLVMValueRef args = LLVMBuildBitCast(ctx->builder, ctx->kernel_args[0],
                                        ptr_ptr_type, "");
   LLVMValueRef mask = LLVMBuildLoad2(ctx->builder, ptr_type,
      LLVMBuildGEP2(ctx->builder, ptr_type, args,
         &(LLVMValueRef){LLVMConstInt(i64, CP_ARG_SLOT_DISCARD, false)}, 1, ""),
      "discard_mask");

   LLVMValueRef bid = emit_workgroup_id(ctx, 0);
   LLVMValueRef tid = emit_local_invocation_id(ctx, 0);
   LLVMValueRef idx = LLVMBuildAdd(ctx->builder,
      LLVMBuildMul(ctx->builder, bid, LLVMConstInt(i32, 256, false), ""),
      tid, "");
   idx = LLVMBuildZExt(ctx->builder, idx, i64, "");

   LLVMValueRef slot = LLVMBuildGEP2(ctx->builder, i8, mask, &idx, 1, "");

   /* Store 1 where the condition holds, leaving the mask untouched otherwise,
    * so no branch is needed around the store. */
   LLVMValueRef old = LLVMBuildLoad2(ctx->builder, i8, slot, "");
   LLVMValueRef set = LLVMBuildSelect(ctx->builder, cond,
                                      LLVMConstInt(i8, 1, false), old, "");
   LLVMBuildStore(ctx->builder, set, slot);
}

/*
 * Storage image formats. NIR keeps the format on the intrinsic and hands the
 * shader whatever type it asked for, so the driver is responsible both for the
 * texel stride and for packing to and from it.
 */
static bool
cp_image_format_is_unorm8(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
      return true;
   default:
      return false;
   }
}

static unsigned
cp_image_format_size(enum pipe_format format, unsigned fallback_bits)
{
   unsigned size = format != PIPE_FORMAT_NONE
      ? util_format_get_blocksize(format) : 0;
   return size ? size : MAX2(fallback_bits / 8, 1u);
}

/*
 * Report, once per name, that the backend is about to emit undef.
 *
 * Every caller of this is a hole in the backend that produces undef and lets
 * it propagate through everything downstream, which is the failure mode that
 * cost the most time in this driver: load_front_face was missing, so
 * gl_FrontFacing was undef, so every surface in a captured application lost
 * its direct lighting, and the frames looked merely dim rather than wrong.
 * 1c0ad4094fa made the intrinsic case say so unconditionally; the two ALU
 * cases below are the same bug wearing a different hat and were missed.
 *
 * This is compile time, not draw time — a few lines for a shader that will
 * render wrong. Names come from NIR's static info tables, so comparing the
 * pointer is an exact test and costs nothing. Past 32 distinct names the
 * dedup gives up and repeats, which is the right way to degrade: something is
 * very wrong by then and the first lines have already been printed.
 */
static void
warn_undef_once(const char *what, const char *name)
{
   static const char *said[32];
   static unsigned num_said;

   for (unsigned i = 0; i < num_said; i++)
      if (said[i] == name)
         return;

   if (num_said < ARRAY_SIZE(said))
      said[num_said++] = name;

   fprintf(stderr, "cudapipe: %s '%s' is not implemented — the shader using "
           "it computes on undef and will render wrong.\n", what, name);
}

static void
emit_intrinsic(struct ntl_context *ctx, nir_intrinsic_instr *instr)
{
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);

   switch (instr->intrinsic) {
   case nir_intrinsic_load_workgroup_id: {
      LLVMValueRef x = emit_workgroup_id(ctx, 0);
      LLVMValueRef y = emit_workgroup_id(ctx, 1);
      LLVMValueRef z = emit_workgroup_id(ctx, 2);
      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(i32, 3));
      vec = LLVMBuildInsertElement(ctx->builder, vec, x, LLVMConstInt(i32, 0, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, y, LLVMConstInt(i32, 1, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, z, LLVMConstInt(i32, 2, false), "");
      set_ssa_def(ctx, &instr->def, vec);
      break;
   }
   case nir_intrinsic_load_local_invocation_id: {
      LLVMValueRef x = emit_local_invocation_id(ctx, 0);
      LLVMValueRef y = emit_local_invocation_id(ctx, 1);
      LLVMValueRef z = emit_local_invocation_id(ctx, 2);
      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(i32, 3));
      vec = LLVMBuildInsertElement(ctx->builder, vec, x, LLVMConstInt(i32, 0, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, y, LLVMConstInt(i32, 1, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, z, LLVMConstInt(i32, 2, false), "");
      set_ssa_def(ctx, &instr->def, vec);
      break;
   }
   case nir_intrinsic_load_num_workgroups: {
      /* arg_ptrs[0] = pointer to {gridX, gridY, gridZ} */
      LLVMValueRef grid_ptr = cp_arg_slot(ctx, 0);
      /* Cast to i32* and load x, y, z */
      LLVMTypeRef i32_ptr = LLVMPointerType(i32, 0);
      LLVMValueRef grid_i32 = LLVMBuildBitCast(ctx->builder, grid_ptr, i32_ptr, "");
      LLVMValueRef x = LLVMBuildLoad2(ctx->builder, i32, grid_i32, "grid_x");
      LLVMValueRef y = LLVMBuildLoad2(ctx->builder, i32,
         LLVMBuildGEP2(ctx->builder, i32, grid_i32, (LLVMValueRef[]){LLVMConstInt(i32, 1, false)}, 1, ""), "grid_y");
      LLVMValueRef z = LLVMBuildLoad2(ctx->builder, i32,
         LLVMBuildGEP2(ctx->builder, i32, grid_i32, (LLVMValueRef[]){LLVMConstInt(i32, 2, false)}, 1, ""), "grid_z");
      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(i32, 3));
      vec = LLVMBuildInsertElement(ctx->builder, vec, x, LLVMConstInt(i32, 0, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, y, LLVMConstInt(i32, 1, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, z, LLVMConstInt(i32, 2, false), "");
      set_ssa_def(ctx, &instr->def, vec);
      break;
   }
   case nir_intrinsic_load_workgroup_size: {
      uint16_t *ws = ctx->nir->info.workgroup_size;
      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(i32, 3));
      vec = LLVMBuildInsertElement(ctx->builder, vec, LLVMConstInt(i32, ws[0], false), LLVMConstInt(i32, 0, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, LLVMConstInt(i32, ws[1], false), LLVMConstInt(i32, 1, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, LLVMConstInt(i32, ws[2], false), LLVMConstInt(i32, 2, false), "");
      set_ssa_def(ctx, &instr->def, vec);
      break;
   }
   case nir_intrinsic_load_local_invocation_index: {
      /* linearIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y */
      LLVMValueRef tid_x = emit_local_invocation_id(ctx, 0);
      LLVMValueRef tid_y = emit_local_invocation_id(ctx, 1);
      LLVMValueRef tid_z = emit_local_invocation_id(ctx, 2);
      uint16_t *ws = ctx->nir->info.workgroup_size;
      LLVMValueRef bx = LLVMConstInt(i32, ws[0], false);
      LLVMValueRef bxy = LLVMConstInt(i32, ws[0] * ws[1], false);
      LLVMValueRef idx = LLVMBuildAdd(ctx->builder, tid_x,
         LLVMBuildAdd(ctx->builder,
            LLVMBuildMul(ctx->builder, tid_y, bx, ""),
            LLVMBuildMul(ctx->builder, tid_z, bxy, ""), ""), "local_idx");
      set_ssa_def(ctx, &instr->def, idx);
      break;
   }
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_vertex_id_zero_base: {
      /* Read original vertex_id from args[5] array indexed by thread_id */
      LLVMValueRef bid = emit_workgroup_id(ctx, 0);
      LLVMValueRef tid = emit_local_invocation_id(ctx, 0);
      LLVMValueRef bs = LLVMConstInt(i32, 256, false);
      LLVMValueRef thread_id = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, bid, bs, ""), tid, "");
      LLVMValueRef vid_arr_ptr = cp_arg_slot(ctx, 5);
      LLVMValueRef vid_ptr = LLVMBuildGEP2(ctx->builder, i32,
         LLVMBuildBitCast(ctx->builder, vid_arr_ptr, LLVMPointerType(i32, 0), ""),
         &thread_id, 1, "");
      LLVMValueRef vid = LLVMBuildLoad2(ctx->builder, i32, vid_ptr, "vertex_id");
      LLVMSetAlignment(vid, 4);
      set_ssa_def(ctx, &instr->def, vid);
      break;
   }
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
      /* The fragment stage runs one thread per pixel with no quad neighbours,
       * so a general derivative isn't available. Mip selection doesn't rely on
       * this — it uses the varying derivatives computed during interpolation. */
      set_ssa_def(ctx, &instr->def,
                  LLVMConstNull(get_llvm_type(ctx, instr->def.bit_size,
                                              instr->def.num_components)));
      break;
   case nir_intrinsic_load_front_face: {
      /*
       * gl_FrontFacing. A byte per shaded slot at CP_ARG_SLOT_FRONT_FACE,
       * written by cp_interp_pixel from the sign of the primitive's
       * screen-space area — which is the only place the primitive that won a
       * slot is still known. Indexed by the thread id, the way every other
       * per-slot array here is.
       *
       * Undef until this existed, which is not a value a shader survives:
       * the surface shaders flip their normal by it, so an undefined facing
       * costs the whole direct-light term and swaps which hemisphere of the
       * ambient probe is read.
       */
      LLVMValueRef bid = emit_workgroup_id(ctx, 0);
      LLVMValueRef tid = emit_local_invocation_id(ctx, 0);
      LLVMValueRef thread_id = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, bid, LLVMConstInt(i32, 256, false), ""), tid, "");
      LLVMTypeRef i8 = LLVMInt8TypeInContext(ctx->llvm_ctx);
      LLVMValueRef arr = LLVMBuildBitCast(ctx->builder,
         cp_arg_slot(ctx, CP_ARG_SLOT_FRONT_FACE), LLVMPointerType(i8, 0), "");
      LLVMValueRef elem = LLVMBuildGEP2(ctx->builder, i8, arr, &thread_id, 1, "");
      LLVMValueRef face = LLVMBuildLoad2(ctx->builder, i8, elem, "front_face");
      LLVMSetAlignment(face, 1);
      set_ssa_def(ctx, &instr->def,
                  LLVMBuildICmp(ctx->builder, LLVMIntNE, face,
                                LLVMConstInt(i8, 0, false), ""));
      break;
   }
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_draw_id: {
      /* Draw parameters: one row of CP_ARG_DRAW_PARAM_STRIDE uint32 per draw
       * of the batch at args[7], selected by the batch row — so draws that
       * disagree in them can still merge. An unbatched draw has one row and
       * row zero, which reads exactly as the flat triple this used to be.
       * base_vertex is its own field rather than an alias of first_vertex,
       * because it is defined to read zero for a non-indexed draw where
       * first_vertex reads the draw's start. */
      unsigned field =
         instr->intrinsic == nir_intrinsic_load_base_instance ? 1 :
         instr->intrinsic == nir_intrinsic_load_draw_id ? 2 :
         instr->intrinsic == nir_intrinsic_load_base_vertex ? 3 : 0;
      LLVMValueRef params = cp_arg_slot(ctx, 7);
      LLVMValueRef row = emit_batch_row(ctx);
      LLVMValueRef idx = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, row,
                      LLVMConstInt(i32, CP_ARG_DRAW_PARAM_STRIDE, false), ""),
         LLVMConstInt(i32, field, false), "");
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      idx = LLVMBuildZExt(ctx->builder, idx, i64, "");
      LLVMValueRef slot = LLVMBuildGEP2(ctx->builder, i32,
         LLVMBuildBitCast(ctx->builder, params, LLVMPointerType(i32, 0), ""),
         &idx, 1, "");
      LLVMValueRef value = LLVMBuildLoad2(ctx->builder, i32, slot, "draw_param");
      LLVMSetAlignment(value, 4);
      set_ssa_def(ctx, &instr->def, value);
      break;
   }
   case nir_intrinsic_load_instance_id: {
      /* Per-vertex instance index, from args[6]; laid out like the vertex-id
       * array above so the same indexing applies. */
      LLVMValueRef bid = emit_workgroup_id(ctx, 0);
      LLVMValueRef tid = emit_local_invocation_id(ctx, 0);
      LLVMValueRef thread_id = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, bid, LLVMConstInt(i32, 256, false), ""), tid, "");
      LLVMValueRef arr_ptr = cp_arg_slot(ctx, 6);
      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder, i32,
         LLVMBuildBitCast(ctx->builder, arr_ptr, LLVMPointerType(i32, 0), ""),
         &thread_id, 1, "");
      LLVMValueRef iid = LLVMBuildLoad2(ctx->builder, i32, elem_ptr, "instance_id");
      LLVMSetAlignment(iid, 4);
      set_ssa_def(ctx, &instr->def, iid);
      break;
   }
   case nir_intrinsic_load_input: {
      /* VS input: read from input buffer at (base + component) * 4 bytes per vertex.
       * Kernel arg layout: [0]=input_buffers_ptr, args as before.
       * For now, treat input as reading from the vertex's attribute slot. */
      unsigned base = nir_intrinsic_base(instr);
      unsigned comp = nir_intrinsic_component(instr);
      LLVMValueRef offset_val = get_src(ctx, &instr->src[0]);

      /* The classic/fused ABI reads global fs_in. The inline execution reads
       * the caller-owned entry alloca filled by the linked interpolation IR. */
      LLVMValueRef input_ptr =
         ctx->nir->info.stage == MESA_SHADER_FRAGMENT && ctx->inline_interp
         ? ctx->inline_fs_inputs : cp_arg_slot(ctx, 2);

      /* Classic inputs are a global array indexed by invocation. Inline
       * inputs are this invocation's one entry alloca, so no vid/stride term. */
      LLVMValueRef byte_off;
      if (ctx->nir->info.stage == MESA_SHADER_FRAGMENT && ctx->inline_interp) {
         byte_off = LLVMBuildAdd(ctx->builder,
            LLVMConstInt(i32, base * 16 + comp * 4, false),
            LLVMBuildMul(ctx->builder, offset_val,
                         LLVMConstInt(i32, 16, false), ""), "");
      } else {
         LLVMValueRef bid2 = emit_workgroup_id(ctx, 0);
         LLVMValueRef tid2 = emit_local_invocation_id(ctx, 0);
         LLVMValueRef vid = LLVMBuildAdd(ctx->builder,
            LLVMBuildMul(ctx->builder, bid2,
                         LLVMConstInt(i32, 256, false), ""), tid2, "");
         LLVMTypeRef i32_ptr = LLVMPointerType(i32, 0);
         LLVMValueRef stride = LLVMBuildLoad2(ctx->builder, i32,
            LLVMBuildBitCast(ctx->builder, cp_arg_slot(ctx, 3), i32_ptr, ""),
            "stride");
         LLVMSetMetadata(stride, ctx->md_invariant_load,
                         LLVMMDNodeInContext(ctx->llvm_ctx, NULL, 0));
         byte_off = LLVMBuildAdd(ctx->builder,
            LLVMBuildAdd(ctx->builder,
               LLVMBuildMul(ctx->builder, vid, stride, ""),
               LLVMConstInt(i32, base * 16 + comp * 4, false), ""),
            LLVMBuildMul(ctx->builder, offset_val,
                         LLVMConstInt(i32, 16, false), ""), "");
      }

      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), input_ptr, &byte_off, 1, "");

      unsigned nc = instr->def.num_components;
      unsigned bs2 = instr->def.bit_size;
      LLVMTypeRef load_type = get_llvm_type(ctx, bs2, nc);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(load_type, 0), "");
      LLVMValueRef val = LLVMBuildLoad2(ctx->builder, load_type, typed_ptr, "vs_in");
      /* Mark as unaligned — vertex stride may not be aligned to vec4 */
      LLVMSetAlignment(val, 4);
      set_ssa_def(ctx, &instr->def, val);
      break;
   }
   case nir_intrinsic_store_output: {
      /* VS output: write to output buffer at (base + component) * 4 bytes per vertex */
      LLVMValueRef data = get_src(ctx, &instr->src[0]);
      unsigned base = nir_intrinsic_base(instr);
      unsigned comp = nir_intrinsic_component(instr);
      LLVMValueRef offset_val = get_src(ctx, &instr->src[1]);

      /* Output ptr is at args[4] */
      LLVMValueRef output_ptr = cp_arg_slot(ctx, 4);

      /* vertex_id */
      LLVMValueRef bid2 = emit_workgroup_id(ctx, 0);
      LLVMValueRef tid2 = emit_local_invocation_id(ctx, 0);
      LLVMValueRef vid = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, bid2, LLVMConstInt(i32, 256, false), ""), tid2, "");

      /* Output stride = num_output_slots * 16 (packed vec4s) */
      unsigned num_outputs = ctx->nir->num_outputs;
      unsigned out_stride = num_outputs * 16;

      LLVMValueRef byte_off = LLVMBuildAdd(ctx->builder,
         LLVMBuildAdd(ctx->builder,
            LLVMBuildMul(ctx->builder, vid, LLVMConstInt(i32, out_stride, false), ""),
            LLVMConstInt(i32, base * 16 + comp * 4, false), ""),
         LLVMBuildMul(ctx->builder, offset_val, LLVMConstInt(i32, 16, false), ""), "");

      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), output_ptr, &byte_off, 1, "");
      LLVMTypeRef store_type = LLVMTypeOf(data);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(store_type, 0), "");
      LLVMBuildStore(ctx->builder, data, typed_ptr);
      break;
   }
   case nir_intrinsic_load_global_invocation_id: {
      LLVMValueRef wg_x = emit_workgroup_id(ctx, 0);
      LLVMValueRef wg_y = emit_workgroup_id(ctx, 1);
      LLVMValueRef wg_z = emit_workgroup_id(ctx, 2);
      LLVMValueRef li_x = emit_local_invocation_id(ctx, 0);
      LLVMValueRef li_y = emit_local_invocation_id(ctx, 1);
      LLVMValueRef li_z = emit_local_invocation_id(ctx, 2);
      uint16_t *ws = ctx->nir->info.workgroup_size;
      LLVMValueRef x = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, wg_x, LLVMConstInt(i32, ws[0], false), ""), li_x, "gid_x");
      LLVMValueRef y = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, wg_y, LLVMConstInt(i32, ws[1], false), ""), li_y, "gid_y");
      LLVMValueRef z = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, wg_z, LLVMConstInt(i32, ws[2], false), ""), li_z, "gid_z");
      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(i32, 3));
      vec = LLVMBuildInsertElement(ctx->builder, vec, x, LLVMConstInt(i32, 0, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, y, LLVMConstInt(i32, 1, false), "");
      vec = LLVMBuildInsertElement(ctx->builder, vec, z, LLVMConstInt(i32, 2, false), "");
      set_ssa_def(ctx, &instr->def, vec);
      break;
   }
   case nir_intrinsic_load_const_buf_base_addr_cudapipe: {
      /*
       * The 64-bit base address of constant buffer `src[0]`: how a shader
       * reaches a descriptor set. The address lands here and the load_ubo or
       * texture handle that follows dereferences the descriptor row at it.
       * emit_const_buf_base() owns the layout, including the per-draw table
       * the vertex stage reads through when draws are batched.
       */
      LLVMValueRef base = emit_const_buf_base(ctx, get_src(ctx, &instr->src[0]));
      LLVMValueRef addr = LLVMBuildPtrToInt(ctx->builder, base,
         LLVMInt64TypeInContext(ctx->llvm_ctx), "");
      set_ssa_def(ctx, &instr->def, addr);
      break;
   }
   case nir_intrinsic_get_ssbo_size: {
      /*
       * The bound range in bytes, which the descriptor row carries beside the
       * base: struct cpvk_descriptor { u64 base; u32 width_or_range; ... }.
       * NIR divides it by the array stride to answer `buffer.length()`.
       */
      LLVMValueRef desc_addr = get_src(ctx, &instr->src[0]);
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMValueRef offset8 = LLVMConstInt(i64, 8, false);
      LLVMValueRef addr_plus_8 = LLVMBuildAdd(ctx->builder, desc_addr, offset8, "");
      LLVMValueRef size_ptr = LLVMBuildIntToPtr(ctx->builder, addr_plus_8,
         LLVMPointerType(i32, 0), "");
      LLVMValueRef size = LLVMBuildLoad2(ctx->builder, i32, size_ptr, "ssbo_size");
      set_ssa_def(ctx, &instr->def, size);
      break;
   }
   case nir_intrinsic_load_ssbo: {
      /*
       * src[0] is a 64-bit address pointing at a descriptor row whose first
       * member is the buffer base. Read the base, then access base[offset].
       */
      LLVMValueRef buf_ptr = emit_buffer_base(ctx, &instr->src[0]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[1]);
      /* Access buf_ptr + byte_offset */
      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), buf_ptr, &byte_offset, 1, "");
      unsigned bit_size = instr->def.bit_size;
      unsigned num_comp = instr->def.num_components;
      LLVMTypeRef load_type = get_llvm_type(ctx, bit_size, num_comp);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(load_type, 0), "");
      LLVMValueRef val = LLVMBuildLoad2(ctx->builder, load_type, typed_ptr, "ssbo_load");
      set_ssa_def(ctx, &instr->def, val);
      break;
   }
   case nir_intrinsic_store_ssbo: {
      /* src[0] = data, src[1] = 64-bit descriptor address, src[2] = byte offset */
      LLVMValueRef data = get_src(ctx, &instr->src[0]);
      LLVMValueRef buf_ptr = emit_buffer_base(ctx, &instr->src[1]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[2]);
      /* Access buf_ptr + byte_offset */
      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), buf_ptr, &byte_offset, 1, "");
      LLVMTypeRef store_type = LLVMTypeOf(data);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(store_type, 0), "");
      LLVMBuildStore(ctx->builder, data, typed_ptr);
      break;
   }
   case nir_intrinsic_terminate:
      emit_terminate(ctx, LLVMConstInt(LLVMInt1TypeInContext(ctx->llvm_ctx), 1, false));
      break;
   case nir_intrinsic_terminate_if: {
      LLVMValueRef cond = get_src(ctx, &instr->src[0]);
      /* NIR conditions arrive as i32 booleans from this backend's lowering. */
      if (LLVMGetTypeKind(LLVMTypeOf(cond)) == LLVMIntegerTypeKind &&
          LLVMGetIntTypeWidth(LLVMTypeOf(cond)) != 1)
         cond = LLVMBuildICmp(ctx->builder, LLVMIntNE, cond,
                              LLVMConstInt(LLVMTypeOf(cond), 0, false), "");
      else if (LLVMGetTypeKind(LLVMTypeOf(cond)) == LLVMFloatTypeKind)
         cond = LLVMBuildFCmp(ctx->builder, LLVMRealONE, cond,
                              LLVMConstReal(LLVMTypeOf(cond), 0.0), "");
      emit_terminate(ctx, cond);
      break;
   }
   case nir_intrinsic_load_ubo: {
      /*
       * After lavapipe lowering, src[0] is a 64-bit descriptor address
       * pointing to lp_jit_buffer {ptr base, u32 num_elements}.
       * Same pattern as load_ssbo.
       */
      LLVMValueRef buf_ptr = emit_buffer_base(ctx, &instr->src[0]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[1]);
      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), buf_ptr, &byte_offset, 1, "");
      unsigned bit_size = instr->def.bit_size;
      unsigned num_comp = instr->def.num_components;
      LLVMTypeRef load_type = get_llvm_type(ctx, bit_size, num_comp);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(load_type, 0), "");
      LLVMValueRef val = LLVMBuildLoad2(ctx->builder, load_type, typed_ptr, "ubo_load");
      set_ssa_def(ctx, &instr->def, val);
      break;
   }
   case nir_intrinsic_decl_reg: {
      unsigned num_comp = nir_intrinsic_num_components(instr);
      unsigned bit_size = nir_intrinsic_bit_size(instr);
      LLVMTypeRef reg_type = get_llvm_type(ctx, bit_size, num_comp);
      /* Create alloca at current position - it'll be in the entry block
       * since decl_reg always appears first */
      LLVMValueRef alloca_val = LLVMBuildAlloca(ctx->builder, reg_type, "reg");
      LLVMBuildStore(ctx->builder, LLVMConstNull(reg_type), alloca_val);
      set_ssa_def(ctx, &instr->def, alloca_val);
      break;
   }
   case nir_intrinsic_load_reg: {
      LLVMValueRef reg_ptr = get_src(ctx, &instr->src[0]);
      if (!reg_ptr) {
         set_ssa_def(ctx, &instr->def, LLVMConstNull(
            get_llvm_type(ctx, instr->def.bit_size, instr->def.num_components)));
         break;
      }
      unsigned num_comp = instr->def.num_components;
      unsigned bit_size = instr->def.bit_size;
      LLVMTypeRef load_type = get_llvm_type(ctx, bit_size, num_comp);
      LLVMValueRef val = LLVMBuildLoad2(ctx->builder, load_type, reg_ptr, "reg_load");
      set_ssa_def(ctx, &instr->def, val);
      break;
   }
   case nir_intrinsic_store_reg: {
      LLVMValueRef val = get_src(ctx, &instr->src[0]);
      LLVMValueRef reg_ptr = get_src(ctx, &instr->src[1]);
      if (reg_ptr && val)
         LLVMBuildStore(ctx->builder, val, reg_ptr);
      break;
   }
   case nir_intrinsic_load_shared: {
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[0]);
      unsigned base = nir_intrinsic_base(instr);
      if (base != 0) {
         byte_offset = LLVMBuildAdd(ctx->builder, byte_offset,
            LLVMConstInt(i32, base, false), "");
      }
      /* Shared memory is address space 3 in NVPTX */
      LLVMTypeRef shared_ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 3);
      LLVMValueRef shared_base = LLVMConstNull(shared_ptr_type);
      LLVMValueRef ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), shared_base, &byte_offset, 1, "");
      unsigned bit_size = instr->def.bit_size;
      unsigned num_comp = instr->def.num_components;
      LLVMTypeRef load_type = get_llvm_type(ctx, bit_size, num_comp);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, ptr,
         LLVMPointerType(load_type, 3), "");
      LLVMValueRef val = LLVMBuildLoad2(ctx->builder, load_type, typed_ptr, "shared_load");
      set_ssa_def(ctx, &instr->def, val);
      break;
   }
   case nir_intrinsic_store_shared: {
      LLVMValueRef data = get_src(ctx, &instr->src[0]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[1]);
      unsigned base = nir_intrinsic_base(instr);
      if (base != 0) {
         byte_offset = LLVMBuildAdd(ctx->builder, byte_offset,
            LLVMConstInt(i32, base, false), "");
      }
      LLVMTypeRef shared_ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 3);
      LLVMValueRef shared_base = LLVMConstNull(shared_ptr_type);
      LLVMValueRef ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), shared_base, &byte_offset, 1, "");
      LLVMTypeRef store_type = LLVMTypeOf(data);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, ptr,
         LLVMPointerType(store_type, 3), "");
      LLVMBuildStore(ctx->builder, data, typed_ptr);
      break;
   }
   case nir_intrinsic_shared_atomic: {
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[0]);
      LLVMValueRef data = get_src(ctx, &instr->src[1]);
      unsigned base = nir_intrinsic_base(instr);
      if (base != 0) {
         byte_offset = LLVMBuildAdd(ctx->builder, byte_offset,
            LLVMConstInt(i32, base, false), "");
      }
      LLVMTypeRef shared_ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 3);
      LLVMValueRef shared_base = LLVMConstNull(shared_ptr_type);
      LLVMValueRef ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), shared_base, &byte_offset, 1, "");
      LLVMTypeRef val_type = LLVMTypeOf(data);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, ptr,
         LLVMPointerType(val_type, 3), "");
      nir_atomic_op op = nir_intrinsic_atomic_op(instr);
      LLVMAtomicRMWBinOp llvm_op;
      switch (op) {
      case nir_atomic_op_iadd: llvm_op = LLVMAtomicRMWBinOpAdd; break;
      case nir_atomic_op_iand: llvm_op = LLVMAtomicRMWBinOpAnd; break;
      case nir_atomic_op_ior:  llvm_op = LLVMAtomicRMWBinOpOr; break;
      case nir_atomic_op_ixor: llvm_op = LLVMAtomicRMWBinOpXor; break;
      case nir_atomic_op_imin: llvm_op = LLVMAtomicRMWBinOpMin; break;
      case nir_atomic_op_umin: llvm_op = LLVMAtomicRMWBinOpUMin; break;
      case nir_atomic_op_imax: llvm_op = LLVMAtomicRMWBinOpMax; break;
      case nir_atomic_op_umax: llvm_op = LLVMAtomicRMWBinOpUMax; break;
      case nir_atomic_op_xchg: llvm_op = LLVMAtomicRMWBinOpXchg; break;
      default: llvm_op = LLVMAtomicRMWBinOpAdd; break;
      }
      LLVMValueRef result = LLVMBuildAtomicRMW(ctx->builder, llvm_op, typed_ptr,
         data, LLVMAtomicOrderingMonotonic, false);
      if (nir_intrinsic_infos[instr->intrinsic].has_dest)
         set_ssa_def(ctx, &instr->def, result);
      break;
   }
   case nir_intrinsic_bindless_image_load: {
      /*
       * src[0] = 64-bit descriptor address (points to lp_image_descriptor)
       * src[1] = vec4 coordinate (x, y, z, w)
       * src[2] = sample index
       * src[3] = lod
       *
       * lp_jit_image layout at descriptor:
       *   offset 0:  base pointer (8 bytes)
       *   offset 8:  width (4 bytes)
       *   offset 12: height (2 bytes) + depth (2 bytes)
       *   offset 16: num_samples (1 byte) + pad (3 bytes)
       *   offset 20: sample_stride (4 bytes)
       *   offset 24: row_stride (4 bytes)
       *   offset 28: img_stride (4 bytes)
       *   offset 32: residency (8 bytes)
       *   offset 40: base_offset (4 bytes)
       */
      LLVMValueRef desc_addr = get_src(ctx, &instr->src[0]);
      LLVMValueRef coord = get_src(ctx, &instr->src[1]);
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);

      /* Load base pointer from offset 0 */
      LLVMValueRef base_ptr_ptr = LLVMBuildIntToPtr(ctx->builder, desc_addr,
         LLVMPointerType(ptr_type, 0), "");
      LLVMValueRef base_ptr = LLVMBuildLoad2(ctx->builder, ptr_type, base_ptr_ptr, "img_base");

      /* Load row_stride from offset 24 */
      LLVMValueRef stride_addr = LLVMBuildAdd(ctx->builder, desc_addr,
         LLVMConstInt(i64, 24, false), "");
      LLVMValueRef stride_ptr = LLVMBuildIntToPtr(ctx->builder, stride_addr,
         LLVMPointerType(i32, 0), "");
      LLVMValueRef row_stride = LLVMBuildLoad2(ctx->builder, i32, stride_ptr, "row_stride");

      /* Load base_offset from offset 40 */
      LLVMValueRef boff_addr = LLVMBuildAdd(ctx->builder, desc_addr,
         LLVMConstInt(i64, 40, false), "");
      LLVMValueRef boff_ptr = LLVMBuildIntToPtr(ctx->builder, boff_addr,
         LLVMPointerType(i32, 0), "");
      LLVMValueRef base_offset = LLVMBuildLoad2(ctx->builder, i32, boff_ptr, "base_off");

      /* Load img_stride from offset 28 */
      LLVMValueRef istride_addr = LLVMBuildAdd(ctx->builder, desc_addr,
         LLVMConstInt(i64, 28, false), "");
      LLVMValueRef istride_ptr = LLVMBuildIntToPtr(ctx->builder, istride_addr,
         LLVMPointerType(i32, 0), "");
      LLVMValueRef img_stride = LLVMBuildLoad2(ctx->builder, i32, istride_ptr, "img_stride");

      /* Get x, y, z from coordinate vector */
      LLVMValueRef x = LLVMBuildExtractElement(ctx->builder, coord,
         LLVMConstInt(i32, 0, false), "x");
      LLVMValueRef y = LLVMBuildExtractElement(ctx->builder, coord,
         LLVMConstInt(i32, 1, false), "y");
      LLVMValueRef z = LLVMBuildExtractElement(ctx->builder, coord,
         LLVMConstInt(i32, 2, false), "z");

      /*
       * Clamp to the image before touching memory. A convolution reads a
       * neighbourhood, so it addresses (-1, -1) at the first pixel; computing
       * that offset unclamped reads before the allocation and faults the
       * kernel, and a CUDA fault is sticky — every later launch and even
       * buffer allocation in the context fails with it, so one out of range
       * texel took down the rest of the frame.
       *
       * The clamp only keeps the address legal. The value has to read as zero,
       * which is what robustImageAccess requires and what the hardware does,
       * so it is selected back in below. Clamping the value too — returning
       * the edge texel — is a visible bug rather than a subtle one: it turns
       * the border of an emboss or edge detect filter into a frame around the
       * image, because a convolution whose outside taps repeat the edge comes
       * out flat where it should saturate.
       *
       * llvmpipe does the same in lp_build_sample_image_nearest(): force the
       * offset inside with an andnot, fetch, then select the out of bounds
       * value over the result.
       *
       * Descriptor layout: width is 32 bits at offset 8, height 16 bits at 12.
       */
      LLVMValueRef width = LLVMBuildLoad2(ctx->builder, i32,
         LLVMBuildIntToPtr(ctx->builder,
            LLVMBuildAdd(ctx->builder, desc_addr, LLVMConstInt(i64, 8, false), ""),
            LLVMPointerType(i32, 0), ""), "img_width");
      LLVMTypeRef i16 = LLVMInt16TypeInContext(ctx->llvm_ctx);
      LLVMValueRef height = LLVMBuildZExt(ctx->builder,
         LLVMBuildLoad2(ctx->builder, i16,
            LLVMBuildIntToPtr(ctx->builder,
               LLVMBuildAdd(ctx->builder, desc_addr, LLVMConstInt(i64, 12, false), ""),
               LLVMPointerType(i16, 0), ""), "img_height16"), i32, "img_height");

      LLVMValueRef zero_i = LLVMConstInt(i32, 0, false);
      LLVMValueRef one_i = LLVMConstInt(i32, 1, false);
      LLVMValueRef xmax = LLVMBuildSub(ctx->builder, width, one_i, "");
      LLVMValueRef ymax = LLVMBuildSub(ctx->builder, height, one_i, "");

      LLVMValueRef oob = LLVMBuildOr(ctx->builder,
         LLVMBuildOr(ctx->builder,
            LLVMBuildICmp(ctx->builder, LLVMIntSLT, x, zero_i, ""),
            LLVMBuildICmp(ctx->builder, LLVMIntSGT, x, xmax, ""), ""),
         LLVMBuildOr(ctx->builder,
            LLVMBuildICmp(ctx->builder, LLVMIntSLT, y, zero_i, ""),
            LLVMBuildICmp(ctx->builder, LLVMIntSGT, y, ymax, ""), ""), "img_oob");

      x = LLVMBuildSelect(ctx->builder,
             LLVMBuildICmp(ctx->builder, LLVMIntSLT, x, zero_i, ""), zero_i, x, "");
      y = LLVMBuildSelect(ctx->builder,
             LLVMBuildICmp(ctx->builder, LLVMIntSLT, y, zero_i, ""), zero_i, y, "");
      x = LLVMBuildSelect(ctx->builder,
             LLVMBuildICmp(ctx->builder, LLVMIntSGT, x, xmax, ""), xmax, x, "");
      y = LLVMBuildSelect(ctx->builder,
             LLVMBuildICmp(ctx->builder, LLVMIntSGT, y, ymax, ""), ymax, y, "");

      /* byte_offset = base_offset + z * img_stride + y * row_stride + x * pixel_size */
      unsigned bit_size = instr->def.bit_size;
      /*
       * The texel's size comes from the image format, not from the type the
       * shader wants back. NIR asks for a vec4 of float32 out of an 8 bit per
       * channel image, and taking the destination's width as the stride walks
       * the image four times too fast.
       */
      unsigned pixel_size = cp_image_format_size(nir_intrinsic_format(instr),
                                                 bit_size);
      LLVMValueRef offset_val = LLVMBuildAdd(ctx->builder, base_offset,
         LLVMBuildAdd(ctx->builder,
            LLVMBuildAdd(ctx->builder,
               LLVMBuildMul(ctx->builder, z, img_stride, ""),
               LLVMBuildMul(ctx->builder, y, row_stride, ""), ""),
            LLVMBuildMul(ctx->builder, x, LLVMConstInt(i32, pixel_size, false), ""), ""), "");

      /* Load pixel */
      LLVMValueRef pixel_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), base_ptr, &offset_val, 1, "");
      unsigned num_comp = instr->def.num_components;

      /*
       * An 8 bit per channel image has to be unpacked: the shader asked for
       * floats in [0, 1], and handing it the raw word leaves the emboss filter
       * convolving bit patterns.
       */
      if (cp_image_format_is_unorm8(nir_intrinsic_format(instr))) {
         LLVMTypeRef f32t = LLVMFloatTypeInContext(ctx->llvm_ctx);
         LLVMValueRef word = LLVMBuildLoad2(ctx->builder, i32,
            LLVMBuildBitCast(ctx->builder, pixel_ptr,
                             LLVMPointerType(i32, 0), ""), "img_word");
         word = LLVMBuildSelect(ctx->builder, oob,
                                LLVMConstInt(i32, 0, false), word, "");
         LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(f32t, num_comp));
         for (unsigned c = 0; c < num_comp; c++) {
            LLVMValueRef byte = LLVMBuildAnd(ctx->builder,
               LLVMBuildLShr(ctx->builder, word,
                             LLVMConstInt(i32, c * 8, false), ""),
               LLVMConstInt(i32, 0xFF, false), "");
            LLVMValueRef f = LLVMBuildFMul(ctx->builder,
               LLVMBuildUIToFP(ctx->builder, byte, f32t, ""),
               LLVMConstReal(f32t, 1.0 / 255.0), "");
            vec = LLVMBuildInsertElement(ctx->builder, vec, f,
                                         LLVMConstInt(i32, c, false), "");
         }
         set_ssa_def(ctx, &instr->def, num_comp > 1 ? vec :
            LLVMBuildExtractElement(ctx->builder, vec,
                                    LLVMConstInt(i32, 0, false), ""));
         break;
      }

      LLVMTypeRef pixel_type = get_llvm_type(ctx, bit_size, 1);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, pixel_ptr,
         LLVMPointerType(pixel_type, 0), "");
      LLVMValueRef pixel_val = LLVMBuildLoad2(ctx->builder, pixel_type, typed_ptr, "img_load");
      pixel_val = LLVMBuildSelect(ctx->builder, oob,
                                  LLVMConstNull(pixel_type), pixel_val, "");

      /* Return as vec4 (only .x is meaningful for r32 formats) */
      if (num_comp > 1) {
         LLVMValueRef vec = LLVMGetUndef(get_llvm_type(ctx, bit_size, num_comp));
         vec = LLVMBuildInsertElement(ctx->builder, vec, pixel_val,
            LLVMConstInt(i32, 0, false), "");
         for (unsigned c = 1; c < num_comp; c++)
            vec = LLVMBuildInsertElement(ctx->builder, vec, LLVMConstNull(pixel_type),
               LLVMConstInt(i32, c, false), "");
         set_ssa_def(ctx, &instr->def, vec);
      } else {
         set_ssa_def(ctx, &instr->def, pixel_val);
      }
      break;
   }
   case nir_intrinsic_bindless_image_store: {
      /*
       * src[0] = descriptor address
       * src[1] = vec4 coordinate
       * src[2] = sample
       * src[3] = data to store
       */
      LLVMValueRef desc_addr = get_src(ctx, &instr->src[0]);
      LLVMValueRef coord = get_src(ctx, &instr->src[1]);
      LLVMValueRef data = get_src(ctx, &instr->src[3]);
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);

      LLVMValueRef base_ptr_ptr = LLVMBuildIntToPtr(ctx->builder, desc_addr,
         LLVMPointerType(ptr_type, 0), "");
      LLVMValueRef base_ptr = LLVMBuildLoad2(ctx->builder, ptr_type, base_ptr_ptr, "img_base");

      LLVMValueRef stride_addr = LLVMBuildAdd(ctx->builder, desc_addr,
         LLVMConstInt(i64, 24, false), "");
      LLVMValueRef stride_ptr = LLVMBuildIntToPtr(ctx->builder, stride_addr,
         LLVMPointerType(i32, 0), "");
      LLVMValueRef row_stride = LLVMBuildLoad2(ctx->builder, i32, stride_ptr, "row_stride");

      LLVMValueRef boff_addr = LLVMBuildAdd(ctx->builder, desc_addr,
         LLVMConstInt(i64, 40, false), "");
      LLVMValueRef boff_ptr = LLVMBuildIntToPtr(ctx->builder, boff_addr,
         LLVMPointerType(i32, 0), "");
      LLVMValueRef base_offset = LLVMBuildLoad2(ctx->builder, i32, boff_ptr, "base_off");

      LLVMValueRef istride_addr2 = LLVMBuildAdd(ctx->builder, desc_addr,
         LLVMConstInt(i64, 28, false), "");
      LLVMValueRef istride_ptr2 = LLVMBuildIntToPtr(ctx->builder, istride_addr2,
         LLVMPointerType(i32, 0), "");
      LLVMValueRef img_stride2 = LLVMBuildLoad2(ctx->builder, i32, istride_ptr2, "img_stride");

      LLVMValueRef x = LLVMBuildExtractElement(ctx->builder, coord,
         LLVMConstInt(i32, 0, false), "x");
      LLVMValueRef y = LLVMBuildExtractElement(ctx->builder, coord,
         LLVMConstInt(i32, 1, false), "y");
      LLVMValueRef z = LLVMBuildExtractElement(ctx->builder, coord,
         LLVMConstInt(i32, 2, false), "z");

      /* An 8 bit per channel image takes a packed word, not the shader's
       * float vector; storing one component of that wrote a quarter of the
       * texel and left the rest of the image untouched. */
      bool pack_unorm8 = cp_image_format_is_unorm8(nir_intrinsic_format(instr)) &&
                         LLVMGetTypeKind(LLVMTypeOf(data)) == LLVMVectorTypeKind;
      LLVMValueRef store_val = data;

      if (pack_unorm8) {
         LLVMTypeRef f32t = LLVMFloatTypeInContext(ctx->llvm_ctx);
         unsigned comps = LLVMGetVectorSize(LLVMTypeOf(data));
         LLVMValueRef word = LLVMConstInt(i32, 0, false);
         for (unsigned c = 0; c < comps && c < 4; c++) {
            LLVMValueRef f = LLVMBuildExtractElement(ctx->builder, data,
               LLVMConstInt(i32, c, false), "");
            /* Clamp before scaling so an out of range value saturates instead
             * of wrapping into a neighbouring channel. */
            LLVMValueRef lo = LLVMConstReal(f32t, 0.0);
            LLVMValueRef hi = LLVMConstReal(f32t, 1.0);
            f = LLVMBuildSelect(ctx->builder,
                   LLVMBuildFCmp(ctx->builder, LLVMRealOLT, f, lo, ""), lo, f, "");
            f = LLVMBuildSelect(ctx->builder,
                   LLVMBuildFCmp(ctx->builder, LLVMRealOGT, f, hi, ""), hi, f, "");
            LLVMValueRef b = LLVMBuildFPToUI(ctx->builder,
               LLVMBuildFAdd(ctx->builder,
                  LLVMBuildFMul(ctx->builder, f, LLVMConstReal(f32t, 255.0), ""),
                  LLVMConstReal(f32t, 0.5), ""), i32, "");
            word = LLVMBuildOr(ctx->builder, word,
               LLVMBuildShl(ctx->builder, b,
                            LLVMConstInt(i32, c * 8, false), ""), "");
         }
         store_val = word;
      } else if (LLVMGetTypeKind(LLVMTypeOf(data)) == LLVMVectorTypeKind) {
         store_val = LLVMBuildExtractElement(ctx->builder, data,
            LLVMConstInt(i32, 0, false), "");
      }

      unsigned pixel_size = cp_image_format_size(nir_intrinsic_format(instr),
         LLVMGetTypeKind(LLVMTypeOf(store_val)) == LLVMIntegerTypeKind
            ? LLVMGetIntTypeWidth(LLVMTypeOf(store_val)) : 32);

      LLVMValueRef offset_val = LLVMBuildAdd(ctx->builder, base_offset,
         LLVMBuildAdd(ctx->builder,
            LLVMBuildAdd(ctx->builder,
               LLVMBuildMul(ctx->builder, z, img_stride2, ""),
               LLVMBuildMul(ctx->builder, y, row_stride, ""), ""),
            LLVMBuildMul(ctx->builder, x, LLVMConstInt(i32, pixel_size, false), ""), ""), "");

      LLVMValueRef pixel_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), base_ptr, &offset_val, 1, "");
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, pixel_ptr,
         LLVMPointerType(LLVMTypeOf(store_val), 0), "");
      LLVMBuildStore(ctx->builder, store_val, typed_ptr);
      break;
   }
   case nir_intrinsic_bindless_image_atomic: {
      /* src[0]=desc, src[1]=coord, src[2]=sample, src[3]=data */
      LLVMValueRef desc_addr = get_src(ctx, &instr->src[0]);
      LLVMValueRef coord = get_src(ctx, &instr->src[1]);
      LLVMValueRef data = get_src(ctx, &instr->src[3]);
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMValueRef base_ptr_ptr = LLVMBuildIntToPtr(ctx->builder, desc_addr,
         LLVMPointerType(ptr_type, 0), "");
      LLVMValueRef base_ptr = LLVMBuildLoad2(ctx->builder, ptr_type, base_ptr_ptr, "img_base");
      LLVMValueRef stride_addr = LLVMBuildAdd(ctx->builder, desc_addr, LLVMConstInt(i64, 24, false), "");
      LLVMValueRef stride_ptr = LLVMBuildIntToPtr(ctx->builder, stride_addr, LLVMPointerType(i32, 0), "");
      LLVMValueRef row_stride = LLVMBuildLoad2(ctx->builder, i32, stride_ptr, "row_stride");
      LLVMValueRef boff_addr = LLVMBuildAdd(ctx->builder, desc_addr, LLVMConstInt(i64, 40, false), "");
      LLVMValueRef boff_ptr = LLVMBuildIntToPtr(ctx->builder, boff_addr, LLVMPointerType(i32, 0), "");
      LLVMValueRef base_offset = LLVMBuildLoad2(ctx->builder, i32, boff_ptr, "base_off");
      LLVMValueRef istride_addr3 = LLVMBuildAdd(ctx->builder, desc_addr, LLVMConstInt(i64, 28, false), "");
      LLVMValueRef istride_ptr3 = LLVMBuildIntToPtr(ctx->builder, istride_addr3, LLVMPointerType(i32, 0), "");
      LLVMValueRef img_stride3 = LLVMBuildLoad2(ctx->builder, i32, istride_ptr3, "img_stride");
      LLVMValueRef x = LLVMBuildExtractElement(ctx->builder, coord, LLVMConstInt(i32, 0, false), "x");
      LLVMValueRef y = LLVMBuildExtractElement(ctx->builder, coord, LLVMConstInt(i32, 1, false), "y");
      LLVMValueRef z = LLVMBuildExtractElement(ctx->builder, coord, LLVMConstInt(i32, 2, false), "z");
      unsigned pixel_size = instr->def.bit_size / 8;
      LLVMValueRef offset_val = LLVMBuildAdd(ctx->builder, base_offset,
         LLVMBuildAdd(ctx->builder,
            LLVMBuildAdd(ctx->builder,
               LLVMBuildMul(ctx->builder, z, img_stride3, ""),
               LLVMBuildMul(ctx->builder, y, row_stride, ""), ""),
            LLVMBuildMul(ctx->builder, x, LLVMConstInt(i32, pixel_size, false), ""), ""), "");
      LLVMValueRef pixel_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), base_ptr, &offset_val, 1, "");
      LLVMTypeRef val_type = LLVMTypeOf(data);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, pixel_ptr,
         LLVMPointerType(val_type, 0), "");
      nir_atomic_op op = nir_intrinsic_atomic_op(instr);
      LLVMAtomicRMWBinOp llvm_op;
      switch (op) {
      case nir_atomic_op_iadd: llvm_op = LLVMAtomicRMWBinOpAdd; break;
      case nir_atomic_op_iand: llvm_op = LLVMAtomicRMWBinOpAnd; break;
      case nir_atomic_op_ior:  llvm_op = LLVMAtomicRMWBinOpOr; break;
      case nir_atomic_op_ixor: llvm_op = LLVMAtomicRMWBinOpXor; break;
      case nir_atomic_op_imin: llvm_op = LLVMAtomicRMWBinOpMin; break;
      case nir_atomic_op_umin: llvm_op = LLVMAtomicRMWBinOpUMin; break;
      case nir_atomic_op_imax: llvm_op = LLVMAtomicRMWBinOpMax; break;
      case nir_atomic_op_umax: llvm_op = LLVMAtomicRMWBinOpUMax; break;
      case nir_atomic_op_xchg: llvm_op = LLVMAtomicRMWBinOpXchg; break;
      default: llvm_op = LLVMAtomicRMWBinOpAdd; break;
      }
      LLVMValueRef result = LLVMBuildAtomicRMW(ctx->builder, llvm_op, typed_ptr,
         data, LLVMAtomicOrderingMonotonic, false);
      if (nir_intrinsic_infos[instr->intrinsic].has_dest)
         set_ssa_def(ctx, &instr->def, result);
      break;
   }
   case nir_intrinsic_ssbo_atomic: {
      /* src[0] = 64-bit descriptor address, src[1] = byte offset, src[2] = data */
      LLVMValueRef desc_addr = get_src(ctx, &instr->src[0]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[1]);
      LLVMValueRef data = get_src(ctx, &instr->src[2]);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      LLVMValueRef desc_ptr = LLVMBuildIntToPtr(ctx->builder, desc_addr, ptr_ptr_type, "");
      LLVMValueRef buf_ptr = LLVMBuildLoad2(ctx->builder, ptr_type, desc_ptr, "ssbo_base");
      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), buf_ptr, &byte_offset, 1, "");
      LLVMTypeRef val_type = LLVMTypeOf(data);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(val_type, 0), "");
      nir_atomic_op op = nir_intrinsic_atomic_op(instr);
      LLVMAtomicRMWBinOp llvm_op;
      switch (op) {
      case nir_atomic_op_iadd: llvm_op = LLVMAtomicRMWBinOpAdd; break;
      case nir_atomic_op_iand: llvm_op = LLVMAtomicRMWBinOpAnd; break;
      case nir_atomic_op_ior:  llvm_op = LLVMAtomicRMWBinOpOr; break;
      case nir_atomic_op_ixor: llvm_op = LLVMAtomicRMWBinOpXor; break;
      case nir_atomic_op_imin: llvm_op = LLVMAtomicRMWBinOpMin; break;
      case nir_atomic_op_umin: llvm_op = LLVMAtomicRMWBinOpUMin; break;
      case nir_atomic_op_imax: llvm_op = LLVMAtomicRMWBinOpMax; break;
      case nir_atomic_op_umax: llvm_op = LLVMAtomicRMWBinOpUMax; break;
      case nir_atomic_op_xchg: llvm_op = LLVMAtomicRMWBinOpXchg; break;
      default: llvm_op = LLVMAtomicRMWBinOpAdd; break;
      }
      LLVMValueRef result = LLVMBuildAtomicRMW(ctx->builder, llvm_op, typed_ptr,
         data, LLVMAtomicOrderingMonotonic, false);
      if (nir_intrinsic_infos[instr->intrinsic].has_dest)
         set_ssa_def(ctx, &instr->def, result);
      break;
   }
   case nir_intrinsic_barrier: {
      LLVMTypeRef void_type = LLVMVoidTypeInContext(ctx->llvm_ctx);
      LLVMTypeRef fn_type = LLVMFunctionType(void_type, NULL, 0, false);
      LLVMValueRef fn = LLVMGetNamedFunction(ctx->module, "llvm.nvvm.barrier0");
      if (!fn) {
         fn = LLVMAddFunction(ctx->module, "llvm.nvvm.barrier0", fn_type);
         LLVMSetLinkage(fn, LLVMExternalLinkage);
      }
      LLVMBuildCall2(ctx->builder, fn_type, fn, NULL, 0, "");
      break;
   }
   default: {
      /*
       * Unhandled intrinsic. The undef below propagates through everything it
       * feeds, so a missing intrinsic shows up as a shader that silently
       * computes nothing — and this used to say so only under
       * CUDAPIPE_DEBUG_SHADER, which nobody sets until they already suspect
       * the shader.
       *
       * That cost real time. load_front_face was missing, so gl_FrontFacing
       * was undef, so the tangent frame was flipped by an undefined sign, so
       * every surface in a captured application lost its direct lighting. The
       * frames looked plausible — dim and slightly green — and the driver knew
       * the whole time. Finding it took bisecting a frame to a single draw and
       * dumping every descriptor of it from two drivers.
       *
       * So say it once per intrinsic, always. This is compile time, not draw
       * time: a handful of lines for a shader that will render wrong, against
       * however long it takes to work that out from the picture.
       */
      warn_undef_once("intrinsic", nir_intrinsic_infos[instr->intrinsic].name);

      if (nir_intrinsic_infos[instr->intrinsic].has_dest) {
         unsigned num_comp = instr->def.num_components;
         unsigned bit_size = instr->def.bit_size;
         set_ssa_def(ctx, &instr->def,
                     LLVMGetUndef(get_llvm_type(ctx, bit_size, num_comp)));
      }
      break;
   }
   }
}

/*
 * Call a target-specific NVVM intrinsic, e.g. "llvm.nvvm.ex2.approx.f".
 *
 * The transcendentals have no generic LLVM lowering on NVPTX: llvm.exp2 and
 * friends turn into libcalls to exp2f, which doesn't exist on the device and
 * takes instruction selection down with it. The NVVM intrinsics map straight
 * onto the ex2/lg2/sin/cos approximation instructions the hardware has. They
 * are float-only and not overloaded, hence no type argument.
 */
static LLVMValueRef
build_nvvm_intrinsic(struct ntl_context *ctx, const char *name,
                     LLVMValueRef *args, unsigned num_args)
{
   unsigned id = LLVMLookupIntrinsicID(name, strlen(name));
   if (!id)
      return NULL;

   LLVMValueRef fn = LLVMGetIntrinsicDeclaration(ctx->module, id, NULL, 0);
   LLVMTypeRef fn_type = LLVMIntrinsicGetType(ctx->llvm_ctx, id, NULL, 0);
   return LLVMBuildCall2(ctx->builder, fn_type, fn, args, num_args, "");
}

/*
 * Call a device function linked in from cp_sampler.cu.
 *
 * Used for transcendentals whose NVVM approximation is too coarse. sin.approx
 * and friends are accurate enough to shade with, but not to place geometry: a
 * shader that rotates an object's position by a per-instance angle multiplies
 * the error by the orbit radius, which visibly displaced every asteroid in the
 * instancing sample. Setting needs_link pulls the module in the same way a
 * texture sample does.
 */
static LLVMValueRef
build_device_call(struct ntl_context *ctx, const char *name,
                  LLVMValueRef *args, unsigned num_args)
{
   LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx->llvm_ctx);
   LLVMTypeRef params[4];
   for (unsigned i = 0; i < num_args && i < 4; i++)
      params[i] = f32;

   LLVMTypeRef fn_type = LLVMFunctionType(f32, params, num_args, false);
   LLVMValueRef fn = LLVMGetNamedFunction(ctx->module, name);
   if (!fn)
      fn = LLVMAddFunction(ctx->module, name, fn_type);

   ctx->needs_link = true;
   return LLVMBuildCall2(ctx->builder, fn_type, fn, args, num_args, "");
}

/* Call an LLVM intrinsic by name, e.g. "llvm.sqrt" — overloaded intrinsics are
 * specialised on the type of their first argument. */

/*
 * A constant of the same shape as `like` -- a splat when `like` is a vector.
 *
 * The conversions and the sign/boolean selects below were written when every
 * ALU operation was scalar, so they built scalar constants and scalar result
 * types. With a vector destination that is invalid IR rather than wrong
 * arithmetic, which is why twelve samples rendered wrong the moment
 * scalarisation was turned off.
 */
static LLVMValueRef
const_real_like(LLVMTypeRef like, double v)
{
   if (LLVMGetTypeKind(like) == LLVMVectorTypeKind) {
      LLVMTypeRef elem = LLVMGetElementType(like);
      unsigned n = LLVMGetVectorSize(like);
      LLVMValueRef vals[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < n; i++)
         vals[i] = LLVMConstReal(elem, v);
      return LLVMConstVector(vals, n);
   }
   return LLVMConstReal(like, v);
}

static LLVMValueRef
build_intrinsic(struct ntl_context *ctx, const char *name,
                LLVMValueRef *args, unsigned num_args)
{
   unsigned id = LLVMLookupIntrinsicID(name, strlen(name));
   if (!id)
      return NULL;

   LLVMTypeRef overload = LLVMTypeOf(args[0]);
   LLVMValueRef fn = LLVMGetIntrinsicDeclaration(ctx->module, id, &overload, 1);
   LLVMTypeRef fn_type = LLVMIntrinsicGetType(ctx->llvm_ctx, id, &overload, 1);
   return LLVMBuildCall2(ctx->builder, fn_type, fn, args, num_args, "");
}


/*
 * Call a scalar-only builder once per component.
 *
 * The NVVM approximations (ex2.approx.f, lg2.approx.f, rsqrt.approx.f) and the
 * device functions linked from cp_sampler.cu all take a float and none of them
 * is overloadable, so handing one a vector produces `call float
 * @llvm.nvvm.lg2.approx.f(<3 x float>)` and the module fails verification.
 * Every shader avoided this only because nir_lower_alu_to_scalar had already
 * made these operations scalar -- and scalarising everything is what puts this
 * driver's shaders under the register pressure that spills them.
 */
static LLVMValueRef
build_scalar_only(struct ntl_context *ctx,
                  LLVMValueRef (*build)(struct ntl_context *, const char *,
                                        LLVMValueRef *, unsigned),
                  const char *name, LLVMValueRef *args, unsigned num_args)
{
   if (LLVMGetTypeKind(LLVMTypeOf(args[0])) != LLVMVectorTypeKind)
      return build(ctx, name, args, num_args);

   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   unsigned n = LLVMGetVectorSize(LLVMTypeOf(args[0]));
   LLVMValueRef result = NULL;

   for (unsigned c = 0; c < n; c++) {
      LLVMValueRef lane[4];
      for (unsigned a = 0; a < num_args && a < 4; a++) {
         lane[a] = LLVMGetTypeKind(LLVMTypeOf(args[a])) == LLVMVectorTypeKind
            ? LLVMBuildExtractElement(ctx->builder, args[a],
                                      LLVMConstInt(i32, c, false), "")
            : args[a];
      }
      LLVMValueRef v = build(ctx, name, lane, num_args);
      if (!v)
         return NULL;
      if (!result)
         result = LLVMGetUndef(LLVMVectorType(LLVMTypeOf(v), n));
      result = LLVMBuildInsertElement(ctx->builder, result, v,
                                      LLVMConstInt(i32, c, false), "");
   }
   return result;
}

static void
emit_alu(struct ntl_context *ctx, nir_alu_instr *instr)
{

   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   unsigned num_comp = instr->def.num_components;
   unsigned bit_size = instr->def.bit_size;
   LLVMTypeRef dst_type = get_llvm_type(ctx, bit_size, num_comp);

   LLVMValueRef src[4] = {0};
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      src[i] = get_src(ctx, &instr->src[i].src);
      if (!src[i])
         continue;

      /*
       * Apply the operand's swizzle, at whatever width this operand is
       * consumed at.
       *
       * This used to do it only for a scalar destination and pass a vector
       * source through untouched otherwise, which drops the swizzle and hands
       * LLVM two sources at their own widths -- `fadd <3 x float>, <4 x
       * float>`, which fails verification. Every shader avoided it only
       * because nir_lower_alu_to_scalar had already made every operation
       * scalar, and scalarising everything is what puts this driver's shaders
       * under the register pressure that spills them to local memory.
       *
       * input_sizes[i] is the width this operand is consumed at when the op
       * fixes it (fdot, the vecN constructors); zero means it follows the
       * destination.
       */
      unsigned isz = nir_op_infos[instr->op].input_sizes[i];
      unsigned want = isz ? isz : num_comp;
      bool src_is_vec =
         LLVMGetTypeKind(LLVMTypeOf(src[i])) == LLVMVectorTypeKind;

      if (want == 1) {
         if (src_is_vec)
            src[i] = LLVMBuildExtractElement(ctx->builder, src[i],
               LLVMConstInt(i32, instr->src[i].swizzle[0], false), "");
         continue;
      }

      if (src_is_vec) {
         /* A shuffle against undef selects exactly the swizzled lanes, and
          * needs no instruction when the mask is the identity. */
         LLVMValueRef mask[NIR_MAX_VEC_COMPONENTS];
         bool identity = LLVMGetVectorSize(LLVMTypeOf(src[i])) == want;
         for (unsigned c = 0; c < want; c++) {
            mask[c] = LLVMConstInt(i32, instr->src[i].swizzle[c], false);
            if (instr->src[i].swizzle[c] != c)
               identity = false;
         }
         if (!identity)
            src[i] = LLVMBuildShuffleVector(ctx->builder, src[i],
               LLVMGetUndef(LLVMTypeOf(src[i])),
               LLVMConstVector(mask, want), "");
      } else {
         /* A scalar feeding a vector operation is every lane. */
         LLVMValueRef v = LLVMGetUndef(
            LLVMVectorType(LLVMTypeOf(src[i]), want));
         for (unsigned c = 0; c < want; c++)
            v = LLVMBuildInsertElement(ctx->builder, v, src[i],
                                       LLVMConstInt(i32, c, false), "");
         src[i] = v;
      }
   }

   LLVMValueRef result = NULL;

   /* For float ops, ensure sources are float-typed (bitcast from int if needed) */
   /* Values live in integer registers regardless of what they represent, so
    * float operations have to reinterpret their sources first. NIR already
    * records the type each operand is consumed as — use that rather than
    * guessing from the opcode, so every float op is covered. */
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      if (nir_alu_type_get_base_type(nir_op_infos[instr->op].input_types[i]) !=
          nir_type_float)
         continue;
      {
         if (src[i] && LLVMGetTypeKind(LLVMTypeOf(src[i])) == LLVMIntegerTypeKind) {
            unsigned w = LLVMGetIntTypeWidth(LLVMTypeOf(src[i]));
            LLVMTypeRef ft = (w == 64) ? LLVMDoubleTypeInContext(ctx->llvm_ctx) :
                             (w == 16) ? LLVMHalfTypeInContext(ctx->llvm_ctx) :
                                         LLVMFloatTypeInContext(ctx->llvm_ctx);
            src[i] = LLVMBuildBitCast(ctx->builder, src[i], ft, "");
         } else if (src[i] &&
                    LLVMGetTypeKind(LLVMTypeOf(src[i])) == LLVMVectorTypeKind &&
                    LLVMGetTypeKind(LLVMGetElementType(LLVMTypeOf(src[i]))) ==
                       LLVMIntegerTypeKind) {
            LLVMTypeRef elem = LLVMGetElementType(LLVMTypeOf(src[i]));
            unsigned w = LLVMGetIntTypeWidth(elem);
            LLVMTypeRef ft = (w == 64) ? LLVMDoubleTypeInContext(ctx->llvm_ctx) :
                             (w == 16) ? LLVMHalfTypeInContext(ctx->llvm_ctx) :
                                         LLVMFloatTypeInContext(ctx->llvm_ctx);
            src[i] = LLVMBuildBitCast(ctx->builder, src[i],
               LLVMVectorType(ft, LLVMGetVectorSize(LLVMTypeOf(src[i]))), "");
         }
      }
   }

   switch (instr->op) {
   case nir_op_iadd:
      result = LLVMBuildAdd(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_isub:
      result = LLVMBuildSub(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_imul:
      result = LLVMBuildMul(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_udiv:
      result = LLVMBuildUDiv(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_idiv:
      result = LLVMBuildSDiv(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_umod:
      result = LLVMBuildURem(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_irem:
      result = LLVMBuildSRem(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_imod: {
      /* Unlike irem, imod's result takes the sign of the divisor, so a
       * remainder with the wrong sign has to be nudged by one divisor. */
      LLVMValueRef rem = LLVMBuildSRem(ctx->builder, src[0], src[1], "");
      LLVMValueRef zero = LLVMConstNull(LLVMTypeOf(rem));
      LLVMValueRef rem_nonzero = LLVMBuildICmp(ctx->builder, LLVMIntNE, rem, zero, "");
      LLVMValueRef rem_neg = LLVMBuildICmp(ctx->builder, LLVMIntSLT, rem, zero, "");
      LLVMValueRef div_neg = LLVMBuildICmp(ctx->builder, LLVMIntSLT, src[1], zero, "");
      LLVMValueRef differ = LLVMBuildXor(ctx->builder, rem_neg, div_neg, "");
      LLVMValueRef adjust = LLVMBuildAnd(ctx->builder, rem_nonzero, differ, "");
      result = LLVMBuildSelect(ctx->builder, adjust,
                               LLVMBuildAdd(ctx->builder, rem, src[1], ""), rem, "");
      break;
   }
   case nir_op_imin:
      result = build_intrinsic(ctx, "llvm.smin", src, 2);
      break;
   case nir_op_imax:
      result = build_intrinsic(ctx, "llvm.smax", src, 2);
      break;
   case nir_op_umin:
      result = build_intrinsic(ctx, "llvm.umin", src, 2);
      break;
   case nir_op_umax:
      result = build_intrinsic(ctx, "llvm.umax", src, 2);
      break;
   case nir_op_iabs: {
      LLVMValueRef args[2] = {
         src[0], LLVMConstInt(LLVMInt1TypeInContext(ctx->llvm_ctx), 0, false)
      };
      result = build_intrinsic(ctx, "llvm.abs", args, 2);
      break;
   }
   case nir_op_fabs:
      result = build_intrinsic(ctx, "llvm.fabs", src, 1);
      break;
   case nir_op_fmin:
      result = build_intrinsic(ctx, "llvm.minnum", src, 2);
      break;
   case nir_op_fmax:
      result = build_intrinsic(ctx, "llvm.maxnum", src, 2);
      break;
   case nir_op_fsqrt:
      result = build_intrinsic(ctx, "llvm.sqrt", src, 1);
      break;
   case nir_op_frsq:
      if (bit_size == 32) {
         result = build_scalar_only(ctx, build_nvvm_intrinsic, "llvm.nvvm.rsqrt.approx.f", src, 1);
      } else {
         LLVMValueRef root = build_intrinsic(ctx, "llvm.sqrt", src, 1);
         result = LLVMBuildFDiv(ctx->builder,
                                LLVMConstReal(LLVMTypeOf(root), 1.0), root, "");
      }
      break;
   case nir_op_frcp:
      result = LLVMBuildFDiv(ctx->builder,
                             LLVMConstReal(LLVMTypeOf(src[0]), 1.0), src[0], "");
      break;
   case nir_op_ffloor:
      result = build_intrinsic(ctx, "llvm.floor", src, 1);
      break;
   case nir_op_fceil:
      result = build_intrinsic(ctx, "llvm.ceil", src, 1);
      break;
   case nir_op_ftrunc:
      result = build_intrinsic(ctx, "llvm.trunc", src, 1);
      break;
   case nir_op_fround_even:
      result = build_intrinsic(ctx, "llvm.rint", src, 1);
      break;
   case nir_op_ffract: {
      LLVMValueRef floor = build_intrinsic(ctx, "llvm.floor", src, 1);
      result = LLVMBuildFSub(ctx->builder, src[0], floor, "");
      break;
   }
   case nir_op_fexp2:
      result = bit_size == 32
         ? build_scalar_only(ctx, build_nvvm_intrinsic, "llvm.nvvm.ex2.approx.f", src, 1)
         : build_intrinsic(ctx, "llvm.exp2", src, 1);
      break;
   case nir_op_flog2:
      result = bit_size == 32
         ? build_scalar_only(ctx, build_nvvm_intrinsic, "llvm.nvvm.lg2.approx.f", src, 1)
         : build_intrinsic(ctx, "llvm.log2", src, 1);
      break;
   case nir_op_fpow:
      result = build_intrinsic(ctx, "llvm.pow", src, 2);
      break;
   case nir_op_fsin:
      result = bit_size == 32
         ? build_scalar_only(ctx, build_device_call, "cp_sinf", src, 1)
         : build_intrinsic(ctx, "llvm.sin", src, 1);
      break;
   case nir_op_fcos:
      result = bit_size == 32
         ? build_scalar_only(ctx, build_device_call, "cp_cosf", src, 1)
         : build_intrinsic(ctx, "llvm.cos", src, 1);
      break;
   case nir_op_ffma:
      result = build_intrinsic(ctx, "llvm.fma", src, 3);
      break;
   case nir_op_fsign: {
      /* -1, 0 or +1, with 0 preserved (including -0). */
      LLVMTypeRef ft = LLVMTypeOf(src[0]);
      LLVMValueRef zero = LLVMConstNull(ft);
      LLVMValueRef gt = LLVMBuildFCmp(ctx->builder, LLVMRealOGT, src[0], zero, "");
      LLVMValueRef lt = LLVMBuildFCmp(ctx->builder, LLVMRealOLT, src[0], zero, "");
      result = LLVMBuildSelect(ctx->builder, gt, const_real_like(ft, 1.0),
                 LLVMBuildSelect(ctx->builder, lt, const_real_like(ft, -1.0),
                                 zero, ""), "");
      break;
   }
   case nir_op_isign: {
      LLVMTypeRef it = LLVMTypeOf(src[0]);
      LLVMValueRef zero = LLVMConstNull(it);
      LLVMValueRef gt = LLVMBuildICmp(ctx->builder, LLVMIntSGT, src[0], zero, "");
      LLVMValueRef lt = LLVMBuildICmp(ctx->builder, LLVMIntSLT, src[0], zero, "");
      result = LLVMBuildSelect(ctx->builder, gt, LLVMConstInt(it, 1, true),
                 LLVMBuildSelect(ctx->builder, lt, LLVMConstAllOnes(it), zero, ""), "");
      break;
   }
   case nir_op_fadd:
      result = LLVMBuildFAdd(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_fsub:
      result = LLVMBuildFSub(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_fmul:
      result = LLVMBuildFMul(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_fdiv:
      result = LLVMBuildFDiv(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_iand:
      result = LLVMBuildAnd(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_ior:
      result = LLVMBuildOr(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_ixor:
      result = LLVMBuildXor(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_ishl:
      result = LLVMBuildShl(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_ishr:
      result = LLVMBuildAShr(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_ushr:
      result = LLVMBuildLShr(ctx->builder, src[0], src[1], "");
      break;
   case nir_op_ineg:
      result = LLVMBuildNeg(ctx->builder, src[0], "");
      break;
   case nir_op_fneg:
      result = LLVMBuildFNeg(ctx->builder, src[0], "");
      break;
   case nir_op_inot:
      result = LLVMBuildNot(ctx->builder, src[0], "");
      break;
   case nir_op_u2f32:
      result = LLVMBuildUIToFP(ctx->builder, src[0], get_float_type(ctx, 32), "");
      break;
   case nir_op_i2f32:
      result = LLVMBuildSIToFP(ctx->builder, src[0], get_float_type(ctx, 32), "");
      break;
   case nir_op_f2i32:
      result = LLVMBuildFPToSI(ctx->builder, src[0],
                               get_llvm_type(ctx, 32, num_comp), "");
      break;
   case nir_op_f2u32:
      result = LLVMBuildFPToUI(ctx->builder, src[0],
                               get_llvm_type(ctx, 32, num_comp), "");
      break;
   case nir_op_f2f32: {
      LLVMTypeRef st = LLVMTypeOf(src[0]);
      LLVMTypeRef elem = LLVMGetTypeKind(st) == LLVMVectorTypeKind
         ? LLVMGetElementType(st) : st;
      LLVMTypeKind kind = LLVMGetTypeKind(elem);
      LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx->llvm_ctx);
      LLVMTypeRef out = num_comp > 1 ? LLVMVectorType(f32, num_comp) : f32;
      if (kind == LLVMHalfTypeKind)
         result = LLVMBuildFPExt(ctx->builder, src[0], out, "");
      else if (kind == LLVMDoubleTypeKind)
         result = LLVMBuildFPTrunc(ctx->builder, src[0], out, "");
      else
         result = src[0];
      break;
   }
   case nir_op_unpack_32_2x16: {
      LLVMTypeRef i16 = LLVMInt16TypeInContext(ctx->llvm_ctx);
      LLVMValueRef lo = LLVMBuildTrunc(ctx->builder, src[0], i16, "");
      LLVMValueRef hi32 = LLVMBuildLShr(ctx->builder, src[0],
         LLVMConstInt(i32, 16, false), "");
      LLVMValueRef hi = LLVMBuildTrunc(ctx->builder, hi32, i16, "");
      result = LLVMGetUndef(LLVMVectorType(i16, 2));
      result = LLVMBuildInsertElement(ctx->builder, result, lo,
         LLVMConstInt(i32, 0, false), "");
      result = LLVMBuildInsertElement(ctx->builder, result, hi,
         LLVMConstInt(i32, 1, false), "");
      break;
   }
   case nir_op_i2i64:
      result = LLVMBuildSExt(ctx->builder, src[0],
                             get_llvm_type(ctx, 64, num_comp), "");
      break;
   case nir_op_u2u64:
      result = LLVMBuildZExt(ctx->builder, src[0],
                             get_llvm_type(ctx, 64, num_comp), "");
      break;
   case nir_op_i2i32:
   case nir_op_u2u32:
      result = LLVMBuildTrunc(ctx->builder, src[0],
                              get_llvm_type(ctx, 32, num_comp), "");
      break;
   case nir_op_mov:
      result = src[0];
      break;
   case nir_op_bcsel:
   case nir_op_b32csel: {
      LLVMValueRef cond_val = src[0];
      if (LLVMGetTypeKind(LLVMTypeOf(cond_val)) != LLVMIntegerTypeKind ||
          LLVMGetIntTypeWidth(LLVMTypeOf(cond_val)) != 1) {
         cond_val = LLVMBuildICmp(ctx->builder, LLVMIntNE, cond_val,
            LLVMConstNull(LLVMTypeOf(cond_val)), "");
      }
      /*
       * Ensure both branches have matching types -- NIR constant folding can
       * produce a bcsel with an i32 zero on one side and a float on the other.
       *
       * This tested the scalar type kinds only, so a vector pair never
       * matched and `select <4 x i1>, <4 x i32>, <4 x float>` reached the
       * verifier. Compare the element kinds instead, which is the same
       * question at any width, and bitcast the integer side to the float
       * side's type.
       */
      LLVMTypeRef t1 = LLVMTypeOf(src[1]);
      LLVMTypeRef t2 = LLVMTypeOf(src[2]);
      if (t1 != t2) {
         LLVMTypeRef e1 = LLVMGetTypeKind(t1) == LLVMVectorTypeKind
            ? LLVMGetElementType(t1) : t1;
         LLVMTypeRef e2 = LLVMGetTypeKind(t2) == LLVMVectorTypeKind
            ? LLVMGetElementType(t2) : t2;
         bool f1 = LLVMGetTypeKind(e1) == LLVMFloatTypeKind ||
                   LLVMGetTypeKind(e1) == LLVMHalfTypeKind ||
                   LLVMGetTypeKind(e1) == LLVMDoubleTypeKind;
         bool f2 = LLVMGetTypeKind(e2) == LLVMFloatTypeKind ||
                   LLVMGetTypeKind(e2) == LLVMHalfTypeKind ||
                   LLVMGetTypeKind(e2) == LLVMDoubleTypeKind;
         if (f1 && LLVMGetTypeKind(e2) == LLVMIntegerTypeKind)
            src[2] = LLVMBuildBitCast(ctx->builder, src[2], t1, "");
         else if (f2 && LLVMGetTypeKind(e1) == LLVMIntegerTypeKind)
            src[1] = LLVMBuildBitCast(ctx->builder, src[1], t2, "");
      }
      result = LLVMBuildSelect(ctx->builder, cond_val, src[1], src[2], "");
      break;
   }
   case nir_op_b2f32: {
      LLVMValueRef cond_val = src[0];
      if (LLVMGetTypeKind(LLVMTypeOf(cond_val)) != LLVMIntegerTypeKind ||
          LLVMGetIntTypeWidth(LLVMTypeOf(cond_val)) != 1) {
         cond_val = LLVMBuildICmp(ctx->builder, LLVMIntNE, cond_val,
            LLVMConstNull(LLVMTypeOf(cond_val)), "");
      }
      {
         /* Float, not the integer get_llvm_type() gives for a 32-bit value:
          * LLVMConstReal on an integer type is invalid and segfaulted pbribl. */
         LLVMTypeRef f = LLVMFloatTypeInContext(ctx->llvm_ctx);
         if (num_comp > 1)
            f = LLVMVectorType(f, num_comp);
         result = LLVMBuildSelect(ctx->builder, cond_val,
            const_real_like(f, 1.0), const_real_like(f, 0.0), "");
      }
      break;
   }
   case nir_op_b2i32: {
      LLVMValueRef cond_val = src[0];
      if (LLVMGetTypeKind(LLVMTypeOf(cond_val)) != LLVMIntegerTypeKind ||
          LLVMGetIntTypeWidth(LLVMTypeOf(cond_val)) != 1) {
         cond_val = LLVMBuildICmp(ctx->builder, LLVMIntNE, cond_val,
            LLVMConstNull(LLVMTypeOf(cond_val)), "");
      }
      result = LLVMBuildZExt(ctx->builder, cond_val, i32, "");
      break;
   }
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      unsigned nc = nir_op_infos[instr->op].num_inputs;
      /* For vec ops, sources are always scalars — extract if vector */
      for (unsigned c = 0; c < nc; c++) {
         if (LLVMGetTypeKind(LLVMTypeOf(src[c])) == LLVMVectorTypeKind) {
            src[c] = LLVMBuildExtractElement(ctx->builder, src[c],
               LLVMConstInt(i32, instr->src[c].swizzle[0], false), "");
         }
      }
      /* Components reach here in whichever representation produced them —
       * load_const yields integers even for float data. Settle on one element
       * type and bitcast the rest, or insertelement rejects the mix. */
      LLVMTypeRef elem_type = LLVMTypeOf(src[0]);
      for (unsigned c = 1; c < nc; c++) {
         if (LLVMGetTypeKind(LLVMTypeOf(src[c])) == LLVMFloatTypeKind ||
             LLVMGetTypeKind(LLVMTypeOf(src[c])) == LLVMDoubleTypeKind ||
             LLVMGetTypeKind(LLVMTypeOf(src[c])) == LLVMHalfTypeKind) {
            elem_type = LLVMTypeOf(src[c]);
            break;
         }
      }

      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(elem_type, nc));
      for (unsigned c = 0; c < nc; c++) {
         LLVMValueRef comp = src[c];
         if (LLVMTypeOf(comp) != elem_type)
            comp = coerce_to_type(ctx, comp, elem_type);
         vec = LLVMBuildInsertElement(ctx->builder, vec, comp,
            LLVMConstInt(i32, c, false), "");
      }
      result = vec;
      break;
   }
   case nir_op_ilt:
      result = LLVMBuildICmp(ctx->builder, LLVMIntSLT, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, bit_size, 1), "");
      break;
   case nir_op_ige:
      result = LLVMBuildICmp(ctx->builder, LLVMIntSGE, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, bit_size, 1), "");
      break;
   case nir_op_uge:
      result = LLVMBuildICmp(ctx->builder, LLVMIntUGE, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, bit_size, 1), "");
      break;
   case nir_op_ult:
      result = LLVMBuildICmp(ctx->builder, LLVMIntULT, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, bit_size, 1), "");
      break;
   case nir_op_ieq:
      result = LLVMBuildICmp(ctx->builder, LLVMIntEQ, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, bit_size, 1), "");
      break;
   case nir_op_ine:
      result = LLVMBuildICmp(ctx->builder, LLVMIntNE, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, bit_size, 1), "");
      break;
   case nir_op_flt:
      result = LLVMBuildFCmp(ctx->builder, LLVMRealOLT, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, 32, 1), "");
      break;
   case nir_op_fge:
      result = LLVMBuildFCmp(ctx->builder, LLVMRealOGE, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, 32, 1), "");
      break;
   case nir_op_feq:
      result = LLVMBuildFCmp(ctx->builder, LLVMRealOEQ, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, 32, 1), "");
      break;
   case nir_op_fneu:
      result = LLVMBuildFCmp(ctx->builder, LLVMRealUNE, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, 32, 1), "");
      break;
   default:
      /* Same trap as the intrinsics: undef propagates silently and can fold a
       * whole shader to a constant, so say so — always, not just when someone
       * already suspects the shader enough to set a debug variable. */
      warn_undef_once("ALU op", nir_op_infos[instr->op].name);
      result = LLVMGetUndef(dst_type);
      break;
   }

   /* build_intrinsic() returns NULL if LLVM doesn't know the name. Same
    * outcome as the unhandled-op case above, reached a different way: the op
    * is handled, but the LLVM intrinsic it lowers to does not exist. */
   if (!result) {
      warn_undef_once("the LLVM intrinsic for ALU op",
                      nir_op_infos[instr->op].name);
      result = LLVMGetUndef(dst_type);
   }

   set_ssa_def(ctx, &instr->def, result);
}

static void
emit_load_const(struct ntl_context *ctx, nir_load_const_instr *instr)
{
   unsigned num_comp = instr->def.num_components;
   unsigned bit_size = instr->def.bit_size;

   if (num_comp == 1) {
      LLVMValueRef val;
      if (bit_size == 32)
         val = LLVMConstInt(LLVMInt32TypeInContext(ctx->llvm_ctx), instr->value[0].u32, false);
      else if (bit_size == 64)
         val = LLVMConstInt(LLVMInt64TypeInContext(ctx->llvm_ctx), instr->value[0].u64, false);
      else if (bit_size == 1)
         val = LLVMConstInt(LLVMInt1TypeInContext(ctx->llvm_ctx), instr->value[0].b, false);
      else
         val = LLVMConstInt(get_llvm_type(ctx, bit_size, 1), instr->value[0].u32, false);
      set_ssa_def(ctx, &instr->def, val);
   } else {
      LLVMTypeRef elem_type = get_llvm_type(ctx, bit_size, 1);
      LLVMValueRef *vals = alloca(num_comp * sizeof(LLVMValueRef));
      for (unsigned i = 0; i < num_comp; i++) {
         if (bit_size == 32)
            vals[i] = LLVMConstInt(elem_type, instr->value[i].u32, false);
         else if (bit_size == 64)
            vals[i] = LLVMConstInt(elem_type, instr->value[i].u64, false);
         else
            vals[i] = LLVMConstInt(elem_type, instr->value[i].u32, false);
      }
      set_ssa_def(ctx, &instr->def, LLVMConstVector(vals, num_comp));
   }
}

static int32_t
cp_tex_flags(const nir_tex_instr *tex)
{
   int32_t flags;
   switch (tex->sampler_dim) {
   case GLSL_SAMPLER_DIM_1D:
      flags = tex->is_array ? CP_TEX_1D_ARRAY : CP_TEX_1D;
      break;
   case GLSL_SAMPLER_DIM_3D:
      flags = CP_TEX_3D;
      break;
   case GLSL_SAMPLER_DIM_CUBE:
      flags = tex->is_array ? CP_TEX_CUBE_ARRAY : CP_TEX_CUBE;
      break;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_EXTERNAL:
   case GLSL_SAMPLER_DIM_MS:
      flags = tex->is_array ? CP_TEX_2D_ARRAY : CP_TEX_2D;
      break;
   default:
      flags = -1;
      break;
   }

   if (tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms)
      flags |= CP_TEX_FETCH;
   else if (tex->op == nir_texop_txl)
      flags |= CP_TEX_LOD;
   else if (tex->op == nir_texop_txb)
      flags |= CP_TEX_BIAS;

   return flags;
}

/* Load the opaque CUtexObject stored in the combined descriptor row. */
static LLVMValueRef
cp_hardware_texture_handle(struct ntl_context *ctx)
{
   ctx->reads_const_bufs = true;
   LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   LLVMValueRef table = LLVMBuildBitCast(ctx->builder,
      cp_arg_slot(ctx, CP_ARG_SLOT_HW_TEX_TABLE), LLVMPointerType(i64, 0),
      "hwtex_table");
   LLVMValueRef row = emit_batch_row(ctx);
   LLVMValueRef index = LLVMBuildAdd(ctx->builder,
      LLVMBuildMul(ctx->builder, row,
                   LLVMConstInt(i32, ctx->hardware_texture_site_count, false),
                   ""),
      LLVMConstInt(i32, ctx->hardware_texture_site++, false), "");
   index = LLVMBuildZExt(ctx->builder, index, i64, "");
   LLVMValueRef object = LLVMBuildLoad2(ctx->builder, i64,
      LLVMBuildGEP2(ctx->builder, i64, table, &index, 1, ""), "hwtex");
   LLVMSetMetadata(object, ctx->md_invariant_load,
                   LLVMMDNodeInContext(ctx->llvm_ctx, NULL, 0));
   return object;
}

static LLVMValueRef
cp_tex_src_value(struct ntl_context *ctx, nir_tex_instr *tex,
                 nir_tex_src_type kind)
{
   for (unsigned i = 0; i < tex->num_srcs; i++)
      if (tex->src[i].src_type == kind)
         return get_src(ctx, &tex->src[i].src);
   return NULL;
}

static LLVMValueRef
cp_float_component(struct ntl_context *ctx, LLVMValueRef value, unsigned comp)
{
   LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   if (LLVMGetTypeKind(LLVMTypeOf(value)) == LLVMVectorTypeKind)
      value = LLVMBuildExtractElement(ctx->builder, value,
                                      LLVMConstInt(i32, comp, false), "");
   if (LLVMGetTypeKind(LLVMTypeOf(value)) == LLVMIntegerTypeKind)
      value = LLVMBuildBitCast(ctx->builder, value, f32, "");
   return value;
}

static LLVMValueRef
cp_quad_derivative(struct ntl_context *ctx, LLVMValueRef value, bool y)
{
   LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   LLVMValueRef tid = emit_local_invocation_id(ctx, 0);
   LLVMValueRef quad_base = LLVMBuildAnd(ctx->builder, tid,
                                         LLVMConstInt(i32, 28, false), "");
   LLVMValueRef mask = LLVMBuildShl(ctx->builder,
      LLVMConstInt(i32, 15, false), quad_base, "quad_mask");
   LLVMTypeRef params[] = { i32, f32, i32, i32 };
   LLVMTypeRef fn_type = LLVMFunctionType(f32, params, 4, false);
   LLVMValueRef fn = LLVMGetNamedFunction(ctx->module,
                                          "llvm.nvvm.shfl.sync.bfly.f32");
   if (!fn)
      fn = LLVMAddFunction(ctx->module, "llvm.nvvm.shfl.sync.bfly.f32",
                           fn_type);
   LLVMValueRef args[] = {
      mask, value, LLVMConstInt(i32, y ? 2 : 1, false),
      LLVMConstInt(i32, 0x1c1f, false),
   };
   LLVMValueRef peer = LLVMBuildCall2(ctx->builder, fn_type, fn, args, 4,
                                      y ? "qy" : "qx");
   LLVMValueRef high = LLVMBuildICmp(ctx->builder, LLVMIntNE,
      LLVMBuildAnd(ctx->builder, tid, LLVMConstInt(i32, y ? 2 : 1, false), ""),
      LLVMConstInt(i32, 0, false), "");
   LLVMValueRef low_delta = LLVMBuildFSub(ctx->builder, peer, value, "");
   LLVMValueRef high_delta = LLVMBuildFSub(ctx->builder, value, peer, "");
   return LLVMBuildSelect(ctx->builder, high, high_delta, low_delta,
                          y ? "ddy" : "ddx");
}

/* Emit one static float4 bindless texture-object instruction. The site number
 * is the immutable column in the row-major handle table uploaded before this
 * launch. Implicit LOD is expressed as explicit quad gradients: compute-stage
 * CUDA texture instructions do not infer graphics fragment derivatives. */
static void
emit_hardware_tex(struct ntl_context *ctx, nir_tex_instr *tex,
                  LLVMValueRef coord)
{
   LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   LLVMValueRef object = cp_hardware_texture_handle(ctx);
   LLVMValueRef args[10] = { object };
   unsigned coord_count = tex->sampler_dim == GLSL_SAMPLER_DIM_2D ? 2 : 3;
   unsigned nargs = 1;
   for (unsigned i = 0; i < coord_count; i++)
      args[nargs++] = cp_float_component(ctx, coord, i);

   const char *dim = tex->sampler_dim == GLSL_SAMPLER_DIM_2D ? "2d" :
                     (tex->sampler_dim == GLSL_SAMPLER_DIM_3D ? "3d" : "cube");
   const char *mode = "";
   if (tex->op == nir_texop_txl) {
      LLVMValueRef lod = cp_tex_src_value(ctx, tex, nir_tex_src_lod);
      assert(lod);
      args[nargs++] = cp_float_component(ctx, lod, 0);
      mode = ".level";
   } else {
      LLVMValueRef dx = cp_tex_src_value(ctx, tex, nir_tex_src_ddx);
      LLVMValueRef dy = cp_tex_src_value(ctx, tex, nir_tex_src_ddy);
      LLVMValueRef bias = tex->op == nir_texop_txb
         ? cp_tex_src_value(ctx, tex, nir_tex_src_bias) : NULL;
      LLVMValueRef scale = NULL;
      if (bias) {
         bias = cp_float_component(ctx, bias, 0);
         scale = build_nvvm_intrinsic(ctx, "llvm.nvvm.ex2.approx.f",
                                      &bias, 1);
      }
      for (unsigned i = 0; i < coord_count; i++) {
         LLVMValueRef v = dx ? cp_float_component(ctx, dx, i)
                             : cp_quad_derivative(ctx, args[1 + i], false);
         if (scale)
            v = LLVMBuildFMul(ctx->builder, v, scale, "");
         args[nargs++] = v;
      }
      for (unsigned i = 0; i < coord_count; i++) {
         LLVMValueRef v = dy ? cp_float_component(ctx, dy, i)
                             : cp_quad_derivative(ctx, args[1 + i], true);
         if (scale)
            v = LLVMBuildFMul(ctx->builder, v, scale, "");
         args[nargs++] = v;
      }
      mode = ".grad";
   }

   LLVMTypeRef params[10];
   params[0] = i64;
   for (unsigned i = 1; i < nargs; i++)
      params[i] = f32;
   LLVMTypeRef ret = LLVMStructTypeInContext(ctx->llvm_ctx,
      (LLVMTypeRef[]){ f32, f32, f32, f32 }, 4, false);
   LLVMTypeRef fn_type = LLVMFunctionType(ret, params, nargs, false);
   char name[96];
   snprintf(name, sizeof(name), "llvm.nvvm.tex.unified.%s%s.v4f32.f32",
            dim, mode);
   LLVMValueRef fn = LLVMGetNamedFunction(ctx->module, name);
   if (!fn)
      fn = LLVMAddFunction(ctx->module, name, fn_type);
   LLVMValueRef sample = LLVMBuildCall2(ctx->builder, fn_type, fn, args, nargs,
                                        "hwtex");

   LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(f32, 4));
   for (unsigned i = 0; i < 4; i++)
      vec = LLVMBuildInsertElement(ctx->builder, vec,
         LLVMBuildExtractValue(ctx->builder, sample, i, ""),
         LLVMConstInt(i32, i, false), "");
   set_ssa_def(ctx, &tex->def, vec);
   ctx->uses_tex = true;
}

/*
 * Texture sampling.
 *
 * Rather than emitting a software sampler as LLVM IR, call into the sampler
 * written in CUDA C (cp_sampler.cu). Its relocatable PTX is linked with this
 * shader's PTX at module-load time, so this is just an external call.
 */
/* The descriptor row's stride, from the shader ABI header the front end and
 * the kernels also use. */

static void
emit_tex(struct ntl_context *ctx, nir_tex_instr *tex)
{
   LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx->llvm_ctx);
   unsigned nc = tex->def.num_components;
   unsigned bs = tex->def.bit_size;

   LLVMValueRef tex_handle = NULL, samp_handle = NULL, coord = NULL;
   LLVMValueRef explicit_lod = NULL;
   LLVMValueRef sampler_offset = NULL, texture_offset = NULL;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_handle:
         tex_handle = get_src(ctx, &tex->src[i].src);
         break;
      case nir_tex_src_sampler_handle:
         samp_handle = get_src(ctx, &tex->src[i].src);
         break;
      case nir_tex_src_coord:
         coord = get_src(ctx, &tex->src[i].src);
         break;
      case nir_tex_src_sampler_offset:
         /*
          * The element of a sampler array, which arrives as an offset in
          * descriptors rather than folded into the handle. Dropping it put
          * every access on element 0: texturemipmapgen selects among three
          * samplers with a uniform and always got the first, the one with no
          * mip levels.
          */
         sampler_offset = get_src(ctx, &tex->src[i].src);
         break;
      case nir_tex_src_texture_offset:
         texture_offset = get_src(ctx, &tex->src[i].src);
         break;
      case nir_tex_src_lod:
         explicit_lod = get_src(ctx, &tex->src[i].src);
         break;
      default:
         break;
      }
   }

   int32_t flags = cp_tex_flags(tex);

   bool supported = flags >= 0 && tex_handle && coord &&
      (tex->op == nir_texop_tex || tex->op == nir_texop_txl ||
       tex->op == nir_texop_txb || tex->op == nir_texop_txf ||
       tex->op == nir_texop_txf_ms);

   if (ctx->hardware_texture) {
      /* Shader admission makes both values mandatory. Never link a software
       * or null-handle branch into this resource-isolated module. */
      assert(tex_handle && coord);
      emit_hardware_tex(ctx, tex, coord);
      return;
   }

   /*
    * textureSize(). Its result is an integer vector, and shaders divide by it
    * to step a texel at a time, so a zero here does not degrade quality — it
    * sends every subsequent sample to an infinite coordinate. bloom's gaussian
    * blur does exactly that and came out black.
    */
   if (tex->op == nir_texop_txs && tex_handle) {
      LLVMTypeRef size_params[] = { i64, i32, i32 };
      LLVMTypeRef size_fn_type = LLVMFunctionType(i32, size_params, 3, false);
      LLVMValueRef size_fn = LLVMGetNamedFunction(ctx->module, "cp_tex_size");
      if (!size_fn)
         size_fn = LLVMAddFunction(ctx->module, "cp_tex_size", size_fn_type);

      LLVMValueRef lod_src = LLVMConstInt(i32, 0, false);
      for (unsigned i = 0; i < tex->num_srcs; i++) {
         if (tex->src[i].src_type == nir_tex_src_lod)
            lod_src = get_src(ctx, &tex->src[i].src);
      }
      if (LLVMGetTypeKind(LLVMTypeOf(lod_src)) == LLVMFloatTypeKind)
         lod_src = LLVMBuildFPToSI(ctx->builder, lod_src, i32, "");

      ctx->uses_tex = true;

      unsigned comps = tex->def.num_components;
      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(i32, comps));
      for (unsigned c = 0; c < comps; c++) {
         LLVMValueRef size_args[] = { tex_handle, lod_src,
                                      LLVMConstInt(i32, c, false) };
         LLVMValueRef sz = LLVMBuildCall2(ctx->builder, size_fn_type, size_fn,
                                          size_args, 3, "texsize");
         if (comps == 1) {
            set_ssa_def(ctx, &tex->def, sz);
            return;
         }
         vec = LLVMBuildInsertElement(ctx->builder, vec, sz,
                                      LLVMConstInt(i32, c, false), "");
      }
      set_ssa_def(ctx, &tex->def, vec);
      return;
   }

   /* Shadow compares, gathers and derivative-explicit samples still have to
    * produce a value even though the sampler cannot serve them yet. */
   if (!supported) {
      LLVMTypeRef ft = get_float_type(ctx, bs);
      LLVMValueRef zero = LLVMConstReal(ft, 0.0);
      if (nc == 1) {
         set_ssa_def(ctx, &tex->def, zero);
      } else {
         LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(ft, nc));
         for (unsigned c = 0; c < nc; c++)
            vec = LLVMBuildInsertElement(ctx->builder, vec,
               c == 3 ? LLVMConstReal(ft, 1.0) : zero,
               LLVMConstInt(i32, c, false), "");
         set_ssa_def(ctx, &tex->def, vec);
      }
      return;
   }

   ctx->uses_tex = true;

   if (!samp_handle)
      samp_handle = LLVMConstInt(i64, 0, false);

   /* Array elements, in descriptors, applied to the handles they index. */
   if (sampler_offset) {
      LLVMValueRef off = LLVMBuildZExt(ctx->builder, sampler_offset, i64, "");
      samp_handle = LLVMBuildAdd(ctx->builder, samp_handle,
         LLVMBuildMul(ctx->builder, off,
                      LLVMConstInt(i64, CPVK_DESCRIPTOR_SIZE, false), ""), "");
   }
   if (texture_offset) {
      LLVMValueRef off = LLVMBuildZExt(ctx->builder, texture_offset, i64, "");
      tex_handle = LLVMBuildAdd(ctx->builder, tex_handle,
         LLVMBuildMul(ctx->builder, off,
                      LLVMConstInt(i64, CPVK_DESCRIPTOR_SIZE, false), ""), "");
   }

   /* Pull out up to three coordinate components as floats. Texel fetches carry
    * integer coordinates, which convert numerically; sampled coordinates are
    * already floats and only need a bitcast out of the integer register. */
   bool integer_coords = (flags & CP_TEX_FETCH) != 0;
   LLVMValueRef c[3];
   for (unsigned i = 0; i < 3; i++) {
      if (i < (unsigned)tex->coord_components) {
         c[i] = LLVMGetTypeKind(LLVMTypeOf(coord)) == LLVMVectorTypeKind
            ? LLVMBuildExtractElement(ctx->builder, coord,
                                      LLVMConstInt(i32, i, false), "")
            : coord;
         if (LLVMGetTypeKind(LLVMTypeOf(c[i])) == LLVMIntegerTypeKind) {
            c[i] = integer_coords
               ? LLVMBuildSIToFP(ctx->builder, c[i], f32, "")
               : LLVMBuildBitCast(ctx->builder, c[i], f32, "");
         }
      } else {
         c[i] = LLVMConstReal(f32, 0.0);
      }
   }

   LLVMValueRef lod_arg = LLVMConstReal(f32, 0.0);
   if (explicit_lod) {
      lod_arg = explicit_lod;
      if (LLVMGetTypeKind(LLVMTypeOf(lod_arg)) == LLVMVectorTypeKind)
         lod_arg = LLVMBuildExtractElement(ctx->builder, lod_arg,
                                           LLVMConstInt(i32, 0, false), "");
      if (LLVMGetTypeKind(LLVMTypeOf(lod_arg)) == LLVMIntegerTypeKind) {
         lod_arg = integer_coords
            ? LLVMBuildSIToFP(ctx->builder, lod_arg, f32, "")
            : LLVMBuildBitCast(ctx->builder, lod_arg, f32, "");
      }
   }

   /* If the coordinate is a varying straight from the rasterizer, tell the
    * sampler which one: it can then look up that varying's screen-space
    * derivatives and select a mip level. Anything computed in the shader
    * leaves the sampler on the base level. */
   int32_t coord_slot = -1;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type != nir_tex_src_coord)
         continue;
      nir_instr *parent = nir_def_instr(tex->src[i].src.ssa);
      if (parent->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(parent);
         if (intr->intrinsic == nir_intrinsic_load_input)
            coord_slot = nir_intrinsic_base(intr);
      }
   }

   /* float4 comes back as a struct of four floats under the NVPTX ABI. */
   LLVMTypeRef ret_type = LLVMStructTypeInContext(ctx->llvm_ctx,
      (LLVMTypeRef[]){ f32, f32, f32, f32 }, 4, false);
   LLVMTypeRef param_types[] = { i64, i64, f32, f32, f32, f32, i32, i32 };
   LLVMTypeRef fn_type = LLVMFunctionType(ret_type, param_types, 8, false);

   const char *sample_name =
      (flags & CP_TEX_TARGET_MASK) == CP_TEX_3D &&
      !(flags & CP_TEX_FETCH) ? "cp_tex_sample_3d" : "cp_tex_sample";
   LLVMValueRef fn = LLVMGetNamedFunction(ctx->module, sample_name);
   if (!fn)
      fn = LLVMAddFunction(ctx->module, sample_name, fn_type);

   LLVMValueRef args[] = { tex_handle, samp_handle, c[0], c[1], c[2], lod_arg,
                           LLVMConstInt(i32, (unsigned)coord_slot, true),
                           LLVMConstInt(i32, (unsigned)flags, true) };
   LLVMValueRef call = LLVMBuildCall2(ctx->builder, fn_type, fn, args, 8, "tex");

   LLVMTypeRef ft = get_float_type(ctx, bs);
   if (nc == 1) {
      set_ssa_def(ctx, &tex->def,
                  LLVMBuildExtractValue(ctx->builder, call, 0, ""));
   } else {
      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(ft, nc));
      for (unsigned c = 0; c < nc; c++) {
         LLVMValueRef comp = LLVMBuildExtractValue(ctx->builder, call,
                                                   c < 4 ? c : 3, "");
         vec = LLVMBuildInsertElement(ctx->builder, vec, comp,
                                      LLVMConstInt(i32, c, false), "");
      }
      set_ssa_def(ctx, &tex->def, vec);
   }
}

/* Why a sampler handle could not be matched, for CUDAPIPE_SPEC_STATS. The
 * matcher's failure is otherwise indistinguishable from a shader that simply
 * has no textures. */
static const char *cp_spec_reject_reason;

static bool
capture_tex_desc_ref(nir_tex_instr *tex, struct cp_tex_desc_ref *ref)
{
   nir_src *handle = NULL;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == nir_tex_src_sampler_handle) {
         handle = &tex->src[i].src;
         break;
      }
   }
   if (!handle) {
      cp_spec_reject_reason = "the instruction carries no sampler handle";
      return false;
   }

   nir_instr *parent = nir_def_instr(handle->ssa);

   /*
    * The first binding of a set is at offset zero, so nir_iadd_imm folds the
    * addition away and the handle *is* the base. The matcher only knew the
    * base + constant shape, so every texture at flat offset 0 was
    * unspecialisable -- which is both of gltfscenerendering's fragment
    * shaders and two of pbribl's, and it is why they reported 0 of 19,292
    * specialised launches while their handles were as static as any other's.
    */
   if (parent->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(parent)->intrinsic ==
          nir_intrinsic_load_const_buf_base_addr_cudapipe) {
      nir_intrinsic_instr *base = nir_instr_as_intrinsic(parent);
      if (!nir_src_is_const(base->src[0])) {
         cp_spec_reject_reason = "the constant-buffer slot is not constant";
         return false;
      }
      ref->ubo_slot = nir_src_as_uint(base->src[0]);
      ref->sampler_offset = 0;
      return true;
   }

   if (parent->type != nir_instr_type_alu ||
       nir_instr_as_alu(parent)->op != nir_op_iadd) {
      cp_spec_reject_reason = "the handle is not base + constant offset";
      return false;
   }
   nir_alu_instr *add = nir_instr_as_alu(parent);
   nir_src *base_src = NULL, *offset_src = NULL;
   for (unsigned i = 0; i < 2; i++) {
      nir_src *src = &add->src[i].src;
      nir_instr *src_parent = nir_def_instr(src->ssa);
      if (nir_src_is_const(*src))
         offset_src = src;
      else if (src_parent->type == nir_instr_type_intrinsic &&
               nir_instr_as_intrinsic(src_parent)->intrinsic ==
                  nir_intrinsic_load_const_buf_base_addr_cudapipe)
         base_src = src;
   }
   if (!base_src || !offset_src) {
      cp_spec_reject_reason =
         "neither operand is a constant-buffer base with a constant offset";
      return false;
   }
   nir_intrinsic_instr *base =
      nir_instr_as_intrinsic(nir_def_instr(base_src->ssa));
   if (!nir_src_is_const(base->src[0])) {
      cp_spec_reject_reason = "the constant-buffer slot is not constant";
      return false;
   }

   ref->ubo_slot = nir_src_as_uint(base->src[0]);
   ref->sampler_offset = nir_src_as_uint(*offset_src);
   return true;
}

static void
capture_tex_desc_refs(struct nir_shader *nir, struct cp_shader_binary *bin)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            int32_t flags = cp_tex_flags(tex);
            bool sampled = flags >= 0 &&
               (tex->op == nir_texop_tex || tex->op == nir_texop_txl ||
                tex->op == nir_texop_txb);
            if (!sampled)
               continue;
            bin->num_tex_instrs++;
            struct cp_tex_desc_ref ref = { .flags = flags };
            cp_spec_reject_reason = NULL;
            if (!capture_tex_desc_ref(tex, &ref)) {
               bin->tex_descs_dynamic = true;
               if (cp_debug->spec_stats && !bin->spec_rejected_reported) {
                  bin->spec_rejected_reported = true;
                  fprintf(stderr, "cudapipe: sampler handle not specialisable: "
                          "%s\n", cp_spec_reject_reason ? cp_spec_reject_reason
                                                        : "unknown");
               }
               continue;
            }
            bool seen = false;
            for (unsigned i = 0; i < bin->num_tex_descs; i++) {
               seen |= bin->tex_descs[i].ubo_slot == ref.ubo_slot &&
                       bin->tex_descs[i].sampler_offset == ref.sampler_offset;
            }
            if (!seen && bin->num_tex_descs < CP_MAX_TEX_DESCS)
               bin->tex_descs[bin->num_tex_descs++] = ref;
            else if (!seen)
               bin->tex_descs_dynamic = true;
         }
      }
   }
}

static bool
capture_hw_tex_ref_kind(nir_tex_instr *tex, nir_tex_src_type handle_kind,
                     nir_tex_src_type array_offset_kind,
                     struct cp_hw_tex_ref *ref)
{
   nir_src *handle = NULL;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type == handle_kind) {
         handle = &tex->src[i].src;
         break;
      }
   }
   if (!handle) {
      cp_spec_reject_reason = handle_kind == nir_tex_src_sampler_handle
         ? "the instruction carries no sampler handle"
         : "the instruction carries no texture handle";
      return false;
   }

   nir_instr *parent = nir_def_instr(handle->ssa);
   uint32_t base_offset = 0;
   if (parent->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(parent)->intrinsic ==
          nir_intrinsic_load_const_buf_base_addr_cudapipe) {
      nir_intrinsic_instr *base = nir_instr_as_intrinsic(parent);
      if (!nir_src_is_const(base->src[0])) {
         cp_spec_reject_reason = "the constant-buffer slot is not constant";
         return false;
      }
      ref->ubo_slot = nir_src_as_uint(base->src[0]);
   } else {
      if (parent->type != nir_instr_type_alu ||
          nir_instr_as_alu(parent)->op != nir_op_iadd) {
         cp_spec_reject_reason = "the handle is not base + constant offset";
         return false;
      }
      nir_alu_instr *add = nir_instr_as_alu(parent);
      nir_src *base_src = NULL, *offset_src = NULL;
      for (unsigned i = 0; i < 2; i++) {
         nir_src *src = &add->src[i].src;
         nir_instr *src_parent = nir_def_instr(src->ssa);
         if (nir_src_is_const(*src))
            offset_src = src;
         else if (src_parent->type == nir_instr_type_intrinsic &&
                  nir_instr_as_intrinsic(src_parent)->intrinsic ==
                     nir_intrinsic_load_const_buf_base_addr_cudapipe)
            base_src = src;
      }
      if (!base_src || !offset_src) {
         cp_spec_reject_reason =
            "neither operand is a constant-buffer base with a constant offset";
         return false;
      }
      nir_intrinsic_instr *base =
         nir_instr_as_intrinsic(nir_def_instr(base_src->ssa));
      if (!nir_src_is_const(base->src[0])) {
         cp_spec_reject_reason = "the constant-buffer slot is not constant";
         return false;
      }
      ref->ubo_slot = nir_src_as_uint(base->src[0]);
      base_offset = nir_src_as_uint(*offset_src);
   }

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      if (tex->src[i].src_type != array_offset_kind)
         continue;
      if (!nir_src_is_const(tex->src[i].src)) {
         cp_spec_reject_reason = "the descriptor array offset is dynamic";
         return false;
      }
      uint64_t extra = (uint64_t)nir_src_as_uint(tex->src[i].src) *
                       CPVK_DESCRIPTOR_SIZE;
      if (extra > UINT32_MAX - base_offset) {
         cp_spec_reject_reason = "the descriptor array offset overflows";
         return false;
      }
      base_offset += (uint32_t)extra;
   }
   ref->offset = base_offset;
   return true;
}

static bool
capture_hw_tex_site(nir_tex_instr *tex, struct cp_hw_tex_site *ref)
{
   if (!capture_hw_tex_ref_kind(tex, nir_tex_src_texture_handle,
                            nir_tex_src_texture_offset, &ref->image) ||
       !capture_hw_tex_ref_kind(tex, nir_tex_src_sampler_handle,
                            nir_tex_src_sampler_offset, &ref->sampler))
      return false;
   ref->flags = cp_tex_flags(tex);
   ref->op = (uint8_t)tex->op;
   ref->dim = (uint8_t)tex->sampler_dim;
   ref->coord_components = tex->coord_components;
   return true;
}

static void
capture_hw_tex_sites(struct nir_shader *nir, struct cp_shader_binary *bin)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            int32_t flags = cp_tex_flags(tex);
            bool sampled = flags >= 0 &&
               (tex->op == nir_texop_tex || tex->op == nir_texop_txl ||
                tex->op == nir_texop_txb || tex->op == nir_texop_txd);
            if (!sampled)
               continue;
            struct cp_hw_tex_site site = { .flags = flags };
            cp_spec_reject_reason = NULL;
            if (!capture_hw_tex_site(tex, &site) ||
                bin->num_hw_tex_sites >= CP_MAX_HW_TEX_SITES) {
               bin->hw_tex_dynamic = true;
               continue;
            }
            bin->hw_tex_sites[bin->num_hw_tex_sites++] = site;
         }
      }
   }
}

/* All-or-nothing static float4 CUDA texture-object execution gate. */
static bool
cp_hardware_texture_shader_eligible(struct nir_shader *nir)
{
   unsigned sampled = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            bool dim_ok = tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
                          tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ||
                          tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
            if (!dim_ok || tex->is_array || tex->is_shadow || tex->is_sparse ||
                tex->def.bit_size != 32 || tex->def.num_components != 4 ||
                nir_alu_type_get_base_type(tex->dest_type) != nir_type_float ||
                (tex->op != nir_texop_tex && tex->op != nir_texop_txl &&
                 tex->op != nir_texop_txb && tex->op != nir_texop_txd))
               return false;

            unsigned ncoord = 0, nlod = 0, nbias = 0, nddx = 0, nddy = 0;
            for (unsigned i = 0; i < tex->num_srcs; i++) {
               nir_tex_src_type src = tex->src[i].src_type;
               ncoord += src == nir_tex_src_coord;
               nlod += src == nir_tex_src_lod;
               nbias += src == nir_tex_src_bias;
               nddx += src == nir_tex_src_ddx;
               nddy += src == nir_tex_src_ddy;
               bool allowed = src == nir_tex_src_coord ||
                  src == nir_tex_src_texture_handle ||
                  src == nir_tex_src_sampler_handle ||
                  src == nir_tex_src_texture_offset ||
                  src == nir_tex_src_sampler_offset ||
                  (src == nir_tex_src_lod && tex->op == nir_texop_txl) ||
                  (src == nir_tex_src_bias && tex->op == nir_texop_txb) ||
                  ((src == nir_tex_src_ddx || src == nir_tex_src_ddy) &&
                   tex->op == nir_texop_txd);
               if (!allowed)
                  return false;
            }
            unsigned expected_coord = tex->sampler_dim == GLSL_SAMPLER_DIM_2D
               ? 2 : 3;
            if (tex->coord_components != expected_coord || ncoord != 1 ||
                (tex->op == nir_texop_tex && (nlod || nbias || nddx || nddy)) ||
                (tex->op == nir_texop_txl &&
                 (nlod != 1 || nbias || nddx || nddy)) ||
                (tex->op == nir_texop_txb &&
                 (nbias != 1 || nlod || nddx || nddy)) ||
                (tex->op == nir_texop_txd &&
                 (nddx != 1 || nddy != 1 || nlod || nbias)))
               return false;

            struct cp_hw_tex_site ref = {0};
            if (!capture_hw_tex_site(tex, &ref) || sampled >= CP_MAX_HW_TEX_SITES)
               return false;
            sampled++;
         }
      }
   }
   return sampled > 0;
}

static void
emit_block_instrs(struct ntl_context *ctx, nir_block *block)
{
   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_alu:
         emit_alu(ctx, nir_instr_as_alu(instr));
         break;
      case nir_instr_type_intrinsic:
         emit_intrinsic(ctx, nir_instr_as_intrinsic(instr));
         break;
      case nir_instr_type_load_const:
         emit_load_const(ctx, nir_instr_as_load_const(instr));
         break;
      case nir_instr_type_phi:
         break;
      case nir_instr_type_tex:
         emit_tex(ctx, nir_instr_as_tex(instr));
         break;
      case nir_instr_type_undef: {
         nir_undef_instr *undef = nir_instr_as_undef(instr);
         set_ssa_def(ctx, &undef->def,
            LLVMGetUndef(get_llvm_type(ctx, undef->def.bit_size, undef->def.num_components)));
         break;
      }
      case nir_instr_type_jump: {
         nir_jump_instr *jump = nir_instr_as_jump(instr);
         if (jump->type == nir_jump_break) {
            LLVMBuildBr(ctx->builder, ctx->break_block);
         } else if (jump->type == nir_jump_continue) {
            LLVMBuildBr(ctx->builder, ctx->continue_block);
         }
         break;
      }
      default:
         break;
      }
   }
}

static void emit_cf_list(struct ntl_context *ctx, struct exec_list *list);

static void
emit_if(struct ntl_context *ctx, nir_if *if_stmt)
{
   LLVMValueRef cond = get_src(ctx, &if_stmt->condition);
   /* Convert to i1 if needed */
   if (LLVMTypeOf(cond) != LLVMInt1TypeInContext(ctx->llvm_ctx)) {
      cond = LLVMBuildICmp(ctx->builder, LLVMIntNE, cond,
         LLVMConstNull(LLVMTypeOf(cond)), "");
   }

   LLVMBasicBlockRef then_block = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "then");
   LLVMBasicBlockRef else_block = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "else");
   LLVMBasicBlockRef merge_block = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "endif");

   LLVMBuildCondBr(ctx->builder, cond, then_block, else_block);

   LLVMPositionBuilderAtEnd(ctx->builder, then_block);
   emit_cf_list(ctx, &if_stmt->then_list);
   if (!LLVMGetBasicBlockTerminator(LLVMGetInsertBlock(ctx->builder)))
      LLVMBuildBr(ctx->builder, merge_block);

   LLVMPositionBuilderAtEnd(ctx->builder, else_block);
   emit_cf_list(ctx, &if_stmt->else_list);
   if (!LLVMGetBasicBlockTerminator(LLVMGetInsertBlock(ctx->builder)))
      LLVMBuildBr(ctx->builder, merge_block);

   LLVMPositionBuilderAtEnd(ctx->builder, merge_block);
}

static void
emit_loop(struct ntl_context *ctx, nir_loop *loop)
{
   LLVMBasicBlockRef loop_header = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "loop");
   LLVMBasicBlockRef loop_exit = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "endloop");

   /* Save previous break/continue targets */
   LLVMBasicBlockRef prev_break = ctx->break_block;
   LLVMBasicBlockRef prev_continue = ctx->continue_block;
   ctx->break_block = loop_exit;
   ctx->continue_block = loop_header;

   LLVMBuildBr(ctx->builder, loop_header);
   LLVMPositionBuilderAtEnd(ctx->builder, loop_header);

   emit_cf_list(ctx, &loop->body);

   /* If no terminator at end of loop body, branch back to header */
   if (!LLVMGetBasicBlockTerminator(LLVMGetInsertBlock(ctx->builder)))
      LLVMBuildBr(ctx->builder, loop_header);

   LLVMPositionBuilderAtEnd(ctx->builder, loop_exit);

   ctx->break_block = prev_break;
   ctx->continue_block = prev_continue;
}

static void
emit_cf_list(struct ntl_context *ctx, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         emit_block_instrs(ctx, nir_cf_node_as_block(node));
         break;
      case nir_cf_node_if:
         emit_if(ctx, nir_cf_node_as_if(node));
         break;
      case nir_cf_node_loop:
         emit_loop(ctx, nir_cf_node_as_loop(node));
         break;
      default:
         break;
      }
   }
}

static bool
emit_function(struct ntl_context *ctx)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(ctx->nir);

   ctx->num_ssa_defs = impl->ssa_alloc;
   ctx->ssa_defs = calloc(ctx->num_ssa_defs, sizeof(LLVMValueRef));
   ctx->num_regs = impl->ssa_alloc;
   ctx->regs = calloc(ctx->num_regs, sizeof(LLVMValueRef));

   LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "entry");
   LLVMPositionBuilderAtEnd(ctx->builder, entry);

   /* Where cp_arg_slot() puts its loads, so that one dominates every use. */
   ctx->entry_block = entry;
   memset(ctx->arg_slots, 0, sizeof(ctx->arg_slots));
   ctx->md_invariant_load =
      LLVMGetMDKindIDInContext(ctx->llvm_ctx, "invariant.load",
                               strlen("invariant.load"));

   if (ctx->inline_interp) {
      LLVMTypeRef i8_t = LLVMInt8TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef input_array = LLVMArrayType(i8_t, CP_MAX_FS_INPUTS * 16);
      ctx->inline_fs_inputs = LLVMBuildAlloca(
         ctx->builder, input_array, "inline_fs_inputs");
      LLVMSetAlignment(ctx->inline_fs_inputs, 16);
   }

   /*
    * Bounds check: the count is on the device, so both grids are sized for the
    * worst case and each thread bounds itself against slot 0 — the vertex
    * count for a vertex shader, the covered pixel count for a fragment one.
    *
    * The fragment case is the one that pays. Its grid covers the whole
    * framebuffer twice over, because a pixel list that quads may append to
    * more than once cannot be sized more tightly on the host, while a draw's
    * actual coverage is whatever it rasterized: over a frame of multithreading
    * the median draw covers 588 pixels and the grid carries 1,843,200. Without
    * this the shader ran its whole body on every one of them — sampling
    * textures on uninitialised varyings — and the writeback then dropped all
    * but the first `counter` results, which is why the arithmetic came out
    * right and the frame took 400 times longer than the work in it.
    */
   LLVMBasicBlockRef stride_latch = NULL;
   LLVMValueRef vbid_phi = NULL, stride_next = NULL;
   if (ctx->nir->info.stage == MESA_SHADER_VERTEX ||
       ctx->nir->info.stage == MESA_SHADER_FRAGMENT) {
      LLVMTypeRef i32_t = LLVMInt32TypeInContext(ctx->llvm_ctx);
      LLVMValueRef count = LLVMBuildLoad2(ctx->builder, i32_t,
         LLVMBuildBitCast(ctx->builder, cp_arg_slot(ctx, 0),
                          LLVMPointerType(i32_t, 0), ""), "invoc_count");
      /* Written by the launch before this one, so constant throughout this. */
      LLVMSetMetadata(count, ctx->md_invariant_load,
                      LLVMMDNodeInContext(ctx->llvm_ctx, NULL, 0));

      LLVMValueRef bid = emit_workgroup_id(ctx, 0);
      LLVMValueRef tid = emit_local_invocation_id(ctx, 0);
      LLVMValueRef nblocks =
         emit_nvptx_read_sreg(ctx, "llvm.nvvm.read.ptx.sreg.nctaid.x");

      /*
       * Grid-stride: the slot count lives on the device, so the host sizes
       * the grid for the worst case — the whole framebuffer twice over, for
       * a fragment shader — and may cap it. The loop's induction value is a
       * virtual block id that emit_workgroup_id() returns inside the body,
       * so every slot-derived address follows the loop; a grid at the worst
       * case runs each body exactly once, which is the code this replaces.
       */
      LLVMBasicBlockRef header = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "stride_head");
      LLVMBasicBlockRef body = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "shader_body");
      stride_latch = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "stride_latch");
      LLVMBasicBlockRef done = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "shader_ret");

      LLVMBuildBr(ctx->builder, header);
      LLVMPositionBuilderAtEnd(ctx->builder, header);
      vbid_phi = LLVMBuildPhi(ctx->builder, i32_t, "vbid");
      LLVMValueRef vid = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, vbid_phi,
                      LLVMConstInt(i32_t, 256, false), ""), tid, "");
      LLVMValueRef oob = LLVMBuildICmp(ctx->builder, LLVMIntUGE, vid, count, "");
      LLVMBuildCondBr(ctx->builder, oob, done, body);

      LLVMPositionBuilderAtEnd(ctx->builder, done);
      LLVMBuildRetVoid(ctx->builder);

      LLVMPositionBuilderAtEnd(ctx->builder, stride_latch);
      stride_next = LLVMBuildAdd(ctx->builder, vbid_phi, nblocks, "vbid_next");
      LLVMBuildBr(ctx->builder, header);

      LLVMValueRef inc_v[2] = { bid, stride_next };
      LLVMBasicBlockRef inc_b[2] = { entry, stride_latch };
      LLVMAddIncoming(vbid_phi, inc_v, inc_b, 2);

      LLVMPositionBuilderAtEnd(ctx->builder, body);
      ctx->virtual_bid = vbid_phi;

      if (ctx->nir->info.stage == MESA_SHADER_FRAGMENT &&
          (ctx->fused_interp || ctx->inline_interp)) {
         LLVMTypeRef i8_ptr = LLVMPointerType(
            LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
         LLVMValueRef interp = cp_arg_slot(ctx, CP_ARG_SLOT_FUSED_INTERP);
         LLVMBasicBlockRef call_interp = LLVMAppendBasicBlockInContext(
            ctx->llvm_ctx, ctx->function,
            ctx->inline_interp ? "inline_interp" : "fused_interp");
         LLVMBasicBlockRef after_interp = LLVMAppendBasicBlockInContext(
            ctx->llvm_ctx, ctx->function,
            ctx->inline_interp ? "inline_interp_done" : "fused_interp_done");

         if (ctx->inline_interp) {
            /* Inline executions are selected only for the compact/direct or
             * merged A-buffer chains, both of which always supply this block. */
            LLVMBuildBr(ctx->builder, call_interp);
            LLVMPositionBuilderAtEnd(ctx->builder, call_interp);
            LLVMTypeRef params[] = { i8_ptr, i32_t, i8_ptr, i32_t, i32_t,
                                     i32_t, i32_t };
            LLVMTypeRef helper_type = LLVMFunctionType(i32_t, params, 7, false);
            LLVMValueRef helper = LLVMGetNamedFunction(
               ctx->module, "cp_fs_inline_lane");
            if (!helper)
               helper = LLVMAddFunction(ctx->module, "cp_fs_inline_lane",
                                        helper_type);
            LLVMValueRef helper_args[] = {
               interp, vid,
               LLVMBuildBitCast(ctx->builder, ctx->inline_fs_inputs, i8_ptr, ""),
               LLVMConstInt(i32_t, MIN2(ctx->nir->num_inputs,
                                       CP_MAX_FS_INPUTS), false),
               LLVMConstInt(i32_t, ctx->inline_live_slots, false),
               LLVMConstInt(i32_t, (uint32_t)ctx->inline_pntc_input, false),
               LLVMConstInt(i32_t, (uint32_t)ctx->inline_pos_input, false),
            };
            LLVMValueRef valid = LLVMBuildCall2(ctx->builder, helper_type, helper,
                                                 helper_args, 7, "interp_valid");
            valid = LLVMBuildICmp(ctx->builder, LLVMIntNE, valid,
                                 LLVMConstInt(i32_t, 0, false), "");
            LLVMBuildCondBr(ctx->builder, valid, after_interp, stride_latch);
         } else {
            LLVMValueRef enabled = LLVMBuildICmp(ctx->builder, LLVMIntNE, interp,
               LLVMConstNull(i8_ptr), "fused_interp_enabled");
            LLVMBuildCondBr(ctx->builder, enabled, call_interp, after_interp);
            LLVMPositionBuilderAtEnd(ctx->builder, call_interp);
            LLVMTypeRef params[] = { i8_ptr, i32_t };
            LLVMTypeRef helper_type = LLVMFunctionType(i32_t, params, 2, false);
            LLVMValueRef helper = LLVMGetNamedFunction(
               ctx->module, "cp_abuf_interpolate_lane");
            if (!helper)
               helper = LLVMAddFunction(ctx->module, "cp_abuf_interpolate_lane",
                                        helper_type);
            LLVMValueRef helper_args[] = { interp, vid };
            LLVMValueRef valid = LLVMBuildCall2(ctx->builder, helper_type, helper,
                                                 helper_args, 2, "interp_valid");
            valid = LLVMBuildICmp(ctx->builder, LLVMIntNE, valid,
                                 LLVMConstInt(i32_t, 0, false), "");
            LLVMBuildCondBr(ctx->builder, valid, after_interp, stride_latch);
         }
         LLVMPositionBuilderAtEnd(ctx->builder, after_interp);
      }

      /*
       * Helper invocations, for a fragment shader that writes memory.
       *
       * A quad is shaded whole so that derivatives can be taken across it, so
       * a lane the primitive does not cover is shaded too. Vulkan says such a
       * lane's stores and atomics have no effect; cp_fs_writeback enforces
       * that for colour and depth by consulting the same mask, and until a
       * fragment shader could write anything else there was nothing more to
       * enforce. There is now: `oit` appends a node to a per-pixel linked list
       * from the fragment stage, and a helper lane appends one for a pixel the
       * triangle never covered.
       *
       * Emitted only when the shader writes memory, so a shader that does not
       * compiles to exactly what it compiled to before — and the slot is null
       * for one anyway. Such a shader also forfeits its helper lanes'
       * derivatives, which is the price of the rule; nothing in this tree both
       * writes memory and differentiates.
       */
      if (ctx->nir->info.stage == MESA_SHADER_FRAGMENT && ctx->writes_memory) {
         LLVMTypeRef i8_t = LLVMInt8TypeInContext(ctx->llvm_ctx);
         LLVMValueRef cov_arr = cp_arg_slot(ctx, CP_ARG_SLOT_COVERAGE);
         LLVMValueRef cov_ptr = LLVMBuildGEP2(ctx->builder, i8_t,
            LLVMBuildBitCast(ctx->builder, cov_arr,
                             LLVMPointerType(i8_t, 0), ""), &vid, 1, "");
         LLVMValueRef cov = LLVMBuildLoad2(ctx->builder, i8_t, cov_ptr, "coverage");
         LLVMSetAlignment(cov, 1);
         LLVMValueRef helper = LLVMBuildICmp(ctx->builder, LLVMIntEQ, cov,
            LLVMConstInt(i8_t, 0, false), "");

         /* A helper lane skips this slot and strides on to its next one. */
         LLVMBasicBlockRef body2 = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "shader_body_covered");
         LLVMBuildCondBr(ctx->builder, helper, stride_latch, body2);
         LLVMPositionBuilderAtEnd(ctx->builder, body2);
      }
   }

   emit_cf_list(ctx, &impl->body);

   /* Close the body: back to the stride loop's latch for a drawing stage,
    * a plain return for compute. */
   if (!LLVMGetBasicBlockTerminator(LLVMGetInsertBlock(ctx->builder))) {
      if (stride_latch)
         LLVMBuildBr(ctx->builder, stride_latch);
      else
         LLVMBuildRetVoid(ctx->builder);
   }
   ctx->virtual_bid = NULL;

   free(ctx->ssa_defs);
   free(ctx->regs);
   return true;
}

static once_flag cp_nvptx_once = ONCE_FLAG_INIT;

static void
cp_initialize_nvptx(void)
{
   LLVMInitializeNVPTXTargetInfo();
   LLVMInitializeNVPTXTarget();
   LLVMInitializeNVPTXTargetMC();
   LLVMInitializeNVPTXAsmPrinter();
}

#ifdef CP_HAVE_FS_INLINE_BC
static bool
link_and_inline_fs_interp(LLVMModuleRef module, LLVMContextRef llvm_ctx,
                          int sm_major, int sm_minor)
{
   const char triple[] = "nvptx64-nvidia-cuda";
   char cpu[16];
   int llvm_major = sm_major, llvm_minor = sm_minor;
   if (llvm_major * 10 + llvm_minor > CP_MAX_PTX_SM) {
      llvm_major = CP_MAX_PTX_SM / 10;
      llvm_minor = CP_MAX_PTX_SM % 10;
   }
   snprintf(cpu, sizeof(cpu), "sm_%d%d", llvm_major, llvm_minor);
   call_once(&cp_nvptx_once, cp_initialize_nvptx);

   char *error = NULL;
   LLVMTargetRef target = NULL;
   if (LLVMGetTargetFromTriple(triple, &target, &error) != 0) {
      fprintf(stderr, "cudapipe: inline FS target lookup failed: %s\n",
              error ? error : "?");
      LLVMDisposeMessage(error);
      return false;
   }
   LLVMTargetMachineRef tm = LLVMCreateTargetMachine(
      target, triple, cpu, "+ptx75", LLVMCodeGenLevelDefault,
      LLVMRelocDefault, LLVMCodeModelDefault);
   if (!tm)
      return false;
   LLVMTargetDataRef td = LLVMCreateTargetDataLayout(tm);
   char *layout = LLVMCopyStringRepOfTargetData(td);
   LLVMSetTarget(module, triple);
   LLVMSetDataLayout(module, layout);

   LLVMMemoryBufferRef buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
      (const char *)cp_fs_inline_bc, sizeof(cp_fs_inline_bc), "cp_fs_inline.bc");
   LLVMModuleRef helper_module = NULL;
   bool ok = LLVMParseBitcodeInContext2(llvm_ctx, buffer, &helper_module) == 0;
   LLVMDisposeMemoryBuffer(buffer);
   if (!ok || !helper_module) {
      fprintf(stderr, "cudapipe: inline FS LLVM %d bitcode parse failed\n",
              LLVM_VERSION_MAJOR);
      goto out;
   }
   if (strcmp(LLVMGetTarget(helper_module), triple) != 0 ||
       strcmp(LLVMGetDataLayoutStr(helper_module), layout) != 0) {
      fprintf(stderr, "cudapipe: inline FS bitcode target/layout mismatch "
              "(LLVM %d)\n", LLVM_VERSION_MAJOR);
      LLVMDisposeModule(helper_module);
      ok = false;
      goto out;
   }
   if (LLVMLinkModules2(module, helper_module) != 0) {
      fprintf(stderr, "cudapipe: inline FS LLVM module link failed\n");
      ok = false;
      goto out;
   }

   LLVMValueRef helper = LLVMGetNamedFunction(module, "cp_fs_inline_lane");
   if (!helper) {
      fprintf(stderr, "cudapipe: inline FS helper absent after link\n");
      ok = false;
      goto out;
   }
   LLVMSetLinkage(helper, LLVMInternalLinkage);

   LLVMPassBuilderOptionsRef opts = LLVMCreatePassBuilderOptions();
   LLVMErrorRef pass_error = LLVMRunPasses(
      module,
      "always-inline,function(sroa,mem2reg,sccp,simplifycfg,loop-unroll,"
      "sccp,sroa,mem2reg,gvn,dse,adce,simplifycfg),globaldce",
      tm, opts);
   LLVMDisposePassBuilderOptions(opts);
   if (pass_error) {
      char *message = LLVMGetErrorMessage(pass_error);
      fprintf(stderr, "cudapipe: inline FS LLVM optimization failed: %s\n",
              message ? message : "?");
      LLVMDisposeErrorMessage(message);
      ok = false;
      goto out;
   }
   if (LLVMGetNamedFunction(module, "cp_fs_inline_lane")) {
      fprintf(stderr, "cudapipe: inline FS helper survived forced inlining\n");
      ok = false;
      goto out;
   }
   ok = true;

out:
   LLVMDisposeMessage(layout);
   LLVMDisposeTargetData(td);
   LLVMDisposeTargetMachine(tm);
   return ok;
}
#endif

static char *
compile_module_to_ptx(LLVMModuleRef module, int sm_major, int sm_minor, size_t *out_size)
{
   char triple[] = "nvptx64-nvidia-cuda";
   char cpu[16];

   /* LLVM's NVPTX backend only knows the architectures that existed when it
    * was released; on anything newer it warns and silently falls back to a
    * default that produces PTX the driver rejects. PTX is forward compatible,
    * so target the newest architecture this LLVM understands and let the
    * driver JIT it for the actual GPU. */
   int llvm_major = sm_major;
   int llvm_minor = sm_minor;
   if (llvm_major * 10 + llvm_minor > CP_MAX_PTX_SM) {
      llvm_major = CP_MAX_PTX_SM / 10;
      llvm_minor = CP_MAX_PTX_SM % 10;
   }
   snprintf(cpu, sizeof(cpu), "sm_%d%d", llvm_major, llvm_minor);

   call_once(&cp_nvptx_once, cp_initialize_nvptx);

   char *error = NULL;
   LLVMTargetRef target;
   if (LLVMGetTargetFromTriple(triple, &target, &error) != 0) {
      fprintf(stderr, "cudapipe: failed to get NVPTX target: %s\n", error);
      LLVMDisposeMessage(error);
      return NULL;
   }

   LLVMTargetMachineRef tm = LLVMCreateTargetMachine(
      target, triple, cpu, "+ptx75",
      LLVMCodeGenLevelDefault,
      LLVMRelocDefault,
      LLVMCodeModelDefault);

   if (!tm) {
      fprintf(stderr, "cudapipe: failed to create target machine\n");
      return NULL;
   }

   LLVMSetTarget(module, triple);
   LLVMSetDataLayout(module, LLVMCopyStringRepOfTargetData(LLVMCreateTargetDataLayout(tm)));

   LLVMMemoryBufferRef buf = NULL;
   if (LLVMTargetMachineEmitToMemoryBuffer(tm, module, LLVMAssemblyFile, &error, &buf) != 0) {
      if (error) {
         fprintf(stderr, "cudapipe: PTX emission failed: %s\n", error);
         LLVMDisposeMessage(error);
      }
      LLVMDisposeTargetMachine(tm);
      return NULL;
   }

   size_t ptx_size = LLVMGetBufferSize(buf);
   char *ptx = malloc(ptx_size + 1);
   memcpy(ptx, LLVMGetBufferStart(buf), ptx_size);
   ptx[ptx_size] = '\0';

   if (out_size)
      *out_size = ptx_size;

   LLVMDisposeMemoryBuffer(buf);
   LLVMDisposeTargetMachine(tm);
   return ptx;
}

/*
 * Reduce the shader to the subset this backend emits.
 *
 * The PTX emitter has no scratch memory and no support for derefs, so local
 * variables have to become SSA values or if-else trees, and variable-mode
 * copies have to be expanded. Running the optimiser afterwards is not just for
 * speed: constant folding is what turns most dynamic array indices into
 * constant ones, which keeps the if-else trees small.
 */
static void
cp_lower_nir(struct nir_shader *nir)
{
   bool progress;

   NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
   NIR_PASS(progress, nir, nir_split_var_copies);
   NIR_PASS(progress, nir, nir_lower_var_copies);
   NIR_PASS(progress, nir, nir_lower_global_vars_to_local);

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
   } while (progress);

   /* Anything still indexed dynamically becomes a select tree, since there is
    * nowhere else to put it. */
   NIR_PASS(progress, nir, nir_lower_indirect_derefs_to_if_else_trees,
            nir_var_function_temp | nir_var_shader_in | nir_var_shader_out,
            UINT32_MAX);

   NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
   NIR_PASS(progress, nir, nir_remove_dead_variables,
            nir_var_function_temp, NULL);

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);
}

/* Record which varying location each I/O slot carries, so the fragment
 * shader's inputs can be matched to the vertex shader's outputs by location. */
static void
capture_io_locations(struct nir_shader *nir, struct cp_shader_binary *bin)
{
   for (unsigned i = 0; i < CP_MAX_IO_SLOTS; i++) {
      bin->in_location[i] = VARYING_SLOT_MAX;
      bin->out_location[i] = VARYING_SLOT_MAX;
   }

   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in) {
      unsigned slots = glsl_count_attribute_slots(var->type, false);
      for (unsigned s = 0; s < slots; s++) {
         unsigned slot = var->data.driver_location + s;
         if (slot < CP_MAX_IO_SLOTS)
            bin->in_location[slot] = var->data.location + s;
      }
   }

   nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
      unsigned slots = glsl_count_attribute_slots(var->type, false);
      for (unsigned s = 0; s < slots; s++) {
         unsigned slot = var->data.driver_location + s;
         if (slot < CP_MAX_IO_SLOTS)
            bin->out_location[slot] = var->data.location + s;
      }
   }
}

/*
 * The block size every compiled shader is launched with. Both stages use it:
 * the vertex shader over (vertices + 255) / 256 blocks and the fragment shader
 * over a fixed worst case, cp_context.c:994 and :3256. It is a constant of the
 * driver rather than something the JIT can be left to guess at, which is what
 * makes the occupancy arithmetic below exact rather than a heuristic.
 */
#define CP_SHADER_BLOCK_THREADS 256

/*
 * How many of those blocks the register policy tries to fit on an SM.
 *
 * Occupancy at a fixed block size is a step function of the register count —
 * blocks per SM is floor(registers_per_sm / (threads * regs_per_thread)) — so
 * the only caps worth applying are the ones that land on a step. Measured on
 * gltfscenerendering, whose fragment shader is 195 registers and one block,
 * 300 frames on the sweep's orbit:
 *
 *   cap    off     96     112     128     144     160     176
 *   ms   15.06  13.49   13.54   13.57   15.52   14.63   15.08
 *
 * The three caps that reach two blocks are worth 10% and are indistinguishable
 * from each other; the three that leave it at one are indistinguishable from
 * no cap at all. So what is being bought is the step, not the register count,
 * and the cap worth applying is the largest one that reaches the step —
 * cutting further only spills more for no more warps.
 */
#define CP_SHADER_TARGET_BLOCKS 2

/* Load a shader module, optionally capping the register count.
 *
 * `sampler_ptx` non-NULL links the sampler's relocatable PTX in first; the
 * rest load their own PTX directly. Both paths take the same JIT options,
 * which the direct one did not before — CUDAPIPE_MAX_REGISTERS reached the
 * linked shaders only, so a vertex shader that samples no texture was never
 * capped whatever it was set to.
 */
static CUresult
load_shader_module(CUmodule *module, const char *shader_ptx,
                   const char *sampler_ptx, const char *fs_helper_ptx,
                   int max_regs)
{
   CUjit_option jit_opts[5];
   void *jit_vals[5];
   char error_log[16384] = {0};
   char info_log[16384] = {0};
   unsigned num_jit = 0;
   if (max_regs > 0) {
      jit_opts[num_jit] = CU_JIT_MAX_REGISTERS;
      jit_vals[num_jit] = (void *)(uintptr_t)max_regs;
      num_jit++;
   }
   jit_opts[num_jit] = CU_JIT_ERROR_LOG_BUFFER;
   jit_vals[num_jit++] = error_log;
   jit_opts[num_jit] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
   jit_vals[num_jit++] = (void *)(uintptr_t)sizeof(error_log);
   jit_opts[num_jit] = CU_JIT_INFO_LOG_BUFFER;
   jit_vals[num_jit++] = info_log;
   jit_opts[num_jit] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
   jit_vals[num_jit++] = (void *)(uintptr_t)sizeof(info_log);

   if (!sampler_ptx && !fs_helper_ptx) {
      CUresult err = cuModuleLoadDataEx(module, shader_ptx, num_jit, jit_opts,
                                       jit_vals);
      if (err != CUDA_SUCCESS)
         fprintf(stderr, "cudapipe: CUDA JIT error log:\n%s\ninfo:\n%s\n",
                 error_log[0] ? error_log : "(empty)",
                 info_log[0] ? info_log : "(empty)");
      return err;
   }

   CUlinkState link;
   CUresult err = cuLinkCreate(num_jit, jit_opts, jit_vals, &link);
   if (err != CUDA_SUCCESS)
      return err;

   if (sampler_ptx)
      err = cuLinkAddData(link, CU_JIT_INPUT_PTX, (void *)sampler_ptx,
                          strlen(sampler_ptx) + 1, "cp_sampler.ptx", 0,
                          NULL, NULL);
   if (err == CUDA_SUCCESS && fs_helper_ptx)
      err = cuLinkAddData(link, CU_JIT_INPUT_PTX, (void *)fs_helper_ptx,
                          strlen(fs_helper_ptx) + 1, "cp_fs_helper.ptx", 0,
                          NULL, NULL);
   if (err == CUDA_SUCCESS)
      err = cuLinkAddData(link, CU_JIT_INPUT_PTX, (void *)shader_ptx,
                          strlen(shader_ptx) + 1, "shader.ptx", 0, NULL, NULL);

   void *cubin = NULL;
   size_t cubin_size = 0;
   if (err == CUDA_SUCCESS)
      err = cuLinkComplete(link, &cubin, &cubin_size);
   if (err == CUDA_SUCCESS)
      err = cuModuleLoadData(module, cubin);
   if (err != CUDA_SUCCESS)
      fprintf(stderr, "cudapipe: CUDA link/JIT error log:\n%s\ninfo:\n%s\n",
              error_log[0] ? error_log : "(empty)",
              info_log[0] ? info_log : "(empty)");

   cuLinkDestroy(link);
   return err;
}

/* What one built module costs in resources, and what that buys in warps. */
struct cp_shader_cost {
   int regs;         /* registers per thread */
   int spill;        /* bytes of local memory per thread — 0 is no spilling */
   int blocks;       /* 256-thread blocks resident per SM, from the driver */
};

static void
measure_shader_cost(CUfunction fn, struct cp_shader_cost *cost)
{
   cost->regs = cost->spill = cost->blocks = 0;
   cuFuncGetAttribute(&cost->regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
   cuFuncGetAttribute(&cost->spill, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn);
   /* Ask the driver rather than dividing by hand: it knows the allocation
    * granularity, the warp and block limits and the shared memory this
    * function reserves, and none of those are the same on every device. */
   cuOccupancyMaxActiveBlocksPerMultiprocessor(&cost->blocks, fn,
                                               CP_SHADER_BLOCK_THREADS, 0);
}

static int cp_pick_reg_cap(void);

static bool
cp_shader_build_sampler_variant_exec(struct cp_shader_binary *bin,
                                     struct cp_sampler_variant *variant,
                                     const char *sampler_ptx,
                                     enum cp_shader_exec_mode mode)
{
   struct cp_shader_exec *base = &bin->exec[mode];
   struct cp_shader_exec *exec = &variant->exec[mode];
   if (exec->kernel)
      return true;
   if (!base->kernel || !base->ptx_text)
      return false;

   /* A base shader's runtime tuning verdict does not transfer to a specialised
    * sampler binary.  Start variants as built (except for an explicit static
    * policy), then give this execution its own trial below. */
   int initial_cap = (cp_debug->max_registers || cp_debug->regcap_static)
      ? base->reg_cap : 0;
   CUmodule module = NULL;
   CUfunction kernel = NULL;
   if (load_shader_module(&module, base->ptx_text, sampler_ptx,
                          base->fs_helper_ptx, initial_cap) != CUDA_SUCCESS)
      return false;
   if (cuModuleGetFunction(&kernel, module, "main") != CUDA_SUCCESS) {
      cuModuleUnload(module);
      return false;
   }

   struct cp_shader_cost cost;
   measure_shader_cost(kernel, &cost);
   exec->ptx_text = base->ptx_text; /* borrowed from the base execution */
   exec->ptx_size = base->ptx_size;
   exec->module = module;
   exec->kernel = kernel;
   exec->num_regs = cost.regs;
   exec->spill_bytes = cost.spill;
   exec->blocks_per_sm = cost.blocks;
   exec->reg_cap = initial_cap;
   exec->fs_helper_ptx = base->fs_helper_ptx;
   exec->last_sampler_table = ~(uint64_t)0;
   exec->last_quad_derivs = -1;
   exec->tune_done = true;

   if (!cp_debug->no_regcap && !cp_debug->max_registers &&
       !cp_debug->regcap_static && !initial_cap &&
       cost.blocks >= 1 && cost.blocks < CP_SHADER_TARGET_BLOCKS) {
      int cap = cp_pick_reg_cap();
      if (cap > 0 && cap < cost.regs) {
         CUmodule alt = NULL;
         CUfunction alt_fn = NULL;
         struct cp_shader_cost alt_cost = {0};
         if (load_shader_module(&alt, base->ptx_text, sampler_ptx,
                                base->fs_helper_ptx, cap) == CUDA_SUCCESS &&
             cuModuleGetFunction(&alt_fn, alt, "main") == CUDA_SUCCESS)
            measure_shader_cost(alt_fn, &alt_cost);
         if (alt_fn && alt_cost.blocks > cost.blocks) {
            exec->alt_module = alt;
            exec->alt_kernel = alt_fn;
            exec->alt_regs = alt_cost.regs;
            exec->alt_spill_bytes = alt_cost.spill;
            exec->alt_blocks_per_sm = alt_cost.blocks;
            exec->alt_reg_cap = cap;
            exec->alt_last_sampler_table = ~(uint64_t)0;
            exec->alt_last_quad_derivs = -1;
            exec->tune_cap = cap;
            exec->tune_done = false;
            if (cp_debug->shader_stats)
               fprintf(stderr, "cudapipe: sampler variant %-7s regs %3d "
                       "blocks/sm %d -> cap %d: regs %3d blocks/sm %d  "
                       "on trial\n",
                       mode == CP_SHADER_EXEC_INLINE ? "inline" :
                       (mode == CP_SHADER_EXEC_FUSED ? "fused" : "classic"),
                       cost.regs, cost.blocks, cap, alt_cost.regs,
                       alt_cost.blocks);
         } else if (alt) {
            cuModuleUnload(alt);
         }
      }
   }

   return true;
}

bool
cp_shader_build_sampler_variant(struct cp_shader_binary *bin,
                                const char *sampler_ptx,
                                const struct cp_sampler_info *states,
                                unsigned num_states,
                                enum cp_shader_exec_mode mode)
{
   if (!bin || !sampler_ptx || !states || !num_states)
      return false;

   struct cp_sampler_variant *variant =
      cp_shader_find_sampler_variant(bin, states, num_states);
   bool new_variant = !variant;
   if (new_variant) {
      if (bin->num_sampler_variants >= CP_MAX_SAMPLER_VARIANTS)
         return false;
      variant = &bin->sampler_variants[bin->num_sampler_variants];
      variant->states = malloc((size_t)num_states * sizeof(*states));
      if (!variant->states)
         return false;
      memcpy(variant->states, states, (size_t)num_states * sizeof(*states));
      variant->num_states = num_states;
   }

   if (!cp_shader_build_sampler_variant_exec(bin, variant, sampler_ptx,
                                              mode)) {
      /* An existing variant may already own successful executions for other
       * modes; leave those intact. A new one is not visible until its first
       * execution succeeds, so roll its provisional key back completely. */
      if (new_variant) {
         free(variant->states);
         memset(variant, 0, sizeof(*variant));
      }
      return false;
   }
   if (new_variant)
      bin->num_sampler_variants++;
   return true;
}

struct cp_sampler_variant *
cp_shader_find_sampler_variant(struct cp_shader_binary *bin,
                               const struct cp_sampler_info *states,
                               unsigned num_states)
{
   if (!bin || !states || !num_states)
      return NULL;
   for (unsigned i = 0; i < bin->num_sampler_variants; i++) {
      struct cp_sampler_variant *variant = &bin->sampler_variants[i];
      if (variant->num_states == num_states &&
          !memcmp(variant->states, states,
                  (size_t)num_states * sizeof(*states)))
         return variant;
   }
   return NULL;
}


/* Rebuild an already-loaded shader with a different register cap, in place.
 *
 * This is the compile-time shape of the policy — CUDAPIPE_REGCAP_STATIC — and
 * the reason the PTX and the sampler it was built from are kept on the binary.
 * The default policy does not use it: it loads both builds and switches
 * between them, because a rebuild between two draws is a JIT link on a hot
 * path. The caller must have drained the stream if anything of the module
 * being replaced may still be in flight.
 *
 * Returns false and leaves the shader exactly as it was if anything fails,
 * including the JIT declining the cap.
 */
static bool
cp_shader_exec_set_reg_cap(struct cp_shader_exec *exec, int max_regs)
{
   if (!exec || !exec->ptx_text || exec->reg_cap == max_regs)
      return false;

   CUmodule mod = NULL;
   CUfunction fn = NULL;
   if (load_shader_module(&mod, exec->ptx_text, exec->sampler_ptx,
                          exec->fs_helper_ptx,
                          max_regs) != CUDA_SUCCESS)
      return false;
   if (cuModuleGetFunction(&fn, mod, "main") != CUDA_SUCCESS) {
      cuModuleUnload(mod);
      return false;
   }

   struct cp_shader_cost cost;
   measure_shader_cost(fn, &cost);

   cuModuleUnload(exec->module);
   exec->module = mod;
   exec->kernel = fn;
   exec->num_regs = cost.regs;
   exec->spill_bytes = cost.spill;
   exec->blocks_per_sm = cost.blocks;
   exec->reg_cap = max_regs;

   /* The sampler reads its table and its quad-derivative flag through globals
    * of the module, so both addresses and both cached values belong to the
    * module that has just been thrown away. */
   exec->globals_resolved = false;
   exec->sym_sampler_table = 0;
   exec->sym_quad_derivs = 0;

   return true;
}

/*
 * Load the shader, and work out whether capping its registers is even a
 * question worth asking about it.
 *
 * The JIT optimises for instruction-level parallelism and spends registers
 * freely doing it. On Sponza the fragment shader comes out at 195 per thread,
 * which fits one 256-thread block on an SM and caps theoretical occupancy at
 * 16.7%; ncu measures it there with SM and memory throughput both under 8%,
 * no spilling at all, and warps stalled on long_scoreboard and wait — global
 * load latency and fixed-latency dependencies, both of which are hidden by
 * having more warps and by nothing else. Capping it to 128 buys the second
 * block and is worth 10% of that sample's frame.
 *
 * A fixed cap is still wrong, and the reason is sharper than "the trade is
 * per-shader". **The register count of every textured fragment shader in this
 * driver is 195**, whatever the shader does: it is the linked sampler's
 * allocation, not the shader's, and it comes out identical for Sponza's PBR
 * material and for bloom's blur. So `CU_FUNC_ATTRIBUTE_NUM_REGS` — the signal
 * the performance plan proposed deciding on — carries no per-shader
 * information here at all. Capping on it alone is a fixed cap wearing a
 * measurement's clothes, and it costs bloom 3.2% while paying 9% on Sponza.
 *
 * What it does tell you is which shaders the question is *about*: this
 * function loads the shader as the JIT built it, and marks it for the runtime
 * trial in cp_context.c if, and only if, its own register count is what is
 * holding its occupancy below the target. Everything else is loaded once,
 * never rebuilt, and never timed.
 *
 * Only fragment shaders are candidates. The vertex stage runs over the vertex
 * count, which is one or two blocks on a 170-SM card for most draws in this
 * set, so its occupancy is bounded by the grid long before it is bounded by
 * registers — and measured, none of the sweep's vertex shaders reaches the
 * threshold anyway (22-90 registers, two blocks or better).
 *
 * CUDAPIPE_MAX_REGISTERS overrides everything with a fixed cap on every
 * shader, which is what it always did and is how a regression gets bisected.
 * CUDAPIPE_REGCAP_STATIC=1 applies the cap at compile time without the trial,
 * which is the shape the plan proposed and is kept so it can be measured.
 * CUDAPIPE_NO_REGCAP disables all of it.
 */
/* The largest cap that fits the occupancy target, straight out of the
 * device: on a 64K-register SM at 256 threads and two blocks that is 128. */
static int
cp_pick_reg_cap(void)
{
   CUdevice dev;
   int regs_per_sm = 0;
   if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
      cuDeviceGetAttribute(&regs_per_sm,
                           CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                           dev);
   return regs_per_sm / (CP_SHADER_TARGET_BLOCKS * CP_SHADER_BLOCK_THREADS);
}

static CUresult
load_shader_module_tuned(struct cp_shader_exec *exec, const char *ptx,
                         const char *sampler_ptx, const char *fs_helper_ptx,
                         bool is_fragment,
                         const char *stage)
{
   const int forced = (int)cp_debug->max_registers;
   const int disabled = cp_debug->no_regcap;
   const int use_static = cp_debug->regcap_static;
   const bool stats = cp_debug->shader_stats;

   exec->sampler_ptx = sampler_ptx;
   exec->fs_helper_ptx = fs_helper_ptx;

   CUresult err = load_shader_module(&exec->module, ptx, sampler_ptx,
                                     fs_helper_ptx, forced);
   if (err != CUDA_SUCCESS)
      return err;
   err = cuModuleGetFunction(&exec->kernel, exec->module, "main");
   if (err != CUDA_SUCCESS)
      return err;

   struct cp_shader_cost cost;
   measure_shader_cost(exec->kernel, &cost);
   exec->num_regs = cost.regs;
   exec->spill_bytes = cost.spill;
   exec->blocks_per_sm = cost.blocks;
   exec->reg_cap = forced;

   const char *why = NULL;
   if (forced)
      why = "capped by CUDAPIPE_MAX_REGISTERS";
   else if (disabled)
      why = "policy off";
   else if (!is_fragment)
      why = "kept (not a fragment shader)";
   else if (cost.blocks < 1 || cost.blocks >= CP_SHADER_TARGET_BLOCKS)
      why = "kept (already at the target)";

   int cap = 0;
   if (!why) {
      cap = cp_pick_reg_cap();
      if (cap <= 0)
         why = "kept (device has no register count)";
      else if (cap >= cost.regs)
         /* Shared memory or a block limit is holding it down, and no register
          * cap can move either. */
         why = "kept (not register bound)";
   }

   if (why) {
      if (stats)
         fprintf(stderr, "cudapipe: shader %-21s regs %3d spill %4d "
                 "blocks/sm %d  %s\n", stage, cost.regs, cost.spill,
                 cost.blocks, why);
      return CUDA_SUCCESS;
   }

   if (use_static) {
      bool ok = cp_shader_exec_set_reg_cap(exec, cap);
      if (stats)
         fprintf(stderr, "cudapipe: shader %-21s regs %3d spill %4d "
                 "blocks/sm %d  static cap %d -> regs %3d spill %4d "
                 "blocks/sm %d\n", stage, cost.regs, cost.spill, cost.blocks,
                 cap, exec->num_regs, exec->spill_bytes, exec->blocks_per_sm);
      (void)ok;
      return CUDA_SUCCESS;
   }

   /*
    * A candidate: build the capped variant now, beside the one the JIT chose,
    * and let the driver time both against real draws before picking. Both are
    * built here, where pipeline creation already expects to pay for a JIT
    * link, rather than one of them in the middle of the run where it would be
    * a JIT link and a stream drain between two draws.
    */
   CUmodule alt = NULL;
   CUfunction alt_fn = NULL;
   struct cp_shader_cost alt_cost = {0};
   if (load_shader_module(&alt, ptx, sampler_ptx, fs_helper_ptx, cap) == CUDA_SUCCESS &&
       cuModuleGetFunction(&alt_fn, alt, "main") == CUDA_SUCCESS) {
      measure_shader_cost(alt_fn, &alt_cost);
   } else if (alt) {
      cuModuleUnload(alt);
      alt = NULL;
   }

   /* Only worth trying if the JIT actually met the cap and it bought a block.
    * Otherwise the shader is left exactly as it was and never timed. */
   if (!alt_fn || alt_cost.blocks <= cost.blocks) {
      if (alt)
         cuModuleUnload(alt);
      if (stats)
         fprintf(stderr, "cudapipe: shader %-21s regs %3d spill %4d "
                 "blocks/sm %d  kept (cap %d buys nothing)\n", stage,
                 cost.regs, cost.spill, cost.blocks, cap);
      return CUDA_SUCCESS;
   }

   exec->tune_cap = cap;
   exec->alt_module = alt;
   exec->alt_kernel = alt_fn;
   exec->alt_regs = alt_cost.regs;
   exec->alt_spill_bytes = alt_cost.spill;
   exec->alt_blocks_per_sm = alt_cost.blocks;
   exec->alt_reg_cap = cap;
   exec->alt_last_sampler_table = ~(uint64_t)0;
   exec->alt_last_quad_derivs = -1;

   if (stats)
      fprintf(stderr, "cudapipe: shader %-21s regs %3d spill %4d blocks/sm %d "
              "-> cap %d: regs %3d spill %4d blocks/sm %d  on trial\n",
              stage, cost.regs, cost.spill, cost.blocks, cap, alt_cost.regs,
              alt_cost.spill, alt_cost.blocks);

   return CUDA_SUCCESS;
}

void
cp_shader_exec_swap_build(struct cp_shader_exec *exec)
{
   if (!exec->alt_module)
      return;

#define CP_SWAP(type, a, b) do { type tmp = (a); (a) = (b); (b) = tmp; } while (0)
   CP_SWAP(CUmodule, exec->module, exec->alt_module);
   CP_SWAP(CUfunction, exec->kernel, exec->alt_kernel);
   CP_SWAP(int, exec->num_regs, exec->alt_regs);
   CP_SWAP(int, exec->spill_bytes, exec->alt_spill_bytes);
   CP_SWAP(int, exec->blocks_per_sm, exec->alt_blocks_per_sm);
   CP_SWAP(int, exec->reg_cap, exec->alt_reg_cap);
   /* The sampler's table and quad-derivative flag are globals of a module, so
    * the resolved addresses and the last values written to them belong to the
    * build that was launched, not to the shader. */
   CP_SWAP(bool, exec->globals_resolved, exec->alt_globals_resolved);
   CP_SWAP(CUdeviceptr, exec->sym_sampler_table, exec->alt_sym_sampler_table);
   CP_SWAP(CUdeviceptr, exec->sym_quad_derivs, exec->alt_sym_quad_derivs);
   CP_SWAP(uint64_t, exec->last_sampler_table, exec->alt_last_sampler_table);
   CP_SWAP(int, exec->last_quad_derivs, exec->alt_last_quad_derivs);
#undef CP_SWAP
}

/*
 * Whether the shader writes globally visible memory — a storage image, an
 * SSBO, a global address — by store or by atomic.
 *
 * Read off the intrinsics rather than out of nir_shader_info, so that it
 * describes the NIR this compile is looking at and cannot be a stale answer
 * left by whatever last ran nir_shader_gather_info().
 */
static bool
cp_nir_writes_memory(struct nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            if (nir_intrinsic_writes_external_memory(nir_instr_as_intrinsic(instr)))
               return true;
         }
      }
   }
   return false;
}

static unsigned
cp_count_substring(const char *text, const char *needle)
{
   unsigned count = 0;
   size_t length = strlen(needle);
   while ((text = strstr(text, needle))) {
      count++;
      text += length;
   }
   return count;
}

static struct cp_shader_binary *
cp_hardware_failure_stub(enum cp_hw_compile_failure reason)
{
   struct cp_shader_binary *bin = CALLOC_STRUCT(cp_shader_binary);
   if (bin)
      bin->hw_compile_failure = reason;
   return bin;
}

static struct cp_shader_binary *
cp_compile_nir_one(struct nir_shader *nir, int sm_major, int sm_minor,
                   const char *sampler_ptx, const char *fs_helper_ptx,
                   bool fused_interp, bool inline_interp,
                   bool hardware_texture)
{
   struct ntl_context ctx = {0};
   ctx.nir = nir;
   ctx.sm_major = sm_major;
   ctx.sm_minor = sm_minor;
   ctx.fused_interp = fused_interp;
   ctx.inline_interp = inline_interp;
   ctx.hardware_texture = hardware_texture;

   cp_lower_nir(nir);
   if (hardware_texture && !cp_hardware_texture_shader_eligible(nir))
      return cp_hardware_failure_stub(CP_HW_COMPILE_INELIGIBLE);

   ctx.inline_pntc_input = -1;
   ctx.inline_pos_input = -1;
   if (inline_interp) {
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_in) {
         if (var->data.location == VARYING_SLOT_PNTC)
            ctx.inline_pntc_input = (int32_t)var->data.driver_location;
         if (var->data.location == VARYING_SLOT_POS)
            ctx.inline_pos_input = (int32_t)var->data.driver_location;
      }
      bool input_footprint_safe = nir->num_inputs <= CP_MAX_FS_INPUTS;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_load_input)
                  continue;

               /* load_input addresses vec4 slots as base + src[0], then adds
                * the component byte offset. Arrays and matrices therefore do
                * not necessarily read `base` itself. Keep the interpolation
                * writes in exact agreement with the emitted address. */
               if (!nir_src_is_const(intr->src[0])) {
                  /* A valid indirect can select any declared input. */
                  ctx.inline_live_slots = (1u << CP_MAX_FS_INPUTS) - 1u;
                  continue;
               }

               uint64_t offset = nir_src_as_uint(intr->src[0]);
               uint64_t first_byte =
                  ((uint64_t)nir_intrinsic_base(intr) + offset) * 16u +
                  (uint64_t)nir_intrinsic_component(intr) * 4u;
               uint64_t bytes =
                  ((uint64_t)intr->def.num_components * intr->def.bit_size +
                   7u) / 8u;
               uint64_t first_slot = first_byte / 16u;
               uint64_t last_slot = (first_byte + bytes - 1u) / 16u;
               if (!bytes || last_slot >= CP_MAX_FS_INPUTS ||
                   last_slot >= nir->num_inputs) {
                  input_footprint_safe = false;
                  continue;
               }
               for (uint64_t slot = first_slot; slot <= last_slot; slot++)
                  ctx.inline_live_slots |= 1u << slot;
            }
         }
      }
      if (!input_footprint_safe) {
         if (cp_debug->shader_stats || cp_debug->dump_ir || cp_debug->dump_ptx)
            fprintf(stderr, "cudapipe: inline FS declined: input footprint "
                    "exceeds the %u-slot local interpolation array\n",
                    CP_MAX_FS_INPUTS);
         return hardware_texture
            ? cp_hardware_failure_stub(CP_HW_COMPILE_INLINE_FOOTPRINT)
            : NULL;
      }
   }

   /* After the lowering, which is the NIR the prologue below is generated
    * for and the NIR the flag on the binary has to describe. */
   ctx.writes_memory = cp_nir_writes_memory(nir);
   struct cp_shader_binary tex_meta = {0};
   capture_tex_desc_refs(nir, &tex_meta);
   if (hardware_texture) {
      capture_hw_tex_sites(nir, &tex_meta);
      ctx.hardware_texture_site_count = tex_meta.num_hw_tex_sites;
   }

   /*
    * nir_convert_from_ssa() turns every remaining phi into a NIR register, and
    * emit_intrinsic gives a register an alloca, which the NVPTX backend puts
    * in a __local_depot -- off-chip memory, and in instancing's fragment
    * shader inside a loop. That is why the register could not be found
    * anywhere earlier: it does not exist until this line. The Gallium-hosted
    * driver emits no depot because lavapipe had already flattened those
    * branches into bcsel.
    *
    * Flattening them here with nir_opt_peephole_select does remove it
    * completely -- no alloca, no local operation, matching that driver
    * exactly -- and it is **not worth doing**: measured against the same
    * driver in the same mode, instancing goes 1.52x to 1.59x, texture 0.98 to
    * 1.16, vulkanscene 1.37 to 1.33, particlesystem 1.13 to 1.10. It also
    * renders bloom wrong at every limit from 8 up. The depot is real and it is
    * not what makes these shaders slow.
    */
   /*
    * nir_convert_from_ssa() below turns every remaining phi into a NIR
    * register, and emit_intrinsic gives a register an alloca, which the NVPTX
    * backend puts in a __local_depot -- off-chip memory, and in instancing's
    * fragment shader inside a loop. That is why the register could not be
    * found anywhere earlier in the pipeline: it does not exist until this
    * line.
    *
    * The Gallium-hosted driver runs this same code and emits no depot,
    * because lavapipe had already flattened those branches into bcsel before
    * cudapipe saw the shader. This is that pass.
    */

   /* Convert from SSA to reg form to eliminate phi nodes */
   nir_convert_from_ssa(nir, true, false);
   nir_opt_dce(nir);

   ctx.llvm_ctx = LLVMContextCreate();
   ctx.module = LLVMModuleCreateWithNameInContext("cudapipe_compute", ctx.llvm_ctx);
   ctx.builder = LLVMCreateBuilderInContext(ctx.llvm_ctx);

   /* Create the kernel function: void @main(ptr %args) */
   LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx.llvm_ctx), 0);
   LLVMTypeRef fn_type = LLVMFunctionType(LLVMVoidTypeInContext(ctx.llvm_ctx),
                                          &ptr_type, 1, false);
   ctx.function = LLVMAddFunction(ctx.module, "main", fn_type);

   /* Mark as a kernel entry point */
   LLVMSetFunctionCallConv(ctx.function, 71); /* PTX_Kernel */

   /* Add nvvm annotation for kernel */
   LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx.llvm_ctx);
   LLVMValueRef md_vals[] = {
      ctx.function,
      LLVMMDStringInContext(ctx.llvm_ctx, "kernel", 6),
      LLVMConstInt(i32, 1, false),
   };
   LLVMValueRef md_node = LLVMMDNodeInContext(ctx.llvm_ctx, md_vals, 3);
   LLVMAddNamedMetadataOperand(ctx.module, "nvvm.annotations", md_node);

   /*
    * The other shape the performance plan named — emit `__launch_bounds__`
    * and let ptxas work the register count out from the block size — behind
    * an environment variable because measuring it is the only way to see that
    * it does not work here. `CUDAPIPE_LAUNCH_BOUNDS=N` emits `.maxntid 256`
    * and `.minnctapersm N`. Measured on this driver:
    *
    * - `.maxntid 256` alone is not binding. A 256-thread block only requires
    *   256 registers a thread, the shaders are under that, and nothing
    *   changes. The occupancy-forcing half is `.minnctapersm`, which is the
    *   same guess as a register cap in different units.
    * - With `.minnctapersm 2` **every shader that links the sampler fails to
    *   load**: cuModuleLoadData returns 300, `device kernel image is invalid`.
    *   Those are exactly the shaders the item is about. `.minnctapersm 1`
    *   loads and does nothing.
    * - On the shaders that do not link it, the annotation *raises* the
    *   register count rather than lowering it — 56 to 72 on texture's vertex
    *   shader, 66 to 95 on bloom's — because it is a permission to spend up
    *   to the bound and ptxas takes it.
    *
    * So it is left off, and the driver caps registers directly instead.
    */
   if (cp_debug->launch_bounds > 0) {
      LLVMValueRef ntid[] = {
         ctx.function,
         LLVMMDStringInContext(ctx.llvm_ctx, "maxntidx", 8),
         LLVMConstInt(i32, CP_SHADER_BLOCK_THREADS, false),
      };
      LLVMAddNamedMetadataOperand(ctx.module, "nvvm.annotations",
                                  LLVMMDNodeInContext(ctx.llvm_ctx, ntid, 3));
      LLVMValueRef ctasm[] = {
         ctx.function,
         LLVMMDStringInContext(ctx.llvm_ctx, "minctasm", 8),
         LLVMConstInt(i32, (int)cp_debug->launch_bounds, false),
      };
      LLVMAddNamedMetadataOperand(ctx.module, "nvvm.annotations",
                                  LLVMMDNodeInContext(ctx.llvm_ctx, ctasm, 3));
   }

   LLVMValueRef arg0 = LLVMGetParam(ctx.function, 0);
   ctx.kernel_args = &arg0;
   ctx.num_kernel_args = 1;

   if (!emit_function(&ctx)) {
      LLVMDisposeBuilder(ctx.builder);
      LLVMDisposeModule(ctx.module);
      LLVMContextDispose(ctx.llvm_ctx);
      return NULL;
   }

#ifdef CP_HAVE_FS_INLINE_BC
   if (inline_interp &&
       !link_and_inline_fs_interp(ctx.module, ctx.llvm_ctx, sm_major, sm_minor)) {
      LLVMDisposeBuilder(ctx.builder);
      LLVMDisposeModule(ctx.module);
      LLVMContextDispose(ctx.llvm_ctx);
      return NULL;
   }
#else
   assert(!inline_interp);
#endif

   /* For inline execution this is the optimized, post-link IR: the dump is
    * itself the proof surface for helper removal and local-input promotion. */
   if (cp_debug->dump_ir)
      LLVMDumpModule(ctx.module);

   /* Verify — if invalid IR, bail out instead of crashing in PTX emission */
   char *error = NULL;
   if (LLVMVerifyModule(ctx.module, LLVMReturnStatusAction, &error)) {
      fprintf(stderr, "cudapipe: LLVM module verification failed:\n%s\n", error ? error : "(null)");
      LLVMDisposeMessage(error);
      LLVMDisposeBuilder(ctx.builder);
      LLVMDisposeModule(ctx.module);
      LLVMContextDispose(ctx.llvm_ctx);
      return NULL;
   }
   LLVMDisposeMessage(error);

   /* Compile to PTX */
   size_t ptx_size = 0;
   char *ptx = compile_module_to_ptx(ctx.module, sm_major, sm_minor, &ptx_size);

   LLVMDisposeBuilder(ctx.builder);
   LLVMDisposeModule(ctx.module);
   LLVMContextDispose(ctx.llvm_ctx);

   if (!ptx)
      return NULL;

   /* The inline route is admitted only when its caller-owned input array was
    * completely promoted. A surviving helper name means always-inline failed;
    * PTX local memory means the array (or another address-taken temporary)
    * survived SROA. Either condition would turn the intended register dataflow
    * back into a memory round trip, so discard this mode and let the isolated
    * fused/classic binaries handle the shader. */
   bool bad_hardware_ptx = false;
   if (hardware_texture) {
      unsigned grad_count = cp_count_substring(ptx, "tex.grad.");
      unsigned level_count = cp_count_substring(ptx, "tex.level.");
      unsigned tex_count = grad_count + level_count;
      unsigned expected_level = 0;
      for (unsigned i = 0; i < tex_meta.num_hw_tex_sites; i++)
         expected_level += tex_meta.hw_tex_sites[i].op == nir_texop_txl;
      unsigned dim_count = 0;
      const char *dims[] = { "tex.grad.2d", "tex.level.2d",
                             "tex.grad.3d", "tex.level.3d",
                             "tex.grad.cube", "tex.level.cube" };
      for (unsigned i = 0; i < ARRAY_SIZE(dims); i++)
         dim_count += cp_count_substring(ptx, dims[i]);
      bad_hardware_ptx = strstr(ptx, "cp_tex_sample") ||
                         strstr(ptx, "cp_sampler_table") ||
                         tex_count != tex_meta.num_hw_tex_sites ||
                         level_count != expected_level ||
                         grad_count != tex_meta.num_hw_tex_sites - expected_level ||
                         dim_count != tex_meta.num_hw_tex_sites;
   }
   bool surviving_inline_helper = strstr(ptx, "cp_fs_inline_lane");
   bool local_memory = strstr(ptx, ".local") || strstr(ptx, "ld.local") ||
                       strstr(ptx, "st.local");
   if (bad_hardware_ptx ||
       (inline_interp && (surviving_inline_helper || local_memory))) {
      if (cp_debug->shader_stats || cp_debug->dump_ir || cp_debug->dump_ptx)
         fprintf(stderr, "cudapipe: rejected unsafe inline FS PTX "
                 "(helper=%u local=%u texture=%u)\n",
                 surviving_inline_helper, local_memory, bad_hardware_ptx);
      free(ptx);
      if (!hardware_texture)
         return NULL;
      enum cp_hw_compile_failure reason = surviving_inline_helper
         ? CP_HW_COMPILE_SURVIVING_HELPER
         : (bad_hardware_ptx ? CP_HW_COMPILE_BAD_TEXTURE_PTX
                             : CP_HW_COMPILE_LOCAL_MEMORY);
      struct cp_shader_binary *stub = cp_hardware_failure_stub(reason);
      if (stub) {
         stub->num_hw_tex_sites = tex_meta.num_hw_tex_sites;
         memcpy(stub->hw_tex_sites, tex_meta.hw_tex_sites,
                sizeof(stub->hw_tex_sites));
         stub->hw_tex_dynamic = tex_meta.hw_tex_dynamic;
      }
      return stub;
   }

   if (cp_debug->dump_ptx)
      fprintf(stderr, "cudapipe: generated PTX (%zu bytes):\n%s\n", ptx_size, ptx);
   if (cp_debug->dump_nir)
      nir_print_shader(nir, stderr);

   struct cp_shader_binary *bin = CALLOC_STRUCT(cp_shader_binary);
   if (!bin) {
      free(ptx);
      return NULL;
   }
   enum cp_shader_exec_mode mode = hardware_texture
      ? (inline_interp ? CP_SHADER_EXEC_HW_INLINE : CP_SHADER_EXEC_HW_FUSED)
      : (inline_interp ? CP_SHADER_EXEC_INLINE
                       : (fused_interp ? CP_SHADER_EXEC_FUSED
                                       : CP_SHADER_EXEC_CLASSIC));
   struct cp_shader_exec *exec = &bin->exec[mode];
   exec->ptx_text = ptx;
   exec->ptx_size = ptx_size;
   bin->sm_major = sm_major;
   bin->sm_minor = sm_minor;
   bin->shared_size = nir->info.shared_size;
   bin->nir_num_outputs = nir->num_outputs;
   bin->nir_num_inputs = nir->num_inputs;
   bin->is_fragment = nir->info.stage == MESA_SHADER_FRAGMENT;
   bin->reads_const_bufs = ctx.reads_const_bufs;
   bin->writes_memory = ctx.writes_memory;

   bin->num_tex_descs = tex_meta.num_tex_descs;
   memcpy(bin->tex_descs, tex_meta.tex_descs, sizeof(bin->tex_descs));
   bin->tex_descs_dynamic = tex_meta.tex_descs_dynamic;
   bin->num_hw_tex_sites = tex_meta.num_hw_tex_sites;
   memcpy(bin->hw_tex_sites, tex_meta.hw_tex_sites, sizeof(bin->hw_tex_sites));
   bin->hw_tex_dynamic = tex_meta.hw_tex_dynamic;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_vertex_id:
            case nir_intrinsic_load_vertex_id_zero_base:
               bin->reads_vertex_id = true;
               break;
            case nir_intrinsic_load_instance_id:
               bin->reads_instance_id = true;
               break;
            case nir_intrinsic_terminate:
            case nir_intrinsic_terminate_if:
               bin->uses_discard = true;
               break;
            case nir_intrinsic_load_front_face:
               bin->reads_front_face = true;
               break;
            case nir_intrinsic_load_base_instance:
            case nir_intrinsic_load_first_vertex:
            case nir_intrinsic_load_base_vertex:
            case nir_intrinsic_load_draw_id:
               bin->reads_draw_params = true;
               break;
            default:
               break;
            }
         }
      }
   }
   capture_io_locations(nir, bin);

   /* Shaders that sample textures, or that call one of the module's device
    * helpers, need it linked in; the rest load their PTX directly. */
   bool needs_sampler =
      ((!hardware_texture && ctx.uses_tex) || ctx.needs_link) && sampler_ptx;
   const char *stage_name = mesa_shader_stage_name(nir->info.stage);
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      stage_name = hardware_texture
         ? (inline_interp ? "fragment hardware inline" : "fragment hardware fused")
         : (inline_interp ? "fragment inline" :
          (fused_interp ? "fragment fused" : "fragment classic"));
   CUresult err = load_shader_module_tuned(exec, ptx,
                                           needs_sampler ? sampler_ptx : NULL,
                                           fs_helper_ptx,
                                           nir->info.stage == MESA_SHADER_FRAGMENT,
                                           stage_name);

   if (err != CUDA_SUCCESS) {
      if (hardware_texture)
         bin->hw_compile_failure = CP_HW_COMPILE_JIT;
      const char *err_str = NULL;
      cuGetErrorString(err, &err_str);
      fprintf(stderr, "cudapipe: loading shader module failed (%d: %s)\n",
              err, err_str ? err_str : "?");
      /* Keep the PTX for debugging even if load fails */
   }

   return bin;
}

static struct cp_shader_binary *
cp_compile_nir_software(struct nir_shader *nir, int sm_major, int sm_minor,
                        const char *sampler_ptx, const char *fs_helper_ptx,
                        bool no_inline_fs, bool inline_fs,
                        bool force_fused_fs)
{
   if (!nir)
      return NULL;
   if (nir->info.stage != MESA_SHADER_FRAGMENT || !fs_helper_ptx ||
       no_inline_fs)
      return cp_compile_nir_one(nir, sm_major, sm_minor, sampler_ptx, NULL,
                                false, false, false);
   if (force_fused_fs || !inline_fs) {
      /* cp_compile_nir_one lowers its NIR in place. Preserve a pristine clone
       * before the fused attempt so a failed/default JIT never runs classic
       * lowering a second time over the already-lowered shader. */
      nir_shader *classic_nir = force_fused_fs
         ? NULL : nir_shader_clone(NULL, nir);
      struct cp_shader_binary *fused =
         cp_compile_nir_one(nir, sm_major, sm_minor, sampler_ptx,
                            fs_helper_ptx, true, false, false);
      if (fused && fused->exec[CP_SHADER_EXEC_FUSED].kernel) {
         ralloc_free(classic_nir);
         return fused;
      }
      cp_shader_binary_destroy(fused);
      /* FORCE is an exact execution control. The normal default still keeps
       * correctness if its fused JIT is unavailable. */
      struct cp_shader_binary *classic = classic_nir
         ? cp_compile_nir_one(classic_nir, sm_major, sm_minor, sampler_ptx,
                              NULL, false, false, false)
         : NULL;
      ralloc_free(classic_nir);
      return classic;
   }

   /* Opt-in ownership keeps the two binaries needed for correctness: the
    * inline module, and the proven fused module which can also run after the
    * standalone interpolator when scratch allocation refuses the in-shader
    * path. The exact classic binary is built only by NO_INLINE_FS, above. */
   nir_shader *standalone_nir = nir_shader_clone(NULL, nir);
#ifdef CP_HAVE_FS_INLINE_BC
   nir_shader *inline_nir = nir_shader_clone(NULL, nir);
#else
   nir_shader *inline_nir = NULL;
   call_once(&cp_inline_missing_once, cp_warn_inline_missing);
#endif
   struct cp_shader_binary *fused =
      cp_compile_nir_one(nir, sm_major, sm_minor, sampler_ptx,
                         fs_helper_ptx, true, false, false);
   if (fused && !fused->exec[CP_SHADER_EXEC_FUSED].kernel) {
      cp_shader_binary_destroy(fused);
      fused = NULL;
   }
   struct cp_shader_binary *classic = !fused && standalone_nir
      ? cp_compile_nir_one(standalone_nir, sm_major, sm_minor, sampler_ptx,
                           NULL, false, false, false)
      : NULL;
   struct cp_shader_binary *inlined = inline_nir
      ? cp_compile_nir_one(inline_nir, sm_major, sm_minor, sampler_ptx,
                           NULL, false, true, false)
      : NULL;
   ralloc_free(standalone_nir);
   ralloc_free(inline_nir);

   struct cp_shader_binary *bin = fused ? fused : classic;
   if (!bin) {
      cp_shader_binary_destroy(inlined);
      return NULL;
   }
   if (inlined && inlined != bin) {
      bin->exec[CP_SHADER_EXEC_INLINE] = inlined->exec[CP_SHADER_EXEC_INLINE];
      memset(&inlined->exec[CP_SHADER_EXEC_INLINE], 0,
             sizeof(inlined->exec[CP_SHADER_EXEC_INLINE]));
      cp_shader_binary_destroy(inlined);
   }
   return bin;
}


struct cp_shader_binary *
cp_compile_nir_to_ptx(struct nir_shader *nir, int sm_major, int sm_minor,
                      const char *sampler_ptx, const char *math_ptx,
                      const char *fs_helper_ptx,
                      bool no_inline_fs, bool inline_fs, bool force_fused_fs,
                      bool hw_texture)
{
   if (!nir)
      return NULL;

   /* Clone before any software attempt lowers in place. Hardware ownership is
    * independent of generic inline mode and always starts from pristine FS
    * NIR. A build without matching inline bitcode quietly skips this clone. */
   nir_shader *hardware_nir = NULL;
   nir_shader *hardware_fused_nir = NULL;
#ifdef CP_HAVE_FS_INLINE_BC
   if (hw_texture && nir->info.stage == MESA_SHADER_FRAGMENT) {
      hardware_nir = nir_shader_clone(NULL, nir);
      hardware_fused_nir = nir_shader_clone(NULL, nir);
   }
#else
   (void)hw_texture;
#endif

   struct cp_shader_binary *bin = cp_compile_nir_software(
      nir, sm_major, sm_minor, sampler_ptx, fs_helper_ptx,
      no_inline_fs, inline_fs, force_fused_fs);

#ifdef CP_HAVE_FS_INLINE_BC
   struct cp_shader_binary *hardware = hardware_nir
      ? cp_compile_nir_one(hardware_nir, sm_major, sm_minor, math_ptx, NULL,
                           false, true, true)
      : NULL;
   struct cp_shader_binary *hardware_fused =
      hardware_fused_nir &&
      (!hardware || !hardware->exec[CP_SHADER_EXEC_HW_INLINE].kernel)
      ? cp_compile_nir_one(hardware_fused_nir, sm_major, sm_minor,
                           math_ptx, fs_helper_ptx, true, false, true)
      : NULL;
   if (bin && hardware) {
      bin->hw_compile_failure = hardware->hw_compile_failure;
      bin->num_hw_tex_sites = hardware->num_hw_tex_sites;
      memcpy(bin->hw_tex_sites, hardware->hw_tex_sites,
             sizeof(bin->hw_tex_sites));
      bin->hw_tex_dynamic = hardware->hw_tex_dynamic;
      if (hardware->exec[CP_SHADER_EXEC_HW_INLINE].kernel) {
         bin->exec[CP_SHADER_EXEC_HW_INLINE] =
            hardware->exec[CP_SHADER_EXEC_HW_INLINE];
         memset(&hardware->exec[CP_SHADER_EXEC_HW_INLINE], 0,
                sizeof(hardware->exec[CP_SHADER_EXEC_HW_INLINE]));
      }
   }
   if (bin && hardware_fused) {
      bool metadata_match = !bin->num_hw_tex_sites ||
         (bin->num_hw_tex_sites == hardware_fused->num_hw_tex_sites &&
          bin->hw_tex_dynamic == hardware_fused->hw_tex_dynamic &&
          !memcmp(bin->hw_tex_sites, hardware_fused->hw_tex_sites,
                  sizeof(bin->hw_tex_sites)));
      if (!bin->num_hw_tex_sites) {
         bin->num_hw_tex_sites = hardware_fused->num_hw_tex_sites;
         memcpy(bin->hw_tex_sites, hardware_fused->hw_tex_sites,
                sizeof(bin->hw_tex_sites));
         bin->hw_tex_dynamic = hardware_fused->hw_tex_dynamic;
      }
      if (hardware_fused->exec[CP_SHADER_EXEC_HW_FUSED].kernel &&
          metadata_match) {
         bin->hw_compile_failure = CP_HW_COMPILE_NONE;
         bin->exec[CP_SHADER_EXEC_HW_FUSED] =
            hardware_fused->exec[CP_SHADER_EXEC_HW_FUSED];
         memset(&hardware_fused->exec[CP_SHADER_EXEC_HW_FUSED], 0,
                sizeof(hardware_fused->exec[CP_SHADER_EXEC_HW_FUSED]));
      } else {
         bin->hw_compile_failure = metadata_match
            ? hardware_fused->hw_compile_failure : CP_HW_COMPILE_OTHER;
      }
   }
   cp_shader_binary_destroy(hardware);
   cp_shader_binary_destroy(hardware_fused);
#endif
   ralloc_free(hardware_nir);
   ralloc_free(hardware_fused_nir);
   return bin;
}


static void
cp_shader_exec_destroy(struct cp_shader_exec *exec, bool free_ptx)
{
   if (exec->module)
      cuModuleUnload(exec->module);
   if (exec->alt_module)
      cuModuleUnload(exec->alt_module);
   if (exec->tune.events_made) {
      for (unsigned e = 0; e < CP_TUNE_SAMPLES; e++) {
         cuEventDestroy(exec->tune.start[e]);
         cuEventDestroy(exec->tune.stop[e]);
      }
   }
   if (free_ptx)
      free(exec->ptx_text);
   memset(exec, 0, sizeof(*exec));
}

void
cp_shader_binary_destroy(struct cp_shader_binary *bin)
{
   if (!bin)
      return;
   for (unsigned mode = 0; mode < CP_SHADER_EXEC_COUNT; mode++)
      cp_shader_exec_destroy(&bin->exec[mode], true);
   for (unsigned i = 0; i < bin->num_sampler_variants; i++) {
      struct cp_sampler_variant *v = &bin->sampler_variants[i];
      for (unsigned mode = 0; mode < CP_SHADER_EXEC_COUNT; mode++)
         cp_shader_exec_destroy(&v->exec[mode], false);
      free(v->states);
   }
   FREE(bin);
}
