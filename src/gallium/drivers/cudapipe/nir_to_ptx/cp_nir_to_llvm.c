#include "cp_nir_to_llvm.h"

#include "compiler/nir/nir.h"
#include "util/u_memory.h"

#include <llvm-c/Core.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/Transforms/PassBuilder.h>

#include <stdio.h>
#include <string.h>

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

   LLVMBasicBlockRef break_block;
   LLVMBasicBlockRef continue_block;

   struct nir_shader *nir;
   int sm_major;
   int sm_minor;
};

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
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      LLVMValueRef args = ctx->kernel_args[0];
      LLVMValueRef args_as_ptrptr = LLVMBuildBitCast(ctx->builder, args, ptr_ptr_type, "");
      /* Load arg_ptrs[0] which is the grid_size pointer */
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMValueRef zero64 = LLVMConstInt(i64, 0, false);
      LLVMValueRef grid_ptr = LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, args_as_ptrptr, &zero64, 1, ""), "grid_ptr");
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
      /* For VS kernel: vertex_id = blockIdx.x * blockDim.x + threadIdx.x */
      LLVMValueRef bid = emit_workgroup_id(ctx, 0);
      LLVMValueRef tid = emit_local_invocation_id(ctx, 0);
      LLVMValueRef bs = LLVMConstInt(i32, 256, false); /* block size used for VS launch */
      LLVMValueRef vid = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, bid, bs, ""), tid, "vertex_id");
      set_ssa_def(ctx, &instr->def, vid);
      break;
   }
   case nir_intrinsic_load_input: {
      /* VS input: read from input buffer at (base + component) * 4 bytes per vertex.
       * Kernel arg layout: [0]=input_buffers_ptr, args as before.
       * For now, treat input as reading from the vertex's attribute slot. */
      unsigned base = nir_intrinsic_base(instr);
      unsigned comp = nir_intrinsic_component(instr);
      LLVMValueRef offset_val = get_src(ctx, &instr->src[0]);

      /* Input ptr is at args[2] (after grid_info at 0, reserved at 1) */
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      LLVMValueRef args = ctx->kernel_args[0];
      LLVMValueRef args_pp = LLVMBuildBitCast(ctx->builder, args, ptr_ptr_type, "");
      LLVMValueRef input_ptr = LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, args_pp,
            &(LLVMValueRef){LLVMConstInt(i64, 2, false)}, 1, ""), "vs_input");

      /* Compute byte offset: vertex_id * vertex_stride + base * 16 + comp * 4 + offset * 16 */
      LLVMValueRef bid2 = emit_workgroup_id(ctx, 0);
      LLVMValueRef tid2 = emit_local_invocation_id(ctx, 0);
      LLVMValueRef vid = LLVMBuildAdd(ctx->builder,
         LLVMBuildMul(ctx->builder, bid2, LLVMConstInt(i32, 256, false), ""), tid2, "");

      /* Read stride from args[3] */
      LLVMTypeRef i32_ptr = LLVMPointerType(i32, 0);
      LLVMValueRef stride_ptr = LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, args_pp,
            &(LLVMValueRef){LLVMConstInt(i64, 3, false)}, 1, ""), "stride_ptr");
      LLVMValueRef stride = LLVMBuildLoad2(ctx->builder, i32,
         LLVMBuildBitCast(ctx->builder, stride_ptr, i32_ptr, ""), "stride");

      LLVMValueRef byte_off = LLVMBuildAdd(ctx->builder,
         LLVMBuildAdd(ctx->builder,
            LLVMBuildMul(ctx->builder, vid, stride, ""),
            LLVMConstInt(i32, base * 16 + comp * 4, false), ""),
         LLVMBuildMul(ctx->builder, offset_val, LLVMConstInt(i32, 16, false), ""), "");

      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), input_ptr, &byte_off, 1, "");

      unsigned nc = instr->def.num_components;
      unsigned bs2 = instr->def.bit_size;
      LLVMTypeRef load_type = get_llvm_type(ctx, bs2, nc);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(load_type, 0), "");
      LLVMValueRef val = LLVMBuildLoad2(ctx->builder, load_type, typed_ptr, "vs_in");
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
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      LLVMValueRef args = ctx->kernel_args[0];
      LLVMValueRef args_pp = LLVMBuildBitCast(ctx->builder, args, ptr_ptr_type, "");
      LLVMValueRef output_ptr = LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, args_pp,
            &(LLVMValueRef){LLVMConstInt(i64, 4, false)}, 1, ""), "vs_output");

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
   case nir_intrinsic_load_const_buf_base_addr_lvp: {
      /* Returns the 64-bit base address of constant buffer `src[0]`
       * In our layout: kernel arg is a ptr to array of ptrs.
       * UBO pointers start at index 18 in the arg array.
       * The slot index from src[0] maps to arg_ptrs[18 + slot].
       */
      LLVMValueRef slot = get_src(ctx, &instr->src[0]);
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      LLVMValueRef args = ctx->kernel_args[0];
      LLVMValueRef args_as_ptrptr = LLVMBuildBitCast(ctx->builder, args, ptr_ptr_type, "");
      LLVMValueRef ubo_idx = LLVMBuildAdd(ctx->builder, slot, LLVMConstInt(i32, 18, false), "");
      ubo_idx = LLVMBuildZExt(ctx->builder, ubo_idx, i64, "");
      LLVMValueRef ubo_ptr = LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, args_as_ptrptr,
            &ubo_idx, 1, ""), "ubo_base");
      /* Convert pointer to i64 (device address) */
      LLVMValueRef addr = LLVMBuildPtrToInt(ctx->builder, ubo_ptr, i64, "");
      set_ssa_def(ctx, &instr->def, addr);
      break;
   }
   case nir_intrinsic_get_ssbo_size: {
      /*
       * Descriptor struct: { ptr base (8 bytes); u32 num_elements (4 bytes); }
       * num_elements is at offset 8 from the descriptor address.
       * Returns size in BYTES (num_elements is already in the unit the shader expects).
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
       * After lavapipe lowering, src[0] is a 64-bit address pointing to a
       * descriptor struct: { ptr base; u32 num_elements; }
       * We read the base pointer from the descriptor, then access base[offset].
       */
      LLVMValueRef desc_addr = get_src(ctx, &instr->src[0]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[1]);
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      /* Read the base pointer from the descriptor (first 8 bytes) */
      LLVMValueRef desc_ptr = LLVMBuildIntToPtr(ctx->builder, desc_addr, ptr_ptr_type, "");
      LLVMValueRef buf_ptr = LLVMBuildLoad2(ctx->builder, ptr_type, desc_ptr, "ssbo_base");
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
      LLVMValueRef desc_addr = get_src(ctx, &instr->src[1]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[2]);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      /* Read the base pointer from the descriptor */
      LLVMValueRef desc_ptr = LLVMBuildIntToPtr(ctx->builder, desc_addr, ptr_ptr_type, "");
      LLVMValueRef buf_ptr = LLVMBuildLoad2(ctx->builder, ptr_type, desc_ptr, "ssbo_base");
      /* Access buf_ptr + byte_offset */
      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), buf_ptr, &byte_offset, 1, "");
      LLVMTypeRef store_type = LLVMTypeOf(data);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(store_type, 0), "");
      LLVMBuildStore(ctx->builder, data, typed_ptr);
      break;
   }
   case nir_intrinsic_load_ubo: {
      /*
       * After lavapipe lowering, src[0] is a 64-bit descriptor address
       * pointing to lp_jit_buffer {ptr base, u32 num_elements}.
       * Same pattern as load_ssbo.
       */
      LLVMValueRef desc_addr = get_src(ctx, &instr->src[0]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[1]);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      LLVMValueRef desc_ptr = LLVMBuildIntToPtr(ctx->builder, desc_addr, ptr_ptr_type, "");
      LLVMValueRef buf_ptr = LLVMBuildLoad2(ctx->builder, ptr_type, desc_ptr, "ubo_base");
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

      /* byte_offset = base_offset + z * img_stride + y * row_stride + x * pixel_size */
      unsigned bit_size = instr->def.bit_size;
      unsigned pixel_size = bit_size / 8;
      LLVMValueRef offset_val = LLVMBuildAdd(ctx->builder, base_offset,
         LLVMBuildAdd(ctx->builder,
            LLVMBuildAdd(ctx->builder,
               LLVMBuildMul(ctx->builder, z, img_stride, ""),
               LLVMBuildMul(ctx->builder, y, row_stride, ""), ""),
            LLVMBuildMul(ctx->builder, x, LLVMConstInt(i32, pixel_size, false), ""), ""), "");

      /* Load pixel */
      LLVMValueRef pixel_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), base_ptr, &offset_val, 1, "");
      LLVMTypeRef pixel_type = get_llvm_type(ctx, bit_size, 1);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, pixel_ptr,
         LLVMPointerType(pixel_type, 0), "");
      LLVMValueRef pixel_val = LLVMBuildLoad2(ctx->builder, pixel_type, typed_ptr, "img_load");

      /* Return as vec4 (only .x is meaningful for r32 formats) */
      unsigned num_comp = instr->def.num_components;
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

      /* Extract first component of data for single-component formats */
      LLVMValueRef store_val = data;
      if (LLVMGetTypeKind(LLVMTypeOf(data)) == LLVMVectorTypeKind)
         store_val = LLVMBuildExtractElement(ctx->builder, data,
            LLVMConstInt(i32, 0, false), "");

      unsigned pixel_size = LLVMGetIntTypeWidth(LLVMTypeOf(store_val)) / 8;
      if (pixel_size == 0) pixel_size = 4;

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
   default:
      /* Unhandled intrinsic — emit undef for the result */
      if (nir_intrinsic_infos[instr->intrinsic].has_dest) {
         unsigned num_comp = instr->def.num_components;
         unsigned bit_size = instr->def.bit_size;
         set_ssa_def(ctx, &instr->def,
                     LLVMGetUndef(get_llvm_type(ctx, bit_size, num_comp)));
      }
      break;
   }
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
      /* Extract component from vector sources when output is scalar */
      if (LLVMGetTypeKind(LLVMTypeOf(src[i])) == LLVMVectorTypeKind && num_comp == 1) {
         src[i] = LLVMBuildExtractElement(ctx->builder, src[i],
            LLVMConstInt(i32, instr->src[i].swizzle[0], false), "");
      }
   }

   LLVMValueRef result = NULL;

   /* For float ops, ensure sources are float-typed (bitcast from int if needed) */
   bool is_float_op = (instr->op >= nir_op_fadd && instr->op <= nir_op_fneu) ||
                      instr->op == nir_op_fneg || instr->op == nir_op_fabs ||
                      instr->op == nir_op_flt || instr->op == nir_op_fge ||
                      instr->op == nir_op_feq || instr->op == nir_op_fneu ||
                      instr->op == nir_op_fadd || instr->op == nir_op_fsub ||
                      instr->op == nir_op_fmul || instr->op == nir_op_fdiv;
   if (is_float_op) {
      for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
         if (src[i] && LLVMGetTypeKind(LLVMTypeOf(src[i])) == LLVMIntegerTypeKind) {
            unsigned w = LLVMGetIntTypeWidth(LLVMTypeOf(src[i]));
            LLVMTypeRef ft = (w == 64) ? LLVMDoubleTypeInContext(ctx->llvm_ctx) :
                             (w == 16) ? LLVMHalfTypeInContext(ctx->llvm_ctx) :
                                         LLVMFloatTypeInContext(ctx->llvm_ctx);
            src[i] = LLVMBuildBitCast(ctx->builder, src[i], ft, "");
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
      result = LLVMBuildFPToSI(ctx->builder, src[0], get_llvm_type(ctx, 32, 1), "");
      break;
   case nir_op_f2u32:
      result = LLVMBuildFPToUI(ctx->builder, src[0], get_llvm_type(ctx, 32, 1), "");
      break;
   case nir_op_i2i64:
      result = LLVMBuildSExt(ctx->builder, src[0], get_llvm_type(ctx, 64, 1), "");
      break;
   case nir_op_u2u64:
      result = LLVMBuildZExt(ctx->builder, src[0], get_llvm_type(ctx, 64, 1), "");
      break;
   case nir_op_i2i32:
   case nir_op_u2u32:
      result = LLVMBuildTrunc(ctx->builder, src[0], get_llvm_type(ctx, 32, 1), "");
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
      result = LLVMBuildSelect(ctx->builder, cond_val,
         LLVMConstReal(LLVMFloatTypeInContext(ctx->llvm_ctx), 1.0),
         LLVMConstReal(LLVMFloatTypeInContext(ctx->llvm_ctx), 0.0), "");
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
      LLVMTypeRef elem_type = LLVMTypeOf(src[0]);
      LLVMValueRef vec = LLVMGetUndef(LLVMVectorType(elem_type, nc));
      for (unsigned c = 0; c < nc; c++)
         vec = LLVMBuildInsertElement(ctx->builder, vec, src[c],
            LLVMConstInt(i32, c, false), "");
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
      result = LLVMGetUndef(dst_type);
      break;
   }

   if (result)
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

   emit_cf_list(ctx, &impl->body);

   /* Add return if no terminator */
   if (!LLVMGetBasicBlockTerminator(LLVMGetInsertBlock(ctx->builder)))
      LLVMBuildRetVoid(ctx->builder);

   free(ctx->ssa_defs);
   free(ctx->regs);
   return true;
}

static char *
compile_module_to_ptx(LLVMModuleRef module, int sm_major, int sm_minor, size_t *out_size)
{
   char triple[] = "nvptx64-nvidia-cuda";
   char cpu[16];
   snprintf(cpu, sizeof(cpu), "sm_%d%d", sm_major, sm_minor);

   LLVMInitializeNVPTXTargetInfo();
   LLVMInitializeNVPTXTarget();
   LLVMInitializeNVPTXTargetMC();
   LLVMInitializeNVPTXAsmPrinter();

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

struct cp_shader_binary *
cp_compile_nir_to_ptx(struct nir_shader *nir, int sm_major, int sm_minor)
{
   struct ntl_context ctx = {0};
   ctx.nir = nir;
   ctx.sm_major = sm_major;
   ctx.sm_minor = sm_minor;

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

   LLVMValueRef arg0 = LLVMGetParam(ctx.function, 0);
   ctx.kernel_args = &arg0;
   ctx.num_kernel_args = 1;

   if (!emit_function(&ctx)) {
      LLVMDisposeBuilder(ctx.builder);
      LLVMDisposeModule(ctx.module);
      LLVMContextDispose(ctx.llvm_ctx);
      return NULL;
   }

   if (getenv("CUDAPIPE_DUMP_IR"))
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

   if (getenv("CUDAPIPE_DUMP_PTX"))
      fprintf(stderr, "cudapipe: generated PTX (%zu bytes):\n%s\n", ptx_size, ptx);
   if (getenv("CUDAPIPE_DUMP_NIR"))
      nir_print_shader(nir, stderr);

   struct cp_shader_binary *bin = CALLOC_STRUCT(cp_shader_binary);
   bin->ptx_text = ptx;
   bin->ptx_size = ptx_size;
   bin->sm_major = sm_major;
   bin->sm_minor = sm_minor;
   bin->shared_size = nir->info.shared_size;
   bin->nir_num_outputs = nir->num_outputs;

   /* Load PTX into CUDA module */
   CUresult err = cuModuleLoadData(&bin->module, ptx);
   if (err != CUDA_SUCCESS) {
      fprintf(stderr, "cudapipe: cuModuleLoadData failed (%d)\n", err);
      /* Keep the PTX for debugging even if load fails */
   } else {
      cuModuleGetFunction(&bin->kernel, bin->module, "main");
   }

   return bin;
}

void
cp_shader_binary_destroy(struct cp_shader_binary *bin)
{
   if (!bin)
      return;
   if (bin->module)
      cuModuleUnload(bin->module);
   free(bin->ptx_text);
   FREE(bin);
}
