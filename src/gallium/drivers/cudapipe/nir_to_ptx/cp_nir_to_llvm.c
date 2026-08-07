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

   LLVMValueRef *kernel_args;
   unsigned num_kernel_args;

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
      /* Passed as kernel argument 0: pointer to {gridX, gridY, gridZ} */
      LLVMValueRef ptr = ctx->kernel_args[0];
      LLVMTypeRef i32_ptr = LLVMPointerType(i32, 0);
      ptr = LLVMBuildBitCast(ctx->builder, ptr, i32_ptr, "");
      LLVMValueRef x = LLVMBuildLoad2(ctx->builder, i32, ptr, "grid_x");
      LLVMValueRef y = LLVMBuildLoad2(ctx->builder, i32,
         LLVMBuildGEP2(ctx->builder, i32, ptr, (LLVMValueRef[]){LLVMConstInt(i32, 1, false)}, 1, ""), "grid_y");
      LLVMValueRef z = LLVMBuildLoad2(ctx->builder, i32,
         LLVMBuildGEP2(ctx->builder, i32, ptr, (LLVMValueRef[]){LLVMConstInt(i32, 2, false)}, 1, ""), "grid_z");
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
   case nir_intrinsic_load_ssbo: {
      /* src[0] = buffer index, src[1] = byte offset */
      LLVMValueRef buf_idx = get_src(ctx, &instr->src[0]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[1]);
      /* kernel arg is a pointer to an array of buffer pointers */
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      /* args_ptr -> array of {buffer_ptr, buffer_size} structs, starting at offset 16 (after grid_size) */
      LLVMValueRef args = ctx->kernel_args[0];
      LLVMValueRef args_as_ptrptr = LLVMBuildBitCast(ctx->builder, args, ptr_ptr_type, "");
      /* Skip grid info (3 ints = 12 bytes, but aligned to ptr), buffer pointers start at index 2 */
      LLVMValueRef buf_base_idx = LLVMBuildAdd(ctx->builder, buf_idx, LLVMConstInt(i32, 2, false), "");
      buf_base_idx = LLVMBuildZExt(ctx->builder, buf_base_idx, i64, "");
      LLVMValueRef buf_ptr = LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, args_as_ptrptr,
            &buf_base_idx, 1, ""), "buf_ptr");
      /* Add byte offset */
      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), buf_ptr, &byte_offset, 1, "");
      /* Load value */
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
      /* src[0] = data, src[1] = buffer index, src[2] = byte offset */
      LLVMValueRef data = get_src(ctx, &instr->src[0]);
      LLVMValueRef buf_idx = get_src(ctx, &instr->src[1]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[2]);
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      LLVMValueRef args = ctx->kernel_args[0];
      LLVMValueRef args_as_ptrptr = LLVMBuildBitCast(ctx->builder, args, ptr_ptr_type, "");
      LLVMValueRef buf_base_idx = LLVMBuildAdd(ctx->builder, buf_idx, LLVMConstInt(i32, 2, false), "");
      buf_base_idx = LLVMBuildZExt(ctx->builder, buf_base_idx, i64, "");
      LLVMValueRef buf_ptr = LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, args_as_ptrptr,
            &buf_base_idx, 1, ""), "buf_ptr");
      LLVMValueRef elem_ptr = LLVMBuildGEP2(ctx->builder,
         LLVMInt8TypeInContext(ctx->llvm_ctx), buf_ptr, &byte_offset, 1, "");
      LLVMTypeRef store_type = LLVMTypeOf(data);
      LLVMValueRef typed_ptr = LLVMBuildBitCast(ctx->builder, elem_ptr,
         LLVMPointerType(store_type, 0), "");
      LLVMBuildStore(ctx->builder, data, typed_ptr);
      break;
   }
   case nir_intrinsic_load_ubo: {
      /* Same pattern as SSBO but from UBO buffer array (starting at a different offset) */
      LLVMValueRef buf_idx = get_src(ctx, &instr->src[0]);
      LLVMValueRef byte_offset = get_src(ctx, &instr->src[1]);
      LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx->llvm_ctx);
      LLVMTypeRef ptr_type = LLVMPointerType(LLVMInt8TypeInContext(ctx->llvm_ctx), 0);
      LLVMTypeRef ptr_ptr_type = LLVMPointerType(ptr_type, 0);
      LLVMValueRef args = ctx->kernel_args[0];
      LLVMValueRef args_as_ptrptr = LLVMBuildBitCast(ctx->builder, args, ptr_ptr_type, "");
      /* UBO pointers start at index 18 (after grid + 16 SSBO slots) */
      LLVMValueRef ubo_base_idx = LLVMBuildAdd(ctx->builder, buf_idx, LLVMConstInt(i32, 18, false), "");
      ubo_base_idx = LLVMBuildZExt(ctx->builder, ubo_base_idx, i64, "");
      LLVMValueRef buf_ptr = LLVMBuildLoad2(ctx->builder, ptr_type,
         LLVMBuildGEP2(ctx->builder, ptr_type, args_as_ptrptr,
            &ubo_base_idx, 1, ""), "ubo_ptr");
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
      /* Extract component if needed */
      if (instr->src[i].swizzle[0] != 0 || num_comp == 1) {
         if (LLVMGetTypeKind(LLVMTypeOf(src[i])) == LLVMVectorTypeKind) {
            if (num_comp == 1) {
               src[i] = LLVMBuildExtractElement(ctx->builder, src[i],
                  LLVMConstInt(i32, instr->src[i].swizzle[0], false), "");
            }
         }
      }
   }

   LLVMValueRef result = NULL;
   LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx->llvm_ctx);
   (void)f32;

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
   case nir_op_mov:
      result = src[0];
      break;
   case nir_op_ilt:
      result = LLVMBuildICmp(ctx->builder, LLVMIntSLT, src[0], src[1], "");
      result = LLVMBuildZExt(ctx->builder, result, get_llvm_type(ctx, bit_size, 1), "");
      break;
   case nir_op_ige:
      result = LLVMBuildICmp(ctx->builder, LLVMIntSGE, src[0], src[1], "");
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
emit_block(struct ntl_context *ctx, nir_block *block)
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

   LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(ctx->llvm_ctx, ctx->function, "entry");
   LLVMPositionBuilderAtEnd(ctx->builder, entry);

   nir_foreach_block(block, impl) {
      emit_block(ctx, block);
   }

   LLVMBuildRetVoid(ctx->builder);

   free(ctx->ssa_defs);
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
      fprintf(stderr, "cudapipe: PTX emission failed: %s\n", error);
      LLVMDisposeMessage(error);
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

   /* Verify */
   char *error = NULL;
   if (LLVMVerifyModule(ctx.module, LLVMPrintMessageAction, &error)) {
      fprintf(stderr, "cudapipe: LLVM module verification failed:\n%s\n", error);
      LLVMDisposeMessage(error);
   } else {
      LLVMDisposeMessage(error);
   }

   /* Compile to PTX */
   size_t ptx_size = 0;
   char *ptx = compile_module_to_ptx(ctx.module, sm_major, sm_minor, &ptx_size);

   LLVMDisposeBuilder(ctx.builder);
   LLVMDisposeModule(ctx.module);
   LLVMContextDispose(ctx.llvm_ctx);

   if (!ptx)
      return NULL;

   struct cp_shader_binary *bin = CALLOC_STRUCT(cp_shader_binary);
   bin->ptx_text = ptx;
   bin->ptx_size = ptx_size;
   bin->sm_major = sm_major;
   bin->sm_minor = sm_minor;

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
