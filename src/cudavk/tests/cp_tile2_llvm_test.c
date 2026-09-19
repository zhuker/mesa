/* CPU only: real LLVM legalization, verification and constant-store decoding.
 * No execution engine, JIT, CUDA context or Vulkan calls. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/TargetMachine.h>
#include "nir_to_ptx/cp_llvm_vs_tile2.h"

static unsigned bit_at(LLVMValueRef value, unsigned bit, LLVMTargetDataRef td)
{
   LLVMTypeRef type = LLVMTypeOf(value);
   if (bit >= LLVMSizeOfTypeInBits(td, type)) return 0;
   if (LLVMIsAConstantFP(value))
      return bit_at(LLVMConstBitCast(value, LLVMIntTypeInContext(LLVMGetTypeContext(type), LLVMSizeOfTypeInBits(td, type))), bit, td);
   if (LLVMIsAConstantInt(value)) {
      assert(LLVMGetIntTypeWidth(type) <= 64);
      return (LLVMConstIntGetZExtValue(value) >> bit) & 1;
   }
   if (LLVMGetTypeKind(type) == LLVMVectorTypeKind && LLVMIsConstant(value)) {
      unsigned width = LLVMSizeOfTypeInBits(td, LLVMGetElementType(type));
      LLVMValueRef elem = LLVMConstExtractElement(value,
         LLVMConstInt(LLVMInt32TypeInContext(LLVMGetTypeContext(type)), bit / width, false));
      return bit_at(elem, bit % width, td);
   }
   LLVMOpcode op = LLVMIsAConstantExpr(value) ? LLVMGetConstOpcode(value)
                                             : LLVMGetInstructionOpcode(value);
   switch (op) {
   case LLVMBitCast: case LLVMZExt: case LLVMTrunc:
      return bit_at(LLVMGetOperand(value, 0), bit, td);
   case LLVMLShr:
      return bit_at(LLVMGetOperand(value, 0), bit + LLVMConstIntGetZExtValue(LLVMGetOperand(value, 1)), td);
   default: LLVMDumpValue(value); assert(!"unexpected store value instruction"); return 0;
   }
}

static unsigned cases;
static LLVMTargetDataRef target_data;
static LLVMTargetMachineRef target_machine;
static unsigned ptx_cases;
static void emit_ptx(LLVMModuleRef module)
{
   LLVMModuleRef clone = LLVMCloneModule(module);
   LLVMSetTarget(clone, "nvptx64-nvidia-cuda");
   char *layout = LLVMCopyStringRepOfTargetData(target_data);
   LLVMSetDataLayout(clone, layout); LLVMDisposeMessage(layout);
   LLVMMemoryBufferRef buffer = NULL; char *error = NULL;
   if (LLVMTargetMachineEmitToMemoryBuffer(target_machine, clone, LLVMAssemblyFile, &error, &buffer)) {
      fprintf(stderr, "CPU PTX emission: %s\n", error ? error : "unknown error");
      assert(!"PTX emission failed");
   }
   assert(LLVMGetBufferSize(buffer) > 0);
   LLVMDisposeMemoryBuffer(buffer); LLVMDisposeMessage(error); LLVMDisposeModule(clone);
   ptx_cases++;
}

static void check(unsigned width, unsigned nc, unsigned comp, unsigned dyn,
                  unsigned phase, unsigned vertex, bool fp)
{
   LLVMContextRef c = LLVMContextCreate();
   LLVMModuleRef m = LLVMModuleCreateWithNameInContext("tile2_fixture", c);
   LLVMBuilderRef b = LLVMCreateBuilderInContext(c);
   LLVMTypeRef i64 = LLVMInt64TypeInContext(c), i8 = LLVMInt8TypeInContext(c);
   LLVMValueRef f = LLVMAddFunction(m, "fixture", LLVMFunctionType(LLVMVoidTypeInContext(c), NULL, 0, false));
   LLVMPositionBuilderAtEnd(b, LLVMAppendBasicBlockInContext(c, f, "entry"));
   LLVMTypeRef scalar = LLVMIntTypeInContext(c, width);
   LLVMValueRef lanes[16];
   for (unsigned n = 0; n < nc; n++)
      lanes[n] = LLVMConstInt(scalar, UINT64_C(0x8172635445362718) + n * 37, false);
   if (fp) {
      LLVMTypeRef ft = width == 16 ? LLVMHalfTypeInContext(c) :
                       width == 32 ? LLVMFloatTypeInContext(c) : LLVMDoubleTypeInContext(c);
      for (unsigned n = 0; n < nc; n++) lanes[n] = LLVMConstBitCast(lanes[n], ft);
   }
   LLVMValueRef data = nc == 1 ? lanes[0] : LLVMConstVector(lanes, nc);
   const uint64_t p = 0x10000 + phase * 16, stride = 32 * 16;
   assert(cp_llvm_store_tile2(b, c,
      LLVMConstIntToPtr(LLVMConstInt(i64, p, false), LLVMPointerType(i8, 0)),
      LLVMConstInt(i64, vertex, false),
      LLVMConstInt(LLVMInt32TypeInContext(c), dyn, false),
      32, 1, comp, data, target_data));
   LLVMBuildRetVoid(b);
   char *error = NULL;
   assert(!LLVMVerifyModule(m, LLVMReturnStatusAction, &error));
   LLVMDisposeMessage(error);
   unsigned char got[8192], expected[8192];
   memset(got, 0xcd, sizeof(got)); memset(expected, 0xcd, sizeof(expected));
   unsigned bytes = (width * nc + 7) / 8;
   /* Packed LLVM vectors have no per-element byte padding. */
   for (unsigned j = 0; j < bytes; j++) {
      unsigned char value = 0;
      for (unsigned bit = 0; bit < 8 && j * 8 + bit < width * nc; bit++) {
         unsigned logical = j * 8 + bit;
         uint64_t lane = UINT64_C(0x8172635445362718) + (logical / width) * 37;
         value |= ((lane >> (logical % width)) & 1) << bit;
      }
      uint32_t q = (uint32_t)((1u + dyn) * 16u + comp * 4u);
      q += j;
      unsigned v = phase + vertex;
      unsigned address = (v / 2) * (2 * stride) + (q / 16) * 32 + (v % 2) * 16 + q % 16;
      expected[address] = value;
   }
   LLVMTargetDataRef td = target_data;
   unsigned written = 0;
   for (LLVMValueRef i = LLVMGetFirstInstruction(LLVMGetFirstBasicBlock(f)); i; i = LLVMGetNextInstruction(i)) {
      if (LLVMGetInstructionOpcode(i) != LLVMStore) continue;
      LLVMValueRef value = LLVMGetOperand(i, 0), ptr = LLVMGetOperand(i, 1);
      LLVMValueRef addr = LLVMConstPtrToInt(ptr, i64);
      assert(LLVMIsAConstantInt(addr));
      uint64_t offset = LLVMConstIntGetZExtValue(addr) - 0x10000;
      unsigned size = LLVMStoreSizeOfType(td, LLVMTypeOf(value));
      assert(offset + size < sizeof(got));
      assert(offset / 16 == (offset + size - 1) / 16);
      for (unsigned j = 0; j < size; j++) {
         unsigned char byte = 0;
         for (unsigned bit = 0; bit < 8; bit++)
            byte |= bit_at(value, j * 8 + bit, td) << bit;
         assert(got[offset + j] == 0xcd);
         got[offset + j] = byte;
         written++;
      }
   }
   assert(written == bytes);
   assert(!memcmp(got, expected, sizeof(got)));
   if (phase == 0 && vertex == 0 && dyn == 0 && comp == 3 &&
       (nc <= 4 || nc == 8 || nc == 16)) emit_ptx(m);
   LLVMDisposeBuilder(b); LLVMDisposeModule(m); LLVMContextDispose(c);
   cases++;
}
/* Address-only edge fixture: synthetic pointers, no host dereference. Cross
 * the signed-i32 byte boundary after the one baseline start-offset wrap. */
static void address_limits(uint32_t dynamic, unsigned component)
{
   LLVMContextRef c = LLVMContextCreate();
   LLVMModuleRef m = LLVMModuleCreateWithNameInContext("offset_edges", c);
   LLVMBuilderRef b = LLVMCreateBuilderInContext(c);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(c), i64 = LLVMInt64TypeInContext(c);
   LLVMValueRef f = LLVMAddFunction(m,"edges",LLVMFunctionType(LLVMVoidTypeInContext(c),NULL,0,false));
   LLVMPositionBuilderAtEnd(b,LLVMAppendBasicBlockInContext(c,f,"entry"));
   LLVMValueRef lanes[4];
   for (unsigned i=0;i<4;i++) lanes[i]=LLVMConstInt(i64,0x12345678u+i,false);
   const uint64_t p=UINT64_C(0x1000000000010), vertex=UINT64_C(0x100000005);
   assert(cp_llvm_store_tile2(b,c,LLVMConstIntToPtr(LLVMConstInt(i64,p,false),LLVMPointerType(i32,0)),LLVMConstInt(i64,vertex,false),LLVMConstInt(i32,dynamic,false),32,0,component,LLVMConstVector(lanes,4),target_data));
   LLVMBuildRetVoid(b);
   char *error=NULL; assert(!LLVMVerifyModule(m,LLVMReturnStatusAction,&error)); LLVMDisposeMessage(error);
   unsigned j=0;
   for (LLVMValueRef i=LLVMGetFirstInstruction(LLVMGetFirstBasicBlock(f));i;i=LLVMGetNextInstruction(i)) {
      if (LLVMGetInstructionOpcode(i)!=LLVMStore) continue;
      uint64_t address=LLVMConstIntGetZExtValue(LLVMConstPtrToInt(LLVMGetOperand(i,1),i64));
      int64_t q=(int32_t)(dynamic*16u+component*4u);
      q+=j; /* Crucially NOT another i32 wrap here. */
      int64_t slot=q/16, byte=q%16;
      if (byte<0) { slot--; byte+=16; }
      uint64_t v=vertex+1;
      uint64_t expected=(p-16)+(v/2)*1024+(v%2)*16+(uint64_t)(slot*32)+(uint64_t)byte;
      assert(address==expected);
      j+=LLVMStoreSizeOfType(target_data,LLVMTypeOf(LLVMGetOperand(i,0)));
   }
   assert(j==32);
   LLVMDisposeBuilder(b);LLVMDisposeModule(m);LLVMContextDispose(c);
}

static void sequence(bool taken)
{
   LLVMContextRef c = LLVMContextCreate();
   LLVMModuleRef m = LLVMModuleCreateWithNameInContext("measured", c);
   LLVMBuilderRef b = LLVMCreateBuilderInContext(c);
   LLVMTypeRef i32 = LLVMInt32TypeInContext(c), i64 = LLVMInt64TypeInContext(c);
   LLVMValueRef f = LLVMAddFunction(m, "measured", LLVMFunctionType(LLVMVoidTypeInContext(c), NULL, 0, false));
   LLVMBasicBlockRef entry = LLVMAppendBasicBlockInContext(c, f, "entry");
   LLVMBasicBlockRef yes = LLVMAppendBasicBlockInContext(c, f, "conditional");
   LLVMBasicBlockRef done = LLVMAppendBasicBlockInContext(c, f, "done");
   LLVMPositionBuilderAtEnd(b, entry);
   /* Actual measured nine stores, plus a repeated conditional partial write. */
   unsigned slot[] = {0,1,2,2,3,4,5,6,7,1};
   unsigned comp[] = {0,0,0,2,0,0,0,0,0,1};
   unsigned nc[] = {4,2,2,1,4,4,4,4,4,1};
   unsigned char expected[800], got[800];
   memset(expected, 0xcd, sizeof(expected)); memset(got, 0xcd, sizeof(got));
   for (unsigned n = 0; n < 10; n++) {
      if (n == 9) {
         LLVMBuildCondBr(b, LLVMConstInt(LLVMInt1TypeInContext(c), taken, false), yes, done);
         LLVMPositionBuilderAtEnd(b, yes);
      }
      LLVMValueRef lanes[4];
      for (unsigned k = 0; k < nc[n]; k++) lanes[k] = LLVMConstInt(i32, 0x87654321u + n * 23 + k, false);
      LLVMValueRef data = nc[n] == 1 ? lanes[0] : LLVMConstVector(lanes, nc[n]);
      assert(cp_llvm_store_tile2(b,c,
         LLVMConstIntToPtr(LLVMConstInt(i64,0x10010,false),LLVMPointerType(i32,0)),
         LLVMConstInt(i64,0,false),LLVMConstInt(i32,0,false),8,slot[n],comp[n],data,target_data));
      if (n < 9 || taken)
         for (unsigned k = 0; k < nc[n] * 4; k++)
            expected[slot[n] * 32 + 16 + comp[n] * 4 + k] =
               (0x87654321u + n * 23 + k / 4) >> ((k % 4) * 8);
   }
   LLVMBuildBr(b, done); LLVMPositionBuilderAtEnd(b,done); LLVMBuildRetVoid(b);
   char *error = NULL; assert(!LLVMVerifyModule(m,LLVMReturnStatusAction,&error)); LLVMDisposeMessage(error);
   unsigned writes = 0;
   for (LLVMBasicBlockRef block = entry; block;) {
      LLVMBasicBlockRef next = NULL;
      for (LLVMValueRef i = LLVMGetFirstInstruction(block); i; i = LLVMGetNextInstruction(i)) {
         LLVMOpcode op = LLVMGetInstructionOpcode(i);
         if (op == LLVMBr) {
            next = LLVMGetSuccessor(i, LLVMIsConditional(i) && !LLVMConstIntGetZExtValue(LLVMGetCondition(i)) ? 1 : 0);
         } else if (op == LLVMStore) {
            LLVMValueRef value = LLVMGetOperand(i,0);
            uint64_t address = LLVMConstIntGetZExtValue(LLVMConstPtrToInt(LLVMGetOperand(i,1),i64)) - 0x10000;
            unsigned bytes = LLVMStoreSizeOfType(target_data,LLVMTypeOf(value));
            for (unsigned j=0;j<bytes;j++) {
               unsigned char byte=0;
               for (unsigned bit=0;bit<8;bit++) byte |= bit_at(value,j*8+bit,target_data)<<bit;
               got[address+j]=byte; writes++;
            }
         }
      }
      block = next;
   }
   assert(writes == 116 + (taken ? 4 : 0));
   assert(!memcmp(got,expected,sizeof(got)));
   LLVMDisposeBuilder(b);LLVMDisposeModule(m);LLVMContextDispose(c);
}

static void symbolic(void)
{
   LLVMContextRef c = LLVMContextCreate();
   LLVMModuleRef m = LLVMModuleCreateWithNameInContext("symbolic", c);
   LLVMBuilderRef b = LLVMCreateBuilderInContext(c);
   LLVMTypeRef i64 = LLVMInt64TypeInContext(c), i32 = LLVMInt32TypeInContext(c);
   LLVMTypeRef args[] = { LLVMPointerType(LLVMInt8TypeInContext(c), 0), i64, i32, LLVMVectorType(i64, 4) };
   LLVMValueRef f = LLVMAddFunction(m, "symbolic", LLVMFunctionType(LLVMVoidTypeInContext(c), args, 4, false));
   LLVMPositionBuilderAtEnd(b, LLVMAppendBasicBlockInContext(c, f, "entry"));
   assert(cp_llvm_store_tile2(b,c,LLVMGetParam(f,0),LLVMGetParam(f,1),LLVMGetParam(f,2),32,2,3,LLVMGetParam(f,3),target_data));
   LLVMBuildRetVoid(b);
   char *error = NULL;
   assert(!LLVMVerifyModule(m, LLVMReturnStatusAction, &error));
   LLVMDisposeMessage(error);
   unsigned stores = 0, wide_ops = 0, logical_i32_ops = 0, logical_sext = 0;
   for (LLVMValueRef i = LLVMGetFirstInstruction(LLVMGetFirstBasicBlock(f)); i; i = LLVMGetNextInstruction(i)) {
      LLVMOpcode op = LLVMGetInstructionOpcode(i);
      if (op == LLVMStore) stores++;
      if (op == LLVMAdd || op == LLVMMul || op == LLVMSub) {
         if (LLVMTypeOf(i) == i64) wide_ops++;
         else { assert(LLVMTypeOf(i) == i32); logical_i32_ops++; }
      }
      if (op == LLVMSExt) {
         assert(LLVMTypeOf(i) == i64 && LLVMTypeOf(LLVMGetOperand(i,0)) == i32);
         logical_sext++;
      }
   }
   assert(stores == 4 && wide_ops >= 8 && logical_i32_ops == 2 && logical_sext == 1);
   emit_ptx(m);
   LLVMDisposeBuilder(b); LLVMDisposeModule(m); LLVMContextDispose(c);
}

int main(void)
{
   LLVMInitializeNVPTXTargetInfo(); LLVMInitializeNVPTXTarget();
   LLVMInitializeNVPTXTargetMC(); LLVMInitializeNVPTXAsmPrinter();
   LLVMTargetRef target; char *error = NULL;
   assert(!LLVMGetTargetFromTriple("nvptx64-nvidia-cuda", &target, &error));
   LLVMTargetMachineRef tm = LLVMCreateTargetMachine(target, "nvptx64-nvidia-cuda", "sm_86", "+ptx75", LLVMCodeGenLevelDefault, LLVMRelocDefault, LLVMCodeModelDefault);
   assert(tm);
   target_machine = tm;
   target_data = LLVMCreateTargetDataLayout(tm);
   symbolic();
   sequence(false); sequence(true);
   address_limits(0x07ffffffu, 3);
   address_limits(0x08000000u, 0);
   address_limits(UINT32_MAX, 3);
   /* All these nonzero/high-bit offsets cancel or wrap to valid slot0/1
    * under the baseline logical i32 byte-offset arithmetic with base=1. */
   const uint32_t wrapped[] = {UINT32_MAX, INT32_MAX, 0x0fffffffu,
                               0x10000000u, 0x80000000u};
   for (unsigned d = 0; d < sizeof(wrapped)/sizeof(wrapped[0]); d++)
      for (unsigned phase = 0; phase < 2; phase++)
         for (unsigned comp = 0; comp < 4; comp++)
            check(64, 4, comp, wrapped[d], phase, 1, false);
   unsigned widths[] = {1, 8, 16, 32, 64};
   for (unsigned w = 0; w < 5; w++)
      for (unsigned nc = 1; nc <= 16; nc++)
         for (unsigned comp = 0; comp < 4; comp++)
            for (unsigned dyn = 0; dyn < 3; dyn++)
               for (unsigned phase = 0; phase < 2; phase++)
                  for (unsigned vertex = 0; vertex < 4; vertex++)
                     {
                     check(widths[w], nc, comp, dyn, phase, vertex, false);
                     if (widths[w] >= 16) check(widths[w], nc, comp, dyn, phase, vertex, true);
                  }
   printf("TILE2 actual LLVM stores: %u fixtures passed\n", cases);
   printf("CPU NVPTX emission: %u representative partial/cross-slot/dynamic forms passed\n", ptx_cases);
   LLVMDisposeTargetData(target_data); LLVMDisposeTargetMachine(tm);
   return 0;
}
