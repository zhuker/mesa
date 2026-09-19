/* Exact-byte TILE2 legalization. Shared with the CPU LLVM fixture. */
#ifndef CP_LLVM_VS_TILE2_H
#define CP_LLVM_VS_TILE2_H
#include <llvm-c/Core.h>
#include <llvm-c/Target.h>
#include <stdbool.h>
#include <stdint.h>

static bool
cp_llvm_store_tile2(LLVMBuilderRef b, LLVMContextRef c, LLVMValueRef output,
                    LLVMValueRef vertex, LLVMValueRef dynamic_slot,
                    unsigned slots, unsigned base, unsigned component,
                    LLVMValueRef data, LLVMTargetDataRef td)
{
   LLVMTypeRef i64 = LLVMInt64TypeInContext(c);
#define C64(x) LLVMConstInt(i64, (x), false)
   LLVMTypeRef offset_type = LLVMTypeOf(dynamic_slot);
   if (LLVMGetTypeKind(offset_type) != LLVMIntegerTypeKind ||
       LLVMGetIntTypeWidth(offset_type) > 64 || LLVMTypeOf(vertex) != i64)
      return false;
   LLVMTypeRef type = LLVMTypeOf(data);
   LLVMTypeRef scalar = LLVMGetTypeKind(type) == LLVMVectorTypeKind
      ? LLVMGetElementType(type) : type;
   LLVMTypeKind kind = LLVMGetTypeKind(scalar);
   if (kind != LLVMIntegerTypeKind && kind != LLVMHalfTypeKind &&
       kind != LLVMFloatTypeKind && kind != LLVMDoubleTypeKind)
      return false;
   unsigned long long bytes = LLVMStoreSizeOfType(td, type);
   unsigned long long bits = LLVMSizeOfTypeInBits(td, type);
   if (!slots || !bytes || bits > UINT32_MAX - 7u)
      return false;

   /* P is a true slot0 pointer. Recover its half, not an arbitrary slot's. */
   LLVMValueRef p = LLVMBuildPtrToInt(b, output, i64, "tile2_p");
   LLVMValueRef phase = LLVMBuildAnd(b, LLVMBuildLShr(b, p, C64(4), ""), C64(1), "");
   LLVMValueRef origin = LLVMBuildSub(b, p, LLVMBuildMul(b, phase, C64(16), ""), "");
   LLVMValueRef v = LLVMBuildAdd(b, phase, vertex, "");
   LLVMValueRef tile = LLVMBuildMul(b, LLVMBuildLShr(b, v, C64(1), ""),
                                    C64((uint64_t)slots * 32), "");
   LLVMValueRef half = LLVMBuildMul(b, LLVMBuildAnd(b, v, C64(1), ""), C64(16), "");
   LLVMValueRef start = LLVMBuildAdd(b, origin, LLVMBuildAdd(b, tile, half, ""), "");
   /* Preserve the baseline's logical output-offset wrap/cancellation before
    * widening. NIR offsets are i32 slot values: e.g. base=1, offset=-1 names
    * slot0, not slot2^32. Only vertex/tile addressing is newly widened.
    * Sign-extension matches the baseline GEP's i32 byte-offset convention. */
   LLVMTypeRef i32 = LLVMInt32TypeInContext(c);
   LLVMValueRef off32 = offset_type == i32 ? dynamic_slot :
      LLVMGetIntTypeWidth(offset_type) < 32 ?
         LLVMBuildZExt(b, dynamic_slot, i32, "") :
         LLVMBuildTrunc(b, dynamic_slot, i32, "");
   const uint32_t q = (uint32_t)((uint64_t)base * 16 + (uint64_t)component * 4);
   LLVMValueRef q32 = LLVMBuildAdd(b,
      LLVMBuildMul(b, off32, LLVMConstInt(i32, 16, false), ""),
      LLVMConstInt(i32, q, false), "tile2_logical_byte_offset");
   LLVMValueRef first_slot = LLVMBuildAShr(b,
      LLVMBuildSExt(b, q32, i64, "tile2_logical_offset"), C64(4), "");
   LLVMValueRef value = NULL;
   for (uint64_t j = 0; j < bytes;) {
      unsigned in_slot = (q + j) % 16;
      unsigned n = 16 - in_slot;
      if (n > bytes - j) n = bytes - j;
      /* Keep an entire in-slot LLVM store, including packed subword types. */
      bool whole = j == 0 && n == bytes;
      if (!whole) {
         unsigned legal = 1;
         while (legal * 2 <= n && in_slot % (legal * 2) == 0)
            legal *= 2;
         n = legal;
      }
      LLVMValueRef slot = LLVMBuildAdd(b, first_slot, C64(((q % 16) + j) / 16), "");
      LLVMValueRef address = LLVMBuildAdd(b, start,
         LLVMBuildAdd(b, LLVMBuildMul(b, slot, C64(32), ""), C64(in_slot), ""), "");
      LLVMValueRef piece = data;
      if (!whole) {
         if (!value) {
            LLVMTypeRef packed = LLVMIntTypeInContext(c, bits);
            value = type == packed ? data : LLVMBuildBitCast(b, data, packed, "tile2_bits");
            if (bits != bytes * 8)
               value = LLVMBuildZExt(b, value, LLVMIntTypeInContext(c, bytes * 8), "");
         }
         piece = j ? LLVMBuildLShr(b, value, LLVMConstInt(LLVMTypeOf(value), j * 8, false), "") : value;
         LLVMTypeRef chunk = LLVMIntTypeInContext(c, n * 8);
         if (LLVMTypeOf(piece) != chunk)
            piece = LLVMBuildTrunc(b, piece, chunk, "tile2_piece");
      }
      LLVMValueRef ptr = LLVMBuildIntToPtr(b, address, LLVMPointerType(LLVMTypeOf(piece), 0), "");
      LLVMValueRef store = LLVMBuildStore(b, piece, ptr);
      unsigned align = 16;
      while (in_slot % align) align /= 2;
      LLVMSetAlignment(store, align);
      j += n;
   }
#undef C64
   return true;
}
#endif
