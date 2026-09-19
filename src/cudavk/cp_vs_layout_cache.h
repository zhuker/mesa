#ifndef CP_VS_LAYOUT_CACHE_H
#define CP_VS_LAYOUT_CACHE_H
#include "cp_vs_layout_config.h"
#include "util/mesa-blake3.h"
/* Salt the existing graphics-stage key before any cache lookup. FS belongs
 * here even when interpolation is performed by a separately loaded kernel. */
static inline void
cp_vs_layout_hash_stage(unsigned char hash[BLAKE3_OUT_LEN], enum cp_vs_layout layout)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, hash, BLAKE3_OUT_LEN);
   const uint32_t layout_version = (uint32_t)layout;
   _mesa_blake3_update(&ctx, &layout_version, sizeof(layout_version));
   _mesa_blake3_final(&ctx, hash);
}
#endif
