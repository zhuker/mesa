#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "cp_vs_layout_cache.h"
#include "nir_to_ptx/cp_nir_to_llvm.h"
#include "cp_kernels.h"
int main(void)
{
   for (unsigned seed = 0; seed < 256; seed++) {
      unsigned char keys[2][2][BLAKE3_OUT_LEN];
      for (unsigned layout = 0; layout < 2; layout++)
         for (unsigned repeat = 0; repeat < 2; repeat++) {
            memset(keys[layout][repeat], seed, BLAKE3_OUT_LEN);
            cp_vs_layout_hash_stage(keys[layout][repeat], (enum cp_vs_layout)layout);
         }
      assert(memcmp(keys[0][0], keys[1][0], BLAKE3_OUT_LEN));
      assert(!memcmp(keys[0][0], keys[0][1], BLAKE3_OUT_LEN));
      assert(!memcmp(keys[1][0], keys[1][1], BLAKE3_OUT_LEN));
   }
   puts("Actual Mesa BLAKE3 stage cache: 256 seeds separate/stable in both layouts");
   for (unsigned layout = 0; layout < 2; layout++) {
      struct cp_shader_binary bin = {0};
      struct cp_kernels kernels = {0};
      bin.vs_layout = kernels.vs_layout = (enum cp_vs_layout)layout;
      for (unsigned mode = 0; mode < CP_SHADER_EXEC_COUNT; mode++) {
         struct cp_shader_exec *exec = &bin.exec[mode];
         exec->vs_layout = exec->helper_layout = bin.vs_layout;
         assert(cp_shader_exec(&bin, mode) == exec);
         assert(cp_shader_exec_const(&bin, mode) == exec);
         exec->fs_helper_ptx = "same-family helper";
         assert(cp_shader_exec_family_matches(&bin, exec));
         exec->helper_layout = (enum cp_vs_layout)(1-layout);
         assert(!cp_shader_exec_helper_matches(exec));
         assert(!cp_shader_exec(&bin,mode));
         assert(!cp_shader_exec_const(&bin,mode));
         exec->helper_layout = bin.vs_layout;
         exec->vs_layout = (enum cp_vs_layout)(1-layout);
         assert(!cp_shader_exec_family_matches(&bin,exec));
         assert(!cp_shader_exec(&bin,mode));
         exec->fs_helper_ptx = NULL;
         assert(!cp_shader_exec(&bin,mode));
         exec->vs_layout = bin.vs_layout;
         assert(cp_shader_exec(&bin,mode));
         assert(kernels.vs_layout == bin.vs_layout);
      }
   }
   assert(!cp_shader_exec(NULL, CP_SHADER_EXEC_CLASSIC));
   puts("TILE2 all exec/helper family guards passed (CPU only)");
   return 0;
}
