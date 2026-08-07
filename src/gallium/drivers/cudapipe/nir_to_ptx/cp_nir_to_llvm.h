#ifndef CP_NIR_TO_LLVM_H
#define CP_NIR_TO_LLVM_H

#include <cuda.h>
#include <stdbool.h>
#include <stdint.h>

struct nir_shader;

struct cp_shader_binary {
   char *ptx_text;
   size_t ptx_size;
   CUmodule module;
   CUfunction kernel;
   int sm_major;
   int sm_minor;
};

struct cp_shader_binary *
cp_compile_nir_to_ptx(struct nir_shader *nir, int sm_major, int sm_minor);

void
cp_shader_binary_destroy(struct cp_shader_binary *bin);

#endif
