#ifndef CP_NIR_TO_LLVM_H
#define CP_NIR_TO_LLVM_H

#include <cuda.h>
#include <stdbool.h>
#include <stdint.h>

struct nir_shader;

#define CP_MAX_IO_SLOTS 32

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

   /* Whether the shader reads gl_VertexIndex. Only then does a non-indexed
    * draw have to materialise the vertex id array the shader reads from. */
   bool reads_vertex_id;

   /* Whether the shader can discard. An alpha-tested draw needs several passes
    * because visibility is resolved before shading. */
   bool uses_discard;

   /* Which VARYING_SLOT_* each I/O slot carries, so a fragment shader's inputs
    * can be matched to the vertex shader's outputs by location rather than by
    * position. Slots with no variable are VARYING_SLOT_MAX. */
   unsigned in_location[CP_MAX_IO_SLOTS];
   unsigned out_location[CP_MAX_IO_SLOTS];
};

/* `sampler_ptx` is the relocatable PTX of the texture sampler, linked in when
 * the shader samples textures. May be NULL for shaders that cannot. */
struct cp_shader_binary *
cp_compile_nir_to_ptx(struct nir_shader *nir, int sm_major, int sm_minor,
                      const char *sampler_ptx);

void
cp_shader_binary_destroy(struct cp_shader_binary *bin);

#endif
