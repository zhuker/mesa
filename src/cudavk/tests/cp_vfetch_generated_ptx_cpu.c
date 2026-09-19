/* CPU-only production compiler regression. No CUDA library or context.
 * Only module loading and resource queries are mocked; NIR lowering, LLVM
 * generation, embedded helper linking, optimization and PTX emission are real.
 * Mock costs test admission control flow, NOT device occupancy or correctness.
 */
#include "nir_to_ptx/cp_nir_to_llvm.h"
#include "cp_debug.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct cp_debug debug;
const struct cp_debug *cp_debug = &debug;
static unsigned modules;
static char *loaded_ptx[5];

#define REQUIRE(c) do { if (!(c)) { \
   fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); exit(1); \
} } while (0)

CUresult CUDAAPI
cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned n,
                  CUjit_option *options, void **values)
{
   REQUIRE(strstr(image, ".entry main("));
   REQUIRE(modules < ARRAY_SIZE(loaded_ptx));
   loaded_ptx[modules] = strdup(image);
   REQUIRE(loaded_ptx[modules]);
   *module = (CUmodule)(uintptr_t)++modules;
   return CUDA_SUCCESS;
}
CUresult CUDAAPI
cuModuleGetFunction(CUfunction *fn, CUmodule module, const char *name)
{
   REQUIRE(!strcmp(name, "main"));
   *fn = (CUfunction)module;
   return CUDA_SUCCESS;
}
CUresult CUDAAPI
cuFuncGetAttribute(int *value, CUfunction_attribute attr, CUfunction fn)
{
   REQUIRE(attr == CU_FUNC_ATTRIBUTE_NUM_REGS ||
           attr == CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
   /* Deliberately unattractive CHECK costs must not change admission. */
   *value = attr == CU_FUNC_ATTRIBUTE_NUM_REGS
      ? ((uintptr_t)fn == 5 ? 128 : 16)
      : ((uintptr_t)fn == 5 ? 1024 : 0);
   return CUDA_SUCCESS;
}
CUresult CUDAAPI
cuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, CUfunction fn,
                                           int threads, size_t shared)
{
   *blocks = (uintptr_t)fn == 5 ? 1 : 4;
   return CUDA_SUCCESS;
}
CUresult CUDAAPI cuModuleUnload(CUmodule module) { return CUDA_SUCCESS; }

/* Unexpected CUDA paths fail, rather than accidentally touching a driver. */
CUresult CUDAAPI cuCtxGetDevice(CUdevice *dev) { abort(); }
CUresult CUDAAPI cuDeviceGetAttribute(int *v, CUdevice_attribute a, CUdevice d) { abort(); }
CUresult CUDAAPI cuEventDestroy(CUevent event) { abort(); }
CUresult CUDAAPI cuGetErrorString(CUresult err, const char **str) { abort(); }
CUresult CUDAAPI cuModuleLoadData(CUmodule *m, const void *data) { abort(); }
CUresult CUDAAPI
cuLinkCreate(unsigned n, CUjit_option *opts, void **vals, CUlinkState *state) { abort(); }
CUresult CUDAAPI
cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size,
              const char *name, unsigned n, CUjit_option *opts, void **vals) { abort(); }
CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **data, size_t *size) { abort(); }
CUresult CUDAAPI cuLinkDestroy(CUlinkState state) { abort(); }

static struct cp_shader_binary *
compile(bool check, const struct cp_vs_fetch_layout *layout)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &options,
                                                 "vfetch PTX CPU gate");
   b.shader->num_inputs = 3;
   b.shader->num_outputs = 1;
   /* Sparse live slots 0 and 2. Both feed output, so DCE cannot erase them. */
   nir_def *x = nir_load_input(&b, 4, 32, nir_imm_int(&b, 0), .base = 0,
                             .dest_type = nir_type_float32);
   nir_def *y = nir_load_input(&b, 4, 32, nir_imm_int(&b, 0), .base = 2,
                             .dest_type = nir_type_float32);
   nir_store_output(&b, nir_fadd(&b, x, y), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS});
   debug.vfetch_eq_check = check;
   struct cp_shader_binary *bin = cp_compile_nir_to_ptx(
      b.shader, 8, 6, NULL, NULL, NULL, false, false, false, false, layout);
   ralloc_free(b.shader);
   REQUIRE(bin);
   return bin;
}

static unsigned
count(const char *text, const char *needle)
{
   unsigned n = 0;
   while ((text = strstr(text, needle))) { ++n; text += strlen(needle); }
   return n;
}

static void
save(const char *dir, const char *name, const char *ptx)
{
   if (!dir) return;
   char path[4096];
   REQUIRE(snprintf(path, sizeof(path), "%s/%s.ptx", dir, name) < sizeof(path));
   FILE *f = fopen(path, "w");
   REQUIRE(f);
   REQUIRE(fputs(ptx, f) >= 0);
   REQUIRE(fclose(f) == 0);
}

int
main(int argc, char **argv)
{
   REQUIRE(argc <= 2);
   glsl_type_singleton_init_or_ref();
   struct cp_vs_fetch_layout layout = {.count = 3};
   for (unsigned i = 0; i < layout.count; i++) {
      layout.elem[i] = (struct cp_vertex_elem) {
         .src_stride = 16, .attr_size = 16, .nr_chan = 4,
         .chan_bytes = 4, .swizzle = 0xe4,
      };
   }
   struct cp_shader_binary *off = compile(false, &layout);
   struct cp_shader_binary *on = compile(true, &layout);
   REQUIRE(off->vs_census->verdict == CP_VS_FETCH_ADMITTED);
   REQUIRE(on->vs_census->verdict == CP_VS_FETCH_ADMITTED);
   REQUIRE(off->vs_live_slots == 5 && on->vs_live_slots == 5);
   REQUIRE(off->vs_num_slots == 3 && on->vs_num_slots == 3);
   REQUIRE(modules == 5); /* two ordinary builds plus one checked build */
   REQUIRE(!strcmp(loaded_ptx[1], loaded_ptx[3]));
   REQUIRE(on->vs_census->fused_regs == off->vs_census->fused_regs);
   REQUIRE(on->exec[CP_SHADER_EXEC_VS_FETCH].spill_bytes == 1024);
   REQUIRE(on->exec[CP_SHADER_EXEC_VS_FETCH].blocks_per_sm == 1);
   const char *plain = off->exec[CP_SHADER_EXEC_VS_FETCH].ptx_text;
   const char *checked = on->exec[CP_SHADER_EXEC_VS_FETCH].ptx_text;
   REQUIRE(plain && checked);
   REQUIRE(!strstr(plain, "atom."));
   REQUIRE(!strstr(plain, "st.volatile"));
   REQUIRE(strstr(checked, "atom.add.u64"));
   REQUIRE(strstr(checked, "atom.cas.b64"));
   /* Each of the two live slot hooks must retain its byte fault and restore.
    * Volatility forces the actual slot address to survive LLVM optimization.
    */
   REQUIRE(count(checked, "st.volatile.u8") == 4);
   REQUIRE(count(checked, "ld.volatile.u8") == 4);
   REQUIRE(!strstr(checked, "call.uni"));
   REQUIRE(!strcmp(off->exec[CP_SHADER_EXEC_CLASSIC].ptx_text,
                   on->exec[CP_SHADER_EXEC_CLASSIC].ptx_text));
   save(argc == 2 ? argv[1] : NULL, "check-off", plain);
   save(argc == 2 ? argv[1] : NULL, "check-on", checked);
   printf("PASS: real VS PTX; sparse live slots=0,2; CHECK off/on; "
          "report atomics; four volatile byte stores (fault+restore)\n");
   cp_shader_binary_destroy(off);
   cp_shader_binary_destroy(on);
   for (unsigned i = 0; i < modules; i++) free(loaded_ptx[i]);
   glsl_type_singleton_decref();
   return 0;
}
