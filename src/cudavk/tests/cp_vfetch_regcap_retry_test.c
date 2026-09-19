/* CPU-only mock for the production retry helpers extracted by the runner. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "nir_to_ptx/cp_nir_to_llvm.h"
#include "cp_debug.h"
#define MIN2(a,b) ((a)<(b)?(a):(b))
#define CP_SHADER_BLOCK_THREADS 256
static struct cp_debug flags;
const struct cp_debug *cp_debug = &flags;
static int loads, unloads, fail_at, calls, regs = 32, spill, blocks = 8;
static CUmodule last_unloaded;
static struct cp_shader_exec classic, fused;
static int fault(void) { return ++calls == fail_at ? 1 : CUDA_SUCCESS; }
static int mock_device(CUdevice *d) { *d = 0; return fault(); }
static int mock_devattr(int *v, int attr, CUdevice d)
{ (void)attr; (void)d; *v = 65536; return fault(); }
static int mock_attr(int *v, int attr, CUfunction fn)
{
   bool base = fn == classic.kernel;
   *v = attr == CU_FUNC_ATTRIBUTE_NUM_REGS ? (base ? 32 : regs)
                                         : (base ? 0 : spill);
   return fault();
}
static int mock_occupancy(int *v, CUfunction fn, int threads, size_t shared)
{
   assert(threads == 256 && shared == 0);
   *v = fn == classic.kernel ? 8 : blocks;
   return fault();
}
static int mock_function(CUfunction *fn, CUmodule mod, const char *name)
{
   assert(mod == (CUmodule)3 && !strcmp(name, "main"));
   *fn = (CUfunction)3;
   return fault();
}
static int mock_unload(CUmodule mod)
{ ++unloads; last_unloaded = mod; return CUDA_SUCCESS; }
static CUresult load_shader_module(CUmodule *mod, const char *ptx,
                                   const char *sampler, const char *helper,
                                   int cap)
{
   ++loads;
   assert(ptx == fused.ptx_text && sampler == fused.sampler_ptx &&
          helper == fused.fs_helper_ptx && cap == 32);
   *mod = (CUmodule)3;
   return fault();
}
#define cuCtxGetDevice mock_device
#define cuDeviceGetAttribute mock_devattr
#define cuFuncGetAttribute mock_attr
#define cuOccupancyMaxActiveBlocksPerMultiprocessor mock_occupancy
#define cuModuleGetFunction mock_function
#define cuModuleUnload mock_unload
#include "retry.inc"
static void reset(void)
{
   flags = (struct cp_debug){ .vfetch_regcap_retry = true };
   classic = (struct cp_shader_exec){ .kernel = (CUfunction)1,
      .module = (CUmodule)1, .num_regs = 32, .blocks_per_sm = 8 };
   fused = (struct cp_shader_exec){ .kernel = (CUfunction)2,
      .module = (CUmodule)2, .num_regs = 35, .blocks_per_sm = 6,
      .ptx_text = "validated fused PTX", .sampler_ptx = "sampler",
      .globals_resolved = true, .sym_sampler_table = 11, .sym_quad_derivs = 12 };
   loads = unloads = calls = fail_at = spill = 0;
   regs = 32; blocks = 8; last_unloaded = NULL;
}
static bool run(unsigned base_local, unsigned fx_local)
{
   struct cp_shader_exec base = classic, old = fused;
   bool ok = cp_vs_fetch_retry_reg_cap(&classic, &fused, base_local, fx_local, 1);
   assert(!memcmp(&base, &classic, sizeof(base)));
   if (!ok) assert(!memcmp(&old, &fused, sizeof(old)));
   return ok;
}
int main(void)
{
   reset(); assert(run(0, 0));
   assert(loads == 1 && unloads == 1 && last_unloaded == (CUmodule)2);
   assert(fused.module == (CUmodule)3 && fused.reg_cap == 32);
   assert(fused.num_regs == 32 && fused.blocks_per_sm == 8);
   assert(!fused.globals_resolved && !fused.sym_sampler_table && !fused.sym_quad_derivs);
   assert(!strcmp(fused.ptx_text, "validated fused PTX"));
   reset(); flags.vfetch_regcap_retry = false; assert(!run(0, 0) && !loads);
   reset(); flags.no_regcap = true; assert(!run(0, 0) && !loads);
   reset(); flags.max_registers = 32; assert(!run(0, 0) && !loads);
   reset(); flags.vfetch_decline_nth = 1; assert(!run(0, 0) && !loads);
   reset(); flags.vfetch_decline_nth = UINT32_MAX; assert(!run(0, 0) && !loads);
   reset(); assert(!run(0, 1) && !loads);
   reset(); fused.spill_bytes = 4; assert(!run(0, 0) && !loads);
   reset(); fused.blocks_per_sm = 8; assert(!run(0, 0) && !loads);
   reset(); fused.num_regs = 32; assert(!run(0, 0) && !loads);
   reset(); fused.alt_module = (CUmodule)4; assert(!run(0, 0) && !loads);
   reset(); fused.reg_cap = 40; assert(!run(0, 0) && !loads);
   reset(); classic.reg_cap = 40; assert(!run(0, 0) && !loads);
   reset(); fused.ptx_text = NULL; assert(!run(0, 0) && !loads);
   reset(); spill = 8; assert(!run(0, 0) && loads == 1 && unloads == 1);
   assert(last_unloaded == (CUmodule)3);
   reset(); blocks = 6; assert(!run(0, 0) && loads == 1 && unloads == 1);
   reset(); regs = 33; assert(!run(0, 0) && loads == 1 && unloads == 1);
   reset(); regs = 0; assert(!run(0, 0) && loads == 1 && unloads == 1);
   /* Every CUDA query/load/lookup failure: original private module survives. */
   for (int i = 1; i <= 10; ++i) {
      reset(); fail_at = i; assert(!run(0, 0));
      assert(unloads == (i >= 6));
      if (unloads) assert(last_unloaded == (CUmodule)3);
   }
   puts("fused-only register retry: CPU mocks passed (not GPU evidence)");
   return 0;
}
