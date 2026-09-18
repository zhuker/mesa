/* CPU-only finite-corpus check of the independent pre-specialization oracle
 * against the current generic fetch.  The oracle records its exact provenance
 * in cp_vfetch_eq_oracle.h.  This is not device-equivalence replay evidence.
 * Run from the repository root:
 * cc -std=c11 -O2 -Wno-attributes -Wno-unknown-pragmas \
 *    src/cudavk/tests/cp_vfetch_eq_oracle_cpu.c -o /tmp/cp-vfetch-oracle-test
 * /tmp/cp-vfetch-oracle-test
 * Add -fsanitize=address,undefined for host memory/UB checking.
 */
/* CPU-only scalar-layout seam test. No CUDA context or Vulkan device. */
#undef NDEBUG
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "../kernels/cp_vs_inline.c"
#define __float_as_uint(f) cp_vs_float_as_uint(f)
#define CP_VFETCH_EQ_ORACLE_INLINE static inline
#define CP_VFETCH_EQ_ORACLE_UNROLL
#include "../kernels/cp_vfetch_eq_oracle.h"
#undef __float_as_uint

static void
check(struct cp_vertex_fetch_args *a, unsigned e, unsigned vertex,
      unsigned instance, unsigned row)
{
   _Alignas(16) uint32_t generic[4] = {0, 0, 0, 0};
   _Alignas(16) uint32_t oracle[4] = {88, 88, 88, 88};
   cp_vf_lane_element(a, e, vertex, instance, row, (void *)generic);
   cp_vfetch_eq_oracle_element(a, e, vertex, instance, row, (void *)oracle);
   assert(!memcmp(generic, oracle, sizeof(generic)));
}

int main(void)
{
   unsigned char storage[2048];
   for (unsigned i = 0; i < sizeof(storage); i++)
      storage[i] = (i * 73 + 19) & 255;
   struct cp_vertex_fetch_args a = {0};
   a.num_elements = 1;
   a.vb_bases[0] = (uintptr_t)storage;
   a.elem_src_stride[0] = 32;
   a.elem_swizzle[0] = 0x3210;
   a.elem_fill_w[0] = 0x3f800000;
   for (unsigned conv = CP_VF_CONV_COPY32; conv <= CP_VF_CONV_FLOAT16; conv++) {
      a.elem_conv[0] = conv;
      for (unsigned bytes = 1; bytes <= 4; bytes *= 2) {
         /* COPY32 formats have whole 32-bit components. */
         if (conv == CP_VF_CONV_COPY32 && bytes != 4)
            continue;
         a.elem_chan_bytes[0] = bytes;
         for (unsigned chans = 1; chans <= 4; chans++) {
            a.elem_nr_chan[0] = chans;
            a.elem_attr_size[0] = bytes * chans;
            for (unsigned offset = 0; offset < 16; offset++) {
               a.elem_src_offset[0] = offset;
               for (unsigned divisor = 0; divisor <= 3; divisor++) {
                  a.elem_instance_divisor[0] = divisor;
                  a.start_instance = 2;
                  check(&a, 0, 3, 5, 0);
               }
            }
         }
      }
   }
   /* Batch-row bases, nonzero firstInstance and integer divisor. */
   uint64_t bases[2 * CP_VB_TABLE_STRIDE] = {0};
   bases[CP_VB_TABLE_STRIDE] = (uintptr_t)(storage + 128);
   a.elem_bases = (uintptr_t)bases;
   check(&a, 0, 3, 5, 1);
   check(&a, 0, 3, 5, 0); /* null row remains zero */
   /* Absent inputs must never index an element table beyond its end. */
   uint32_t slot[4] = {1, 2, 3, 4};
   cp_vfetch_eq_oracle_element(&a, 31, 0, 0, 0, (void *)slot);
   assert(!(slot[0] | slot[1] | slot[2] | slot[3]));
   /* Independent expected COPY32 value, W completion and byte alignment. */
   a.elem_bases = 0;
   a.elem_src_offset[0] = 1;
   a.elem_src_stride[0] = 0;
   a.elem_instance_divisor[0] = 0;
   a.elem_conv[0] = CP_VF_CONV_COPY32;
   a.elem_attr_size[0] = 12;
   cp_vfetch_eq_oracle_element(&a, 0, 99, 4, 0, (void *)slot);
   assert(!memcmp(slot, storage + 1, 12));
   assert(slot[3] == 0x3f800000);

   /* Out-of-range slots must ignore even deliberately poisoned metadata. */
   a.num_elements = 100;
   a.elem_bases = 1;
   for (unsigned e = 16; e < 64; ++e) {
      memset(slot, 0xa5, sizeof(slot));
      cp_vfetch_eq_oracle_element(&a, e, 0, 0, 0, (void *)slot);
      assert(!(slot[0] | slot[1] | slot[2] | slot[3]));
   }
   a.elem_bases = 0;
   /* Exhaust all half encodings, preserving NaN payloads and signed zero. */
   for (unsigned h = 0; h <= 65535; ++h)
      assert(cp_vf_expand(h, 2, CP_VF_CONV_FLOAT16) ==
             cp_vfetch_eq_oracle_expand(h, 2, CP_VF_CONV_FLOAT16));
   a.num_verts = 24;
   a.first_vertex = 7;
   a.verts_per_instance = 6;
   uint32_t indices[64], refs[64], instances[64];
   for (unsigned j = 0; j < 64; ++j) {
      indices[j] = j * 3; refs[j] = 63-j; instances[j] = j/3;
   }
   struct cp_draw_slice slices[2] = {{0,0,4,3,0,0}, {12,12,8,4,0,0}};
   for (unsigned mode = 0; mode < 4; ++mode) {
      a.index_buffer = mode == 1 || mode == 2 ? (uintptr_t)indices : 0;
      a.index_size = 4;
      a.num_draw_slices = mode == 2 ? 2 : 0;
      a.draw_slices = (uintptr_t)slices;
      a.vertex_ids = mode == 3 ? (uintptr_t)refs : 0;
      a.instance_ids = (uintptr_t)instances;
      for (unsigned v = 0; v < 32; ++v) {
         uint32_t x[3] = {99,98,97}, y[3] = {99,98,97};
         int valid = cp_vf_lane_ids(&a, v, x, x+1, x+2);
         int oracle_valid = cp_vfetch_eq_oracle_ids(&a, v, y, y+1, y+2);
         assert(valid == oracle_valid && !memcmp(x,y,sizeof(x)));
      }
   }

   /* Exercise the actual comparator helpers with private host report rows. */
   struct cp_vfetch_eq_row eq_rows[2] = {0};
   struct cp_vfetch_eq_report report = {0};
   a.eq_rows = (uintptr_t)eq_rows;
   a.eq_report = (uintptr_t)&report;
   a.eq_dispatch = 123;
   a.eq_break = 0;
   eq_rows[0].valid = 1;
   eq_rows[0].vertex_id = 17;
   eq_rows[0].instance_id = 3;
   eq_rows[0].row = 2;
   eq_rows[0].slots[0][0] = 0x80000000u; /* signed zero bit pattern */
   eq_rows[0].slots[0][1] = 0x7fc12345u; /* NaN payload bit pattern */
   memcpy(slot, eq_rows[0].slots[0], sizeof(slot));
   assert(!cp_vs_fetch_eq_ids(&a, 0, 1, 17, 3, 2));
   assert(!cp_vs_fetch_eq_slot(&a, 0, 0, (void *)slot));
   cp_vs_fetch_eq_finish(&a, 0);
   assert(report.invocations == 1 && report.valid == 1);
   assert(report.slots == 1 && report.words == 4);
   assert(!report.differing_invocations && !report.diff_words);
   /* Live-bit negative control must be detected and then restore the slot. */
   a.eq_break = 1;
   assert(cp_vs_fetch_eq_slot(&a, 0, 0, (void *)slot));
   cp_vs_fetch_eq_finish(&a, 1);
   assert(!memcmp(slot, eq_rows[0].slots[0], sizeof(slot)));
   assert(report.injections == 1 && report.diff_slots == 1);
   assert(report.diff_words == 1 && report.differing_invocations == 1);
   assert(report.first_claim == 1 && report.first_dispatch == 123);
   assert(report.first_expected == 0x80000000u);
   assert(report.first_actual == 0x80000001u);
   a.eq_break = 0;
   /* A candidate rejection mismatch is counted without the final hook. */
   assert(cp_vs_fetch_eq_ids(&a, 0, 0, 0, 0, 0));
   assert(report.diff_validity == 1 && report.differing_invocations == 2);
   assert(!cp_vs_fetch_eq_ids(&a, 1, 0, 0, 0, 0));
   /* ID mismatch on a valid invocation is committed by the final hook. */
   assert(cp_vs_fetch_eq_ids(&a, 0, 1, 18, 3, 2));
   cp_vs_fetch_eq_finish(&a, 1);
   assert(report.diff_ids == 1 && report.differing_invocations == 3);
   puts("cp_vfetch_eq_oracle_cpu: PASS");
   return 0;
}
