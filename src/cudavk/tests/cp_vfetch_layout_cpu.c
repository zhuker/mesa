/* CPU-only scalar-layout seam test. No CUDA context or Vulkan device. */
#undef NDEBUG
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "../kernels/cp_vs_inline.c"

static void
check(struct cp_vertex_fetch_args *a, unsigned e, unsigned vertex,
      unsigned instance, unsigned row)
{
   uint32_t generic[4] = {99, 99, 99, 99};
   uint32_t specialized[4] = {88, 88, 88, 88};
   cp_vs_fetch_element(a, e, vertex, instance, row, (void *)generic);
   cp_vs_fetch_element_layout(a, e, vertex, instance, row, (void *)specialized,
      a->num_elements, a->elem_vb_idx[e], a->elem_src_offset[e], a->elem_src_stride[e], a->elem_instance_divisor[e], a->elem_attr_size[e], a->elem_nr_chan[e], a->elem_chan_bytes[e], a->elem_swizzle[e], a->elem_conv[e], a->elem_fill_w[e]);
   assert(!memcmp(generic, specialized, sizeof(generic)));
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
   cp_vs_fetch_element(&a, 31, 0, 0, 0, (void *)slot);
   assert(!(slot[0] | slot[1] | slot[2] | slot[3]));
   /* Independent expected COPY32 value, W completion and byte alignment. */
   a.elem_bases = 0;
   a.elem_src_offset[0] = 1;
   a.elem_src_stride[0] = 0;
   a.elem_instance_divisor[0] = 0;
   a.elem_conv[0] = CP_VF_CONV_COPY32;
   a.elem_attr_size[0] = 12;
   cp_vs_fetch_element_layout(&a, 0, 99, 4, 0, (void *)slot,
      1, 0, 1, 0, 0, 12, 3, 4, 0x3210, CP_VF_CONV_COPY32, 0x3f800000);
   assert(!memcmp(slot, storage + 1, 12));
   assert(slot[3] == 0x3f800000);
   puts("cp_vfetch_layout_cpu: PASS");
   return 0;
}
