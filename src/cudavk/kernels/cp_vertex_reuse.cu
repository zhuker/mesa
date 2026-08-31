/*
 * Diagnostic-only B0 census.  This kernel does not write shader results.  It
 * repeats only cp_vf_lane_ids() after the real vertex fetch and VS have run.
 */
#include "cp_rast_types.h"
#include "cp_vf_lane.h"

struct cp_vertex_reuse_entry {
   uint32_t state;       /* generation << 1, low bit means ready */
   uint32_t row;
   uint32_t vertex;
   uint32_t instance;
};

static_assert(sizeof(struct cp_vertex_fetch_args) == 904,
              "vertex reuse fetch ABI");

__device__ static inline uint64_t
cp_vertex_reuse_mix(uint32_t row, uint32_t vertex, uint32_t instance)
{
   /* Integer fields are mixed independently.  The hash chooses a probe start;
    * equality always compares all three original fields exactly. */
   uint64_t x = 0x9e3779b97f4a7c15ull ^ (uint64_t)row;
   x ^= (uint64_t)vertex + 0x9e3779b97f4a7c15ull + (x << 6) + (x >> 2);
   x ^= (uint64_t)instance + 0x9e3779b97f4a7c15ull + (x << 6) + (x >> 2);
   x ^= x >> 30;
   x *= 0xbf58476d1ce4e5b9ull;
   x ^= x >> 27;
   x *= 0x94d049bb133111ebull;
   return x ^ (x >> 31);
}

extern "C" __global__ void
cp_vertex_reuse_census(struct cp_vertex_fetch_args vf,
                       uint64_t table_address, uint64_t unique_address,
                       uint64_t probes_address, uint64_t overflow_address,
                       uint64_t bad_lanes_address, uint32_t capacity,
                       uint32_t generation)
{
   struct cp_vertex_reuse_entry *table =
      (struct cp_vertex_reuse_entry *)(uintptr_t)table_address;
   unsigned long long *unique =
      (unsigned long long *)(uintptr_t)unique_address;
   unsigned long long *probes =
      (unsigned long long *)(uintptr_t)probes_address;
   unsigned int *overflow = (unsigned int *)(uintptr_t)overflow_address;
   unsigned int *bad_lanes = (unsigned int *)(uintptr_t)bad_lanes_address;
   const uint32_t lock = generation << 1;
   const uint32_t ready = lock | 1u;
   const uint32_t mask = capacity - 1u;

   uint64_t local_probes = 0;
   for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
        v < vf.num_verts;
        v += blockDim.x * gridDim.x) {
      uint32_t vertex, instance, row;
      if (!cp_vf_lane_ids(&vf, v, &vertex, &instance, &row)) {
         atomicAdd(bad_lanes, 1u);
         continue;
      }

      uint32_t slot = (uint32_t)cp_vertex_reuse_mix(row, vertex, instance) & mask;
      bool done = false;
      for (uint32_t p = 0; p < capacity; p++, slot = (slot + 1u) & mask) {
         local_probes++;
         struct cp_vertex_reuse_entry *e = &table[slot];
      retry_slot:
         uint32_t state = atomicAdd(&e->state, 0u);
         if (state != lock && state != ready) {
            if (atomicCAS(&e->state, state, lock) != state)
               goto retry_slot;
            e->row = row;
            e->vertex = vertex;
            e->instance = instance;
            __threadfence();
            atomicExch(&e->state, ready);
            atomicAdd(unique, 1ull);
            done = true;
            break;
         }
         uint32_t spins = 0;
         while (state == lock && spins++ < 1024u)
            state = atomicAdd(&e->state, 0u);
         if (state == lock) {
            /* Never spin forever behind a poisoned or warp-divergent owner. */
            atomicExch(overflow, 1u);
            done = true;
            break;
         }
         if (state == ready && e->row == row && e->vertex == vertex &&
             e->instance == instance) {
            done = true;
            break;
         }
      }
      if (!done)
         atomicExch(overflow, 1u);
   }
   if (local_probes)
      atomicAdd(probes, (unsigned long long)local_probes);
}
