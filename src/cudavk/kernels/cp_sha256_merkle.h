/*
 * Standalone SHA-256 Merkle ABI for cudavk diagnostic kernels.
 *
 * The CUDA translation unit is deliberately independent of renderer types so
 * it can be compiled by NVRTC and registered later without changing this ABI.
 */
#ifndef CP_SHA256_MERKLE_H
#define CP_SHA256_MERKLE_H

#if defined(__cplusplus)
extern "C" {
#endif

typedef unsigned int cp_sha256_merkle_u32;
typedef unsigned long long cp_sha256_merkle_u64;

#define CP_SHA256_MERKLE_DIGEST_BYTES 32u
#define CP_SHA256_MERKLE_LEAF_BYTES 4096u
#define CP_SHA256_MERKLE_DEFAULT_THREADS 128u
/* CUDA's portable one-dimensional grid.x limit. Larger work is split with
 * first_leaf/first_parent while keeping the same output arrays. */
#define CP_SHA256_MERKLE_MAX_GRID_X 2147483647u

/* One thread hashes one leaf. source and output are device addresses.
 *
 * A normal full-region launch sets first_leaf=0 and work_leaf_count to
 * cp_sha256_merkle_leaf_count(total_bytes). Split launches may select a
 * disjoint interval. Output is indexed by the absolute leaf index. The caller
 * must validate first_leaf + work_leaf_count without overflow and within the
 * leaf count. source may be zero only when total_bytes is zero.
 */
struct cp_sha256_merkle_leaf_args {
   cp_sha256_merkle_u64 source;
   cp_sha256_merkle_u64 output;
   cp_sha256_merkle_u64 total_bytes;
   cp_sha256_merkle_u64 region_id;
   cp_sha256_merkle_u64 first_leaf;
   cp_sha256_merkle_u64 work_leaf_count;
};

/* One thread hashes one parent. input and output must not overlap.
 * level is 1 for the parents of leaves, 2 for the next level, and so on.
 * reserved must be zero. Output is indexed by the absolute parent index.
 * input_count must be the plan-derived current level count. The caller must
 * validate first_parent + work_parent_count without overflow and within
 * parent_count(input_count), and must provide allocated, non-overlapping input
 * and output ranges whose byte sizes fit its host size type.
 */
struct cp_sha256_merkle_node_args {
   cp_sha256_merkle_u64 input;
   cp_sha256_merkle_u64 output;
   cp_sha256_merkle_u64 region_id;
   cp_sha256_merkle_u64 input_count;
   cp_sha256_merkle_u64 first_parent;
   cp_sha256_merkle_u64 work_parent_count;
   cp_sha256_merkle_u32 level;
   cp_sha256_merkle_u32 reserved;
};

/* Two equal-size digest planes are enough for every tree. Leaves go in plane
 * zero; each node launch swaps source and destination planes. root_offset is
 * the root digest's byte offset in scratch after all launches.
 *
 * node_launch_count = ceil(log2(leaf_count)); total kernel launches are one
 * leaf launch plus node_launch_count. Split grids increase the physical launch
 * count, but do not change the logical counts reported here. For a per-launch
 * capacity C=threads*MAX_GRID_X, the physical count is ceil(leaves/C) plus
 * ceil(parents_at_level/C) at every node level. Do not launch a zero-block
 * grid; an empty byte region still has one leaf and therefore one block.
 */
struct cp_sha256_merkle_plan {
   cp_sha256_merkle_u64 leaf_count;
   cp_sha256_merkle_u64 plane_bytes;
   cp_sha256_merkle_u64 scratch_bytes;
   cp_sha256_merkle_u64 root_offset;
   cp_sha256_merkle_u32 node_launch_count;
   cp_sha256_merkle_u32 kernel_launch_count;
};

#if defined(__CUDACC__)
#define CP_SHA256_MERKLE_INLINE static __host__ __device__ __forceinline__
#else
#define CP_SHA256_MERKLE_INLINE static inline
#endif

CP_SHA256_MERKLE_INLINE cp_sha256_merkle_u64
cp_sha256_merkle_leaf_count(cp_sha256_merkle_u64 total_bytes)
{
   /* Empty input is represented by one zero-length leaf. This form avoids an
    * overflow in total_bytes + LEAF_BYTES - 1. */
   return total_bytes == 0 ? 1 :
      1 + (total_bytes - 1) / (cp_sha256_merkle_u64)CP_SHA256_MERKLE_LEAF_BYTES;
}

CP_SHA256_MERKLE_INLINE cp_sha256_merkle_u64
cp_sha256_merkle_parent_count(cp_sha256_merkle_u64 child_count)
{
   return child_count / 2 + child_count % 2;
}

/* Returns 0 on success, -1 for a null output or arithmetic overflow. */
CP_SHA256_MERKLE_INLINE int
cp_sha256_merkle_make_plan(cp_sha256_merkle_u64 total_bytes,
                           struct cp_sha256_merkle_plan *plan)
{
   cp_sha256_merkle_u64 count, plane;
   cp_sha256_merkle_u32 levels = 0;

   if (!plan)
      return -1;
   count = cp_sha256_merkle_leaf_count(total_bytes);
   if (count > (~(cp_sha256_merkle_u64)0) /
               (cp_sha256_merkle_u64)CP_SHA256_MERKLE_DIGEST_BYTES)
      return -1;
   plane = count * (cp_sha256_merkle_u64)CP_SHA256_MERKLE_DIGEST_BYTES;
   if (plane > (~(cp_sha256_merkle_u64)0) / 2)
      return -1;

   plan->leaf_count = count;
   plan->plane_bytes = plane;
   plan->scratch_bytes = plane * 2;
   while (count > 1) {
      count = cp_sha256_merkle_parent_count(count);
      levels++;
   }
   plan->node_launch_count = levels;
   plan->kernel_launch_count = levels + 1;
   plan->root_offset = (levels & 1u) ? plane : 0;
   return 0;
}

/* Compute a portable 1-D launch size. Returns -1 if threads is invalid or the
 * work does not fit one grid; use split launches in that case. work=0 is valid
 * and reports zero blocks. */
CP_SHA256_MERKLE_INLINE int
cp_sha256_merkle_launch_blocks(cp_sha256_merkle_u64 work,
                               cp_sha256_merkle_u32 threads,
                               cp_sha256_merkle_u32 *blocks)
{
   cp_sha256_merkle_u64 n;
   if (!blocks || threads == 0 || threads > 1024u)
      return -1;
   n = work / threads + (work % threads != 0);
   if (n > (cp_sha256_merkle_u64)CP_SHA256_MERKLE_MAX_GRID_X)
      return -1;
   *blocks = (cp_sha256_merkle_u32)n;
   return 0;
}

#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
static_assert(sizeof(cp_sha256_merkle_u32) == 4, "Merkle u32 ABI");
static_assert(sizeof(cp_sha256_merkle_u64) == 8, "Merkle u64 ABI");
static_assert(sizeof(struct cp_sha256_merkle_leaf_args) == 48,
              "Merkle leaf argument ABI");
static_assert(sizeof(struct cp_sha256_merkle_node_args) == 56,
              "Merkle node argument ABI");
static_assert(sizeof(struct cp_sha256_merkle_plan) == 40,
              "Merkle plan ABI");
#if !defined(__CUDACC_RTC__)
static_assert(__builtin_offsetof(struct cp_sha256_merkle_leaf_args, source) == 0,
              "Merkle leaf source ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_leaf_args, output) == 8,
              "Merkle leaf output ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_leaf_args, total_bytes) == 16,
              "Merkle leaf length ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_leaf_args, region_id) == 24,
              "Merkle leaf region ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_leaf_args, first_leaf) == 32,
              "Merkle leaf index ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_leaf_args,
                                 work_leaf_count) == 40,
              "Merkle leaf work ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_node_args, input) == 0,
              "Merkle node input ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_node_args, output) == 8,
              "Merkle node output ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_node_args, region_id) == 16,
              "Merkle node region ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_node_args, input_count) == 24,
              "Merkle node count ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_node_args, first_parent) == 32,
              "Merkle node index ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_node_args,
                                 work_parent_count) == 40,
              "Merkle node work ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_node_args, level) == 48,
              "Merkle node level ABI");
static_assert(__builtin_offsetof(struct cp_sha256_merkle_node_args, reserved) == 52,
              "Merkle node reserved ABI");
#endif
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(cp_sha256_merkle_u32) == 4, "Merkle u32 ABI");
_Static_assert(sizeof(cp_sha256_merkle_u64) == 8, "Merkle u64 ABI");
_Static_assert(sizeof(struct cp_sha256_merkle_leaf_args) == 48,
               "Merkle leaf argument ABI");
_Static_assert(sizeof(struct cp_sha256_merkle_node_args) == 56,
               "Merkle node argument ABI");
_Static_assert(sizeof(struct cp_sha256_merkle_plan) == 40,
               "Merkle plan ABI");
#endif

#if defined(__CUDACC__) && !defined(CP_SHA256_MERKLE_HOST_TEST)
/* Each entry point is independently launchable on the caller's CUDA stream. */
extern "C" __global__ void
cp_sha256_merkle_leaf_kernel(struct cp_sha256_merkle_leaf_args args);
extern "C" __global__ void
cp_sha256_merkle_node_kernel(struct cp_sha256_merkle_node_args args);
#endif

#undef CP_SHA256_MERKLE_INLINE

#endif /* CP_SHA256_MERKLE_H */
