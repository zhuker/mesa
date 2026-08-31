/* Diagnostic-only indexed/post-transform vertex reuse census ABI. */
#ifndef CP_VERTEX_REUSE_H
#define CP_VERTEX_REUSE_H

#include <stdint.h>
#include <cuda.h>
#include <stddef.h>

#define CP_VERTEX_REUSE_VERSION 1u
#define CP_VERTEX_REUSE_CAPACITY 262144u
#define CP_VERTEX_REUSE_BANKS 9u
#define CP_VERTEX_REUSE_MAGIC 0x56524555u /* "VREU" */

enum cp_vertex_reuse_status {
   CP_VERTEX_REUSE_LAUNCHED = 0,
   CP_VERTEX_REUSE_EXCLUDED_SETUP = 1,
   CP_VERTEX_REUSE_EXCLUDED_ALLOC = 2,
   CP_VERTEX_REUSE_EXCLUDED_LAUNCH = 3,
};

struct cp_vertex_reuse_result {
   uint64_t unique;
   uint64_t probes;
   uint32_t overflow;
   uint32_t bad_lanes;
};

struct cp_vertex_reuse_meta {
   uint64_t submit_ordinal;
   uint64_t batch_ordinal;
   uint64_t total;
   uint32_t bank;
   uint32_t generation;
   uint32_t table_capacity;
   uint8_t fused;
   uint8_t fetch_runs;
   uint8_t indexed;
   uint8_t writes_memory;
   uint8_t semantic_metadata_complete;
   uint8_t status;
   uint8_t reserved[2];
};

struct cp_vertex_reuse_table_alloc {
   CUdeviceptr ptr;
   struct cp_vertex_reuse_table_alloc *next;
};

struct cp_vertex_reuse_bank {
   CUdeviceptr table;
   uint32_t capacity;
   uint32_t generation;
   struct cp_vertex_reuse_table_alloc *allocations;
};

#endif
