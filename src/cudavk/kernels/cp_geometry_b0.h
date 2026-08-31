#ifndef CP_GEOMETRY_B0_H
#define CP_GEOMETRY_B0_H

#include "cp_sha256_merkle.h"

#define CP_GEOM_B0_MAGIC 0x30424743u /* CGB0 in little-endian memory */
#define CP_GEOM_B0_VERSION 1u
#define CP_GEOM_B0_CAPACITY 131072u
#define CP_GEOM_B0_RECORD_BYTES 128u
#define CP_GEOM_B0_STATUS_COMPLETE 1u
#define CP_GEOM_B0_STATUS_QUEUE_REJECT 2u

struct cp_geom_b0_record {
   cp_sha256_merkle_u32 magic;
   unsigned short version;
   unsigned short status;
   cp_sha256_merkle_u64 salt;
   cp_sha256_merkle_u64 submit_ordinal;
   cp_sha256_merkle_u32 submit_occurrence;
   cp_sha256_merkle_u32 unit_ordinal;
   cp_sha256_merkle_u64 byte_length;
   cp_sha256_merkle_u64 structural_signature;
   cp_sha256_merkle_u32 width;
   cp_sha256_merkle_u32 height;
   unsigned short samples;
   unsigned short pass;
   unsigned char path;
   unsigned char predecessor;
   unsigned char retry;
   unsigned char reserved0;
   cp_sha256_merkle_u32 stage2_blocks;
   cp_sha256_merkle_u32 triangles;
   cp_sha256_merkle_u32 nontrivial_count;
   cp_sha256_merkle_u32 huge_count;
   cp_sha256_merkle_u32 setup_count;
   cp_sha256_merkle_u32 queue_status;
   unsigned char sha256[CP_SHA256_MERKLE_DIGEST_BYTES];
   unsigned char reserved1[8];
};

struct cp_geom_b0_final_args {
   cp_sha256_merkle_u64 root;
   cp_sha256_merkle_u64 root2;
   cp_sha256_merkle_u64 nontrivial_count;
   cp_sha256_merkle_u64 huge_count;
   cp_sha256_merkle_u64 setup_count;
   cp_sha256_merkle_u64 output;
   cp_sha256_merkle_u64 output2;
   struct cp_geom_b0_record record;
};

#if defined(__cplusplus)
static_assert(sizeof(struct cp_geom_b0_record) == CP_GEOM_B0_RECORD_BYTES,
              "geometry B0 record ABI");
static_assert(sizeof(struct cp_geom_b0_final_args) == 184,
              "geometry B0 final ABI");
#else
_Static_assert(sizeof(struct cp_geom_b0_record) == CP_GEOM_B0_RECORD_BYTES,
               "geometry B0 record ABI");
_Static_assert(sizeof(struct cp_geom_b0_final_args) == 184,
               "geometry B0 final ABI");
#endif

#if defined(__CUDACC__)
extern "C" __global__ void
cp_geom_b0_final_kernel(struct cp_geom_b0_final_args args);
#endif
#endif
