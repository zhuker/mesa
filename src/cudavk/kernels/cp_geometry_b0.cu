/* Finalize one diagnostic geometry B0 record on the same stream as its hash. */
#include "cp_geometry_b0.h"

extern "C" __global__ void
cp_geom_b0_final_kernel(struct cp_geom_b0_final_args args)
{
   if (blockIdx.x || threadIdx.x)
      return;

   const unsigned char *src = (const unsigned char *)&args.record;
   unsigned char *dst = (unsigned char *)args.output;
   for (unsigned i = 0; i < sizeof(args.record); i++)
      dst[i] = src[i];

   struct cp_geom_b0_record *record =
      (struct cp_geom_b0_record *)args.output;
   const cp_sha256_merkle_u32 *nontrivial =
      (const cp_sha256_merkle_u32 *)args.nontrivial_count;
   const cp_sha256_merkle_u32 *huge =
      (const cp_sha256_merkle_u32 *)args.huge_count;
   const cp_sha256_merkle_u32 *setup =
      (const cp_sha256_merkle_u32 *)args.setup_count;
   record->nontrivial_count = *nontrivial;
   record->huge_count = *huge;
   record->setup_count = *setup;
   record->queue_status = (*nontrivial > 1000000u ? 1u : 0u) |
                          (*huge > 2000000u ? 2u : 0u) |
                          (*setup > 1024u ? 4u : 0u);

   const unsigned char *root = (const unsigned char *)args.root;
   for (unsigned i = 0; i < CP_SHA256_MERKLE_DIGEST_BYTES; i++)
      record->sha256[i] = root[i];
   if (args.root2 && args.output2) {
      const unsigned char *root2 = (const unsigned char *)args.root2;
      unsigned char *output2 = (unsigned char *)args.output2;
      for (unsigned i = 0; i < CP_SHA256_MERKLE_DIGEST_BYTES; i++)
         output2[i] = root2[i];
   }

   /* COMPLETE (or the classified queue refusal) is the publication store.
    * It is deliberately last, after both roots and every record field. */
   record->status = record->queue_status ? CP_GEOM_B0_STATUS_QUEUE_REJECT :
                                           CP_GEOM_B0_STATUS_COMPLETE;
}
