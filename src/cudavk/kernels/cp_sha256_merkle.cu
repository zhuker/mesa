/*
 * Diagnostic parallel SHA-256 Merkle kernels.
 *
 * SHA-256 compression, padding, constants, and byte order are an independent
 * implementation from FIPS PUB 180-4, sections 5 and 6. No third-party SHA
 * implementation was copied. The surrounding Merkle encoding is cudavk's.
 *
 * Exact encoding (|| means concatenation; all integers are unsigned big
 * endian; digest words use SHA-256's standard big-endian byte form):
 *
 * leaf = SHA256("CPMRKLF1"[8] || region_id:u64 || total_bytes:u64 ||
 *               leaf_index:u64 || payload_bytes:u32 || payload)
 * node = SHA256("CPMRKND1"[8] || region_id:u64 || level:u32 ||
 *               node_index:u64 || child_count:u32 || left[32] || right[32])
 *
 * Leaves are fixed 4096-byte chunks except the final leaf. Empty input has one
 * zero-byte leaf at index zero. A parent with an odd final child sets
 * child_count=1 and right to 32 zero bytes; it never duplicates or silently
 * promotes the child. Other parents set child_count=2. These fields and the
 * distinct fixed-width tags make leaf/node and unary/binary encodings
 * unambiguous. A region_id lets independent logical regions use the same
 * scheme without conflating their roots.
 */
#include "cp_sha256_merkle.h"

#if defined(CP_SHA256_MERKLE_HOST_TEST)
#define CP_SHA_DEVICE static inline
#else
#define CP_SHA_DEVICE static __device__ __forceinline__
#endif

typedef cp_sha256_merkle_u32 cp_sha_u32;
typedef cp_sha256_merkle_u64 cp_sha_u64;
typedef unsigned char cp_sha_u8;

struct cp_sha256_ctx {
   cp_sha_u32 state[8];
   cp_sha_u64 total_bytes;
   cp_sha_u32 used;
   cp_sha_u8 block[64];
};

CP_SHA_DEVICE cp_sha_u32
cp_sha_rotr(cp_sha_u32 x, cp_sha_u32 n)
{
   /* Every caller uses a FIPS rotation in [2, 25], never 0 or 32. */
   return (x >> n) | (x << (32u - n));
}

CP_SHA_DEVICE cp_sha_u32
cp_sha_load_be32(const cp_sha_u8 *p)
{
   return ((cp_sha_u32)p[0] << 24) |
          ((cp_sha_u32)p[1] << 16) |
          ((cp_sha_u32)p[2] << 8) |
          (cp_sha_u32)p[3];
}

CP_SHA_DEVICE void
cp_sha_store_be32(cp_sha_u8 *p, cp_sha_u32 x)
{
   p[0] = (cp_sha_u8)(x >> 24);
   p[1] = (cp_sha_u8)(x >> 16);
   p[2] = (cp_sha_u8)(x >> 8);
   p[3] = (cp_sha_u8)x;
}

CP_SHA_DEVICE void
cp_sha_store_be64(cp_sha_u8 *p, cp_sha_u64 x)
{
   p[0] = (cp_sha_u8)(x >> 56);
   p[1] = (cp_sha_u8)(x >> 48);
   p[2] = (cp_sha_u8)(x >> 40);
   p[3] = (cp_sha_u8)(x >> 32);
   p[4] = (cp_sha_u8)(x >> 24);
   p[5] = (cp_sha_u8)(x >> 16);
   p[6] = (cp_sha_u8)(x >> 8);
   p[7] = (cp_sha_u8)x;
}

CP_SHA_DEVICE void
cp_sha256_compress(cp_sha256_ctx *ctx, const cp_sha_u8 block[64])
{
   static const cp_sha_u32 k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
   };
   cp_sha_u32 w[64];
   cp_sha_u32 a, b, c, d, e, f, g, h;

   for (cp_sha_u32 i = 0; i < 16; i++)
      w[i] = cp_sha_load_be32(block + i * 4);
   for (cp_sha_u32 i = 16; i < 64; i++) {
      cp_sha_u32 s0 = cp_sha_rotr(w[i - 15], 7) ^
                      cp_sha_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      cp_sha_u32 s1 = cp_sha_rotr(w[i - 2], 17) ^
                      cp_sha_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
   }

   a = ctx->state[0]; b = ctx->state[1];
   c = ctx->state[2]; d = ctx->state[3];
   e = ctx->state[4]; f = ctx->state[5];
   g = ctx->state[6]; h = ctx->state[7];
   for (cp_sha_u32 i = 0; i < 64; i++) {
      cp_sha_u32 s1 = cp_sha_rotr(e, 6) ^ cp_sha_rotr(e, 11) ^
                      cp_sha_rotr(e, 25);
      cp_sha_u32 ch = (e & f) ^ ((~e) & g);
      cp_sha_u32 t1 = h + s1 + ch + k[i] + w[i];
      cp_sha_u32 s0 = cp_sha_rotr(a, 2) ^ cp_sha_rotr(a, 13) ^
                      cp_sha_rotr(a, 22);
      cp_sha_u32 maj = (a & b) ^ (a & c) ^ (b & c);
      cp_sha_u32 t2 = s0 + maj;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
   }

   ctx->state[0] += a; ctx->state[1] += b;
   ctx->state[2] += c; ctx->state[3] += d;
   ctx->state[4] += e; ctx->state[5] += f;
   ctx->state[6] += g; ctx->state[7] += h;
}

CP_SHA_DEVICE void
cp_sha256_init(cp_sha256_ctx *ctx)
{
   ctx->state[0] = 0x6a09e667u; ctx->state[1] = 0xbb67ae85u;
   ctx->state[2] = 0x3c6ef372u; ctx->state[3] = 0xa54ff53au;
   ctx->state[4] = 0x510e527fu; ctx->state[5] = 0x9b05688cu;
   ctx->state[6] = 0x1f83d9abu; ctx->state[7] = 0x5be0cd19u;
   ctx->total_bytes = 0;
   ctx->used = 0;
}

CP_SHA_DEVICE void
cp_sha256_update(cp_sha256_ctx *ctx, const cp_sha_u8 *data, cp_sha_u32 size)
{
   ctx->total_bytes += size;
   while (size != 0) {
      cp_sha_u32 available = 64u - ctx->used;
      cp_sha_u32 take = size < available ? size : available;
      for (cp_sha_u32 i = 0; i < take; i++)
         ctx->block[ctx->used + i] = data[i];
      ctx->used += take;
      data += take;
      size -= take;
      if (ctx->used == 64u) {
         cp_sha256_compress(ctx, ctx->block);
         ctx->used = 0;
      }
   }
}

CP_SHA_DEVICE void
cp_sha256_final(cp_sha256_ctx *ctx, cp_sha_u8 digest[32])
{
   cp_sha_u64 bit_count = ctx->total_bytes * 8u;
   cp_sha_u8 byte = 0x80u;
   cp_sha_u8 encoded_length[8];

   cp_sha256_update(ctx, &byte, 1);
   byte = 0;
   while (ctx->used != 56u)
      cp_sha256_update(ctx, &byte, 1);
   cp_sha_store_be64(encoded_length, bit_count);
   cp_sha256_update(ctx, encoded_length, 8);

   for (cp_sha_u32 i = 0; i < 8; i++)
      cp_sha_store_be32(digest + i * 4, ctx->state[i]);
}

CP_SHA_DEVICE void
cp_sha256_leaf_one(const struct cp_sha256_merkle_leaf_args *args,
                   cp_sha_u64 leaf_index)
{
   static const cp_sha_u8 tag[8] = {
      'C', 'P', 'M', 'R', 'K', 'L', 'F', '1'
   };
   cp_sha256_ctx ctx;
   cp_sha_u8 fields[28];
   cp_sha_u64 offset = leaf_index *
      (cp_sha_u64)CP_SHA256_MERKLE_LEAF_BYTES;
   cp_sha_u64 remaining = args->total_bytes - offset;
   cp_sha_u32 payload_bytes = remaining < CP_SHA256_MERKLE_LEAF_BYTES ?
      (cp_sha_u32)remaining : CP_SHA256_MERKLE_LEAF_BYTES;
   const cp_sha_u8 *source = (const cp_sha_u8 *)args->source;
   cp_sha_u8 *output = (cp_sha_u8 *)args->output;

   cp_sha_store_be64(fields + 0, args->region_id);
   cp_sha_store_be64(fields + 8, args->total_bytes);
   cp_sha_store_be64(fields + 16, leaf_index);
   cp_sha_store_be32(fields + 24, payload_bytes);
   cp_sha256_init(&ctx);
   cp_sha256_update(&ctx, tag, 8);
   cp_sha256_update(&ctx, fields, 28);
   if (payload_bytes != 0)
      cp_sha256_update(&ctx, source + offset, payload_bytes);
   cp_sha256_final(&ctx,
                   output + leaf_index * CP_SHA256_MERKLE_DIGEST_BYTES);
}

CP_SHA_DEVICE void
cp_sha256_node_one(const struct cp_sha256_merkle_node_args *args,
                   cp_sha_u64 node_index)
{
   static const cp_sha_u8 tag[8] = {
      'C', 'P', 'M', 'R', 'K', 'N', 'D', '1'
   };
   cp_sha256_ctx ctx;
   cp_sha_u8 fields[24];
   cp_sha_u8 zeros[CP_SHA256_MERKLE_DIGEST_BYTES];
   cp_sha_u64 left_index = node_index * 2u;
   cp_sha_u32 child_count = left_index + 1u < args->input_count ? 2u : 1u;
   const cp_sha_u8 *input = (const cp_sha_u8 *)args->input;
   cp_sha_u8 *output = (cp_sha_u8 *)args->output;

   cp_sha_store_be64(fields + 0, args->region_id);
   cp_sha_store_be32(fields + 8, args->level);
   cp_sha_store_be64(fields + 12, node_index);
   cp_sha_store_be32(fields + 20, child_count);
   for (cp_sha_u32 i = 0; i < CP_SHA256_MERKLE_DIGEST_BYTES; i++)
      zeros[i] = 0;

   cp_sha256_init(&ctx);
   cp_sha256_update(&ctx, tag, 8);
   cp_sha256_update(&ctx, fields, 24);
   cp_sha256_update(&ctx,
      input + left_index * CP_SHA256_MERKLE_DIGEST_BYTES,
      CP_SHA256_MERKLE_DIGEST_BYTES);
   cp_sha256_update(&ctx,
      child_count == 2u ?
         input + (left_index + 1u) * CP_SHA256_MERKLE_DIGEST_BYTES : zeros,
      CP_SHA256_MERKLE_DIGEST_BYTES);
   cp_sha256_final(&ctx,
      output + node_index * CP_SHA256_MERKLE_DIGEST_BYTES);
}

#if defined(CP_SHA256_MERKLE_HOST_TEST)
/* CPU hooks compile the exact device implementation with an ordinary C++
 * compiler. They are only for the hashlib oracle test and are absent in PTX. */
extern "C" void
cp_sha256_merkle_host_sha256(const cp_sha_u8 *data, cp_sha_u32 size,
                             cp_sha_u8 digest[32])
{
   cp_sha256_ctx ctx;
   cp_sha256_init(&ctx);
   cp_sha256_update(&ctx, data, size);
   cp_sha256_final(&ctx, digest);
}

extern "C" void
cp_sha256_merkle_host_leaf(struct cp_sha256_merkle_leaf_args args,
                           cp_sha_u64 leaf_index)
{
   cp_sha256_leaf_one(&args, leaf_index);
}

extern "C" void
cp_sha256_merkle_host_node(struct cp_sha256_merkle_node_args args,
                           cp_sha_u64 node_index)
{
   cp_sha256_node_one(&args, node_index);
}
#else
extern "C" __global__ void
cp_sha256_merkle_leaf_kernel(struct cp_sha256_merkle_leaf_args args)
{
   cp_sha_u64 local = (cp_sha_u64)blockIdx.x * blockDim.x + threadIdx.x;
   cp_sha_u64 leaf_count = cp_sha256_merkle_leaf_count(args.total_bytes);
   cp_sha_u64 leaf_index;

   if (local >= args.work_leaf_count ||
       local > (~(cp_sha_u64)0) - args.first_leaf)
      return;
   leaf_index = args.first_leaf + local;
   if (leaf_index >= leaf_count)
      return;
   cp_sha256_leaf_one(&args, leaf_index);
}

extern "C" __global__ void
cp_sha256_merkle_node_kernel(struct cp_sha256_merkle_node_args args)
{
   cp_sha_u64 local = (cp_sha_u64)blockIdx.x * blockDim.x + threadIdx.x;
   cp_sha_u64 parent_count = cp_sha256_merkle_parent_count(args.input_count);
   cp_sha_u64 node_index;

   if (args.level == 0 || args.reserved != 0 || args.input_count <= 1 ||
       args.input_count > (~(cp_sha_u64)0) /
                             CP_SHA256_MERKLE_DIGEST_BYTES ||
       local >= args.work_parent_count ||
       local > (~(cp_sha_u64)0) - args.first_parent)
      return;
   node_index = args.first_parent + local;
   if (node_index >= parent_count)
      return;
   cp_sha256_node_one(&args, node_index);
}
#endif
