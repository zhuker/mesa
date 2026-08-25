#version 450

/* The check pattern of PROTOCOL.md, "The frame".
 *
 *    R = (x + seq)     & 255
 *    G = (y + 2 * seq) & 255
 *    B = (x ^ y)       & 255
 *    A = 255
 *
 * plus the stamp: pixel (0,0) holds seq as u32 little-endian in its four
 * bytes, pixel (1,0) holds ~seq.  R8G8B8A8_UINT puts R in byte 0, so the
 * little-endian order is (v, v>>8, v>>16, v>>24).
 *
 * Additions only.  This function is written three times -- here, in C in
 * cpvk_interop_render.c, and in Python in the consumer.  Any change is a
 * change to all three. */

layout(location = 0) out uvec4 o_color;

layout(push_constant) uniform PC {
   uint seq;
} pc;

void main()
{
   uint x = uint(gl_FragCoord.x);
   uint y = uint(gl_FragCoord.y);
   uint seq = pc.seq;

   uvec4 c = uvec4((x + seq)      & 255u,
                   (y + 2u * seq) & 255u,
                   (x ^ y)        & 255u,
                   255u);

   if (y == 0u && x < 2u) {
      uint v = (x == 0u) ? seq : ~seq;
      c = uvec4(v & 255u, (v >> 8) & 255u, (v >> 16) & 255u, (v >> 24) & 255u);
   }

   o_color = c;
}
