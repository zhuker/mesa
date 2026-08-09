#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;
/* Per-instance, fetched with a divisor of 1. */
layout(location = 3) in vec4 in_offset;
layout(location = 4) in vec4 in_tint;

layout(binding = 0) uniform Matrices {
   mat4 mvp;
} u;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec4 out_tint;
layout(location = 2) out vec3 out_normal;

void main()
{
   vec3 world = in_pos + in_offset.xyz;
   gl_Position = u.mvp * vec4(world, 1.0);
   out_uv = in_uv;
   out_normal = in_normal;
   /* Vary the tint per instance so an instancing bug is visible rather than
    * merely wrong-looking. */
   out_tint = in_tint * (0.6 + 0.4 * float(gl_InstanceIndex % 3));
}
