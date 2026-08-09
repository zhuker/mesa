#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_tint;

layout(binding = 1) uniform sampler2D tex;

layout(location = 0) out vec4 out_color;

void main()
{
   out_color = texture(tex, in_uv) * in_tint;
}
