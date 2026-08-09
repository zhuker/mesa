#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_tint;
layout(location = 2) in vec3 in_normal;

layout(binding = 1) uniform sampler2D tex;

layout(location = 0) out vec4 out_color;

/*
 * Diffuse lighting over a texture, which is about as much shading as a simple
 * blocky game does. It is here mainly to exercise the shader maths a real
 * fragment shader uses — normalize, dot, max, pow, clamp — none of which the
 * simpler test shaders reach.
 */
void main()
{
   const vec3 light_dir = normalize(vec3(0.4, 0.8, 0.45));

   vec3 n = normalize(in_normal);
   float diffuse = max(dot(n, light_dir), 0.0);
   float ambient = 0.25;

   /* A little specular, for the pow() and reflect() paths. */
   vec3 view_dir = vec3(0.0, 0.0, 1.0);
   float spec = pow(max(dot(reflect(-light_dir, n), view_dir), 0.0), 16.0);

   vec4 albedo = texture(tex, in_uv) * in_tint;
   vec3 lit = albedo.rgb * (ambient + diffuse) + vec3(spec * 0.3);

   out_color = vec4(clamp(lit, 0.0, 1.0), albedo.a);
}
