#version 450

/* Fullscreen triangle from gl_VertexIndex alone.  No vertex buffer, no
 * vertex input state, no attributes.  Vertex 0 -> (-1,-1), 1 -> (3,-1),
 * 2 -> (-1,3); the triangle covers the whole clip rectangle exactly once. */

void main()
{
   vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
