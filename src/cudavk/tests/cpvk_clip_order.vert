#version 450
layout(location=0) flat out vec4 colour;
const vec2 p[6] = vec2[6](vec2(-2,-.75),vec2(.75,-.75),vec2(0,.75),
                         vec2(-.5,-.5),vec2(.5,-.5),vec2(0,.5));
void main() {
 gl_Position=vec4(p[gl_VertexIndex],.5,1);
 colour=gl_VertexIndex<3 ? vec4(1,0,0,1) : vec4(0,1,0,1);
}
