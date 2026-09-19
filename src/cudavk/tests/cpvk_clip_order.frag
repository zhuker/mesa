#version 450
layout(location=0) flat in vec4 colour;
layout(location=0) out vec4 o;
void main() { o=colour; }
