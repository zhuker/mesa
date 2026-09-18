#pragma once
#include <optix.h>
struct Vertex { float x,y,z; };
struct Params {
    OptixTraversableHandle handle;
    const Vertex* vertices;
    unsigned* output;
    unsigned width, height;
};
