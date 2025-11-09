#pragma once

#include "pyrmaths.cuh"

typedef struct { Float3* values; int count; } VertexGroup;
typedef struct { Float3 position, lookat; } Transform;
typedef struct { Transform t; float ncp, fov, aspect; } Camera;
typedef struct { float pwidth, pheight; Camera cam; Maf44 ltwm; } ViewParams;
typedef struct { Float3 origin, dir; } Ray;