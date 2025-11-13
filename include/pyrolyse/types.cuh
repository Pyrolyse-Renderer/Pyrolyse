#pragma once

#include "pyrmaths.cuh"

typedef struct { Float3 posa, posb, posc, norma, normb, normc; } Triangle;
typedef struct { Float3 position, lookat; } Transform;
typedef struct { Transform t; float ncp, fov, aspect; } Camera;
typedef struct { float pwidth, pheight; Camera cam; Maf44 ltwm; } ViewParams;
typedef struct { Float3 origin, dir; } Ray;
typedef struct { Float3 color; } Material;
typedef struct { bool didHit; float distance; Float3 location, normal; Material material; } Hit;
typedef struct { Triangle* triangles; int ntri; Material mat; } Mesh;
typedef struct { int tindex, ntri, mindex; } DeviceMesh;