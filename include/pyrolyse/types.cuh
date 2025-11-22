#pragma once

typedef struct { float m[4][4]; } Maf44;

typedef struct { float3 posa, posb, posc, norma, normb, normc; } Triangle;
typedef struct { float3 position, lookat; } Transform;
typedef struct { Transform t; float ncp, fov, aspect; } Camera;
typedef struct { float pwidth, pheight; Camera cam; Maf44 ltwm; } ViewParams;
typedef struct { float3 origin, dir; } Ray;
typedef struct { float3 color, emissive; float emstrength; } Material;
typedef struct { bool didHit; float distance; float3 location, normal; Material material; } Hit;
typedef struct { Triangle* triangles; int ntri; Material mat; } Mesh;
typedef struct { int tindex, ntri, mindex; } DeviceMesh;