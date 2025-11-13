#pragma once
#include "types.cuh"

Mesh* load_models(int& out_mesh_count, int& out_triangle_count);
void cook_buffers(const Mesh* meshes, Float3* out_triangles, Float3* out_materials, DeviceMesh* out_meshes, int meshcount);
