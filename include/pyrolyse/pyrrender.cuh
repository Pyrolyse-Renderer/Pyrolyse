#pragma once

#include "types.cuh"

Maf44 make_cam_matrix(Float3 pos, Float3 target);
ViewParams cook_view_params(const Transform& cameraTransform);

__global__ void render(Float3* out, const Float3* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, ViewParams vp, int nmesh, int imageWidth, int imageHeight);

__device__ Float3 frag(Float2 uv, const ViewParams* vp, const Float3* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh);
__device__ Float3 trace(Ray ray, const Float3* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh);
__device__ __forceinline__ Hit hit_triangle(const Ray& ray, const Triangle& triangle);
__device__ Hit ray_collision(const Ray& ray, const Float3* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh);
__device__ Float3 rand_direction();
__device__ Float3 rand_angular_direction(Float3 normal);