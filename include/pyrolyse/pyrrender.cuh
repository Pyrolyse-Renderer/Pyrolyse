#pragma once

#include "types.cuh"

Maf44 make_cam_matrix(Float3 pos, Float3 target);
ViewParams cook_view_params(const Transform& cameraTransform);

__global__ void render(Float3* out, const Material* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, ViewParams vp, int nmesh, long seed, int imageWidth, int imageHeight, int rayPerPixel, int maxBounces);

__device__ Float3 frag(Float2 uv, const ViewParams* vp, const Material* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh, long seed, int rayPerPixel, int maxBounces, int imageWidth);
__device__ Float3 trace(const Ray& ray, const Material* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh, long seed, int maxBounces);
__device__ __forceinline__ Hit hit_triangle(const Ray& ray, const Triangle& triangle);
__device__ Hit ray_collision(const Ray& ray, const Material* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh);
__device__ Float3 rand_direction(unsigned int state);
__device__ float rand_value_normal(unsigned int state);
__device__ float rand_value(unsigned int state);
__device__ Float3 rand_normal_direction(Float3 normal, unsigned int state);
__device__ Float2 rand_circle_point(unsigned int state);
__device__ Float3 environmental_light(const Ray& ray);