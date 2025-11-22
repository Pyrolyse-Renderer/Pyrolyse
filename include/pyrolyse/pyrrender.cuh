#pragma once

#include "types.cuh"

Maf44 make_cam_matrix(float3 pos, float3 target);
ViewParams cook_view_params(const Transform& cameraTransform);

__global__ void render(float3* out, const Material* materialBuffer, const float3* triangleBuffer, const DeviceMesh* meshBuffer, ViewParams vp, int nmesh, long seed, int imageWidth, int imageHeight, int rayPerPixel, int maxBounces);

__device__ float3 frag(float2 uv, const ViewParams* vp, const Material* materialBuffer, const float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh, long seed, int rayPerPixel, int maxBounces, int imageWidth);
__device__ float3 trace(const Ray& ray, const Material* materialBuffer, const float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh, long seed, int maxBounces);
__device__ __forceinline__ Hit hit_triangle(const Ray& ray, const Triangle& triangle);
__device__ Hit ray_collision(const Ray& ray, const Material* materialBuffer, const float3* triangleBuffer, const DeviceMesh* meshBuffer, int nmesh);
__device__ float3 rand_direction(unsigned int state);
__device__ float rand_value_normal(unsigned int state);
__device__ float rand_value(unsigned int state);
__device__ float3 rand_normal_direction(float3 normal, unsigned int state);
__device__ float2 rand_circle_point(unsigned int state);
__device__ float3 environmental_light(const Ray& ray);