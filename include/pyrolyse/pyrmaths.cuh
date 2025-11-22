#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846
#include "types.cuh"
#endif

#define DEG2RAD (M_PI * 2.0 / 360.0)

// TODO change customs FloatX structs to built-in structs floatx

__host__ __device__ float4 mul_maf4_f4(const Maf44& M, float4 v);
__host__ __device__ float3 f4_to_f3(float4 f);
__host__ __device__ float4 f3_to_f4(float3 f, float w);
__host__ __device__ float2 sub_f2_f(float2 a, float b);
__host__ __device__ float2 mul_f2_f(float2 a, float b);
__host__ __device__ float3 mul_f3(float3 a, float3 b);
__host__ __device__ float3 mul_f3_f(float3 a, float b);
__host__ __device__ float3 add_f3(float3 a, float3 b);
__host__ __device__ float3 add_f3_f(float3 f, float s);
__host__ __device__ float3 norm_f3(float3 f);
__host__ __device__ float3 sub_f3(float3 a, float3 b);
__host__ __device__ float3 cross_f3(float3 a, float3 b);
__host__ __device__ float3 sign_f3(float3 f);
__host__ __device__ float3 div_f3_f(float3 f, float d);
__host__ __device__ float3 lerp_f3(float3 start, float3 end, float t);
__host__ __device__ float smoothstep(float edge0, float edge1, float x);
__host__ __device__ float sign_f(float f);
__host__ __device__ float dot_f3(float3 a, float3 b);
__host__ __device__ float maxf(float a, float b);
__host__ __device__ unsigned char clamp_to_255(float v);