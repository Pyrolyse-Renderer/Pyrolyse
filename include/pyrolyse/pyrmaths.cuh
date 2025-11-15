#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846
#include "types.cuh"
#endif

#define DEG2RAD (M_PI * 2.0 / 360.0)

// TODO change customs FloatX structs to built-in structs floatx

__host__ __device__ Float4 mul_maf4_f4(const Maf44& M, Float4 v);
__host__ __device__ Float3 f4_to_f3(Float4 f);
__host__ __device__ Float4 f3_to_f4(Float3 f, float w);
__host__ __device__ Float2 sub_f2_f(Float2 a, float b);
__host__ __device__ Float2 mul_f2_f(Float2 a, float b);
__host__ __device__ Float3 mul_f3(Float3 a, Float3 b);
__host__ __device__ Float3 mul_f3_f(Float3 a, float b);
__host__ __device__ Float3 add_f3(Float3 a, Float3 b);
__host__ __device__ Float3 add_f3_f(Float3 f, float s);
__host__ __device__ Float3 norm_f3(Float3 f);
__host__ __device__ Float3 sub_f3(Float3 a, Float3 b);
__host__ __device__ Float3 cross_f3(Float3 a, Float3 b);
__host__ __device__ Float3 sign_f3(Float3 f);
__host__ __device__ Float3 div_f3_f(Float3 f, float d);
__host__ __device__ Float3 lerp_f3(Float3 start, Float3 end, float t);
__host__ __device__ float smoothstep(float edge0, float edge1, float x);
__host__ __device__ float sign_f(float f);
__host__ __device__ float dot_f3(Float3 a, Float3 b);
__host__ __device__ float maxf(float a, float b);
__host__ __device__ unsigned char clamp_to_255(float v);