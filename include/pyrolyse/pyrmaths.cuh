#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD (M_PI * 2.0 / 360.0)

typedef struct { float u, v; } Float2;
typedef struct { float r, g, b; } Float3;
typedef struct { float r, g, b, a; } Float4;
typedef struct { float m[4][4]; } Maf44;

__host__ __device__ Float4 mul_maf4_f4(const Maf44* M, const Float4* v);
__host__ __device__ Float3 f4_to_f3(Float4 f);
__host__ __device__ Float4 f3_to_f4(Float3 f, float w);
__host__ __device__ Float2 sub_f2_f(Float2 a, float b);
__host__ __device__ Float3 mul_f3(Float3 a, Float3 b);
__host__ __device__ Float3 norm_f3(Float3 f);
__host__ __device__ Float3 sub_f3(Float3 a, Float3 b);

__host__ __device__ unsigned char clamp_to_255(float v);