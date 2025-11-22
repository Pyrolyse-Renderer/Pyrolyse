# include "pyrolyse/pyrmaths.cuh"

__host__ __device__ float4 mul_maf4_f4(const Maf44& M, const float4 v)
{
    float4 result;
    result.x = M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z + M.m[0][3] * v.w;
    result.y = M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z + M.m[1][3] * v.w;
    result.z = M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z + M.m[2][3] * v.w;
    result.w = M.m[3][0] * v.x + M.m[3][1] * v.y + M.m[3][2] * v.z + M.m[3][3] * v.w;
    return result;
}

__host__ __device__ float3 f4_to_f3(const float4 f)
{
    return {f.x, f.y, f.z};
}

__host__ __device__ float4 f3_to_f4(const float3 f, const float w)
{
    return {f.x, f.y, f.z, w};
}

__host__ __device__ float2 sub_f2_f(const float2 a, const float b)
{
    return {a.x - b, a.y - b};
}

__host__ __device__ float2 mul_f2_f(const float2 a, const float b)
{
    return {a.x * b, a.y * b};
}

__host__ __device__ float3 mul_f3(const float3 a, const float3 b)
{
    return { a.x* b.x, a.y* b.y, a.z* b.z };
}

__host__ __device__ float3 mul_f3_f(const float3 a, const float b)
{
    return { a.x * b, a.y * b, a.z * b };
}

__host__ __device__ float3 add_f3(const float3 a, const float3 b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__host__ __device__ float3 add_f3_f(const float3 f, const float s)
{
    return { f.x + s, f.y + s, f.z + s };
}

__host__ __device__ float3 norm_f3(const float3 f)
{
    const float length = sqrtf(f.x * f.x + f.y * f.y + f.z * f.z);
    if (length == 0.0f) return {0.0f, 0.0f, 0.0f};
    return {f.x / length, f.y / length, f.z / length};
}

__host__ __device__ float3 sub_f3(const float3 a, const float3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ float3 cross_f3(const float3 a, const float3 b)
{
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

__host__ __device__ float3 sign_f3(const float3 f)
{
    float3 result;
    result.x = static_cast<float>((f.x > 0.0f) - (f.x < 0.0f));
    result.y = static_cast<float>((f.y > 0.0f) - (f.y < 0.0f));
    result.z = static_cast<float>((f.z > 0.0f) - (f.z < 0.0f));
    return result;
}

__host__ __device__ float3 div_f3_f(const float3 f, const float d)
{
    return float3{f.x / d, f.y / d, f.z / d};
}

__host__ __device__ float3 lerp_f3(const float3 start, const float3 end, const float t)
{
    float3 result;
    result.x = start.x + t * (end.x - start.x);
    result.y = start.y + t * (end.y - start.y);
    result.z = start.z + t * (end.z - start.z);
    return result;
}

__host__ __device__ float smoothstep(const float edge0, const float edge1, const float x)
{
    float t = (x - edge0) / (edge1 - edge0);
    t = fmax(0.0f, fmin(1.0f, t));
    return t * t * (3.0f - 2.0f * t);
}

__host__ __device__ float sign_f(const float f)
{
    return static_cast<float>((f > 0.0f) - (f < 0.0f));
}

__host__ __device__ float dot_f3(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float maxf(const float a, const float b) {
    return a > b ? a : b;
}

__host__ __device__ unsigned char clamp_to_255(const float v)
{
    if (v < 0.0f) return 0;
    if (v > 1.0f) return 255;
    return static_cast<unsigned char>(v * 255.0f);
}