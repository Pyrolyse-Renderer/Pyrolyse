# include "pyrolyse/pyrmaths.cuh"

__host__ __device__ Float4 mul_maf4_f4(const Maf44& M, const Float4 v)
{
    Float4 result;
    result.r = M.m[0][0] * v.r + M.m[0][1] * v.g + M.m[0][2] * v.b + M.m[0][3] * v.a;
    result.g = M.m[1][0] * v.r + M.m[1][1] * v.g + M.m[1][2] * v.b + M.m[1][3] * v.a;
    result.b = M.m[2][0] * v.r + M.m[2][1] * v.g + M.m[2][2] * v.b + M.m[2][3] * v.a;
    result.a = M.m[3][0] * v.r + M.m[3][1] * v.g + M.m[3][2] * v.b + M.m[3][3] * v.a;
    return result;
}

__host__ __device__ Float3 f4_to_f3(const Float4 f)
{
    return {f.r, f.g, f.b};
}

__host__ __device__ Float4 f3_to_f4(const Float3 f, const float w)
{
    return {f.r, f.g, f.b, w};
}

__host__ __device__ Float2 sub_f2_f(const Float2 a, const float b)
{
    return {a.u - b, a.v - b};
}

__host__ __device__ Float3 mul_f3(const Float3 a, const Float3 b)
{
    return { a.r* b.r, a.g* b.g, a.b* b.b };
}

__host__ __device__ Float3 mul_f3_f(const Float3 a, const float b)
{
    return { a.r * b, a.g * b, a.b * b };
}

__host__ __device__ Float3 add_f3(const Float3 a, const Float3 b)
{
    return { a.r + b.r, a.g + b.g, a.b + b.b };
}

__host__ __device__ Float3 norm_f3(const Float3 f)
{
    const float length = sqrtf(f.r * f.r + f.g * f.g + f.b * f.b);
    if (length == 0.0f) return {0.0f, 0.0f, 0.0f};
    return {f.r / length, f.g / length, f.b / length};
}

__host__ __device__ Float3 sub_f3(const Float3 a, const Float3 b)
{
    return {a.r - b.r, a.g - b.g, a.b - b.b};
}

__host__ __device__ Float3 cross_f3(const Float3 a, const Float3 b)
{
    Float3 result;
    result.r = a.g * b.b - a.b * b.g;
    result.g = a.b * b.r - a.r * b.b;
    result.b = a.r * b.g - a.g * b.r;
    return result;
}

__host__ __device__ Float3 sign_f3(const Float3 f)
{
    Float3 result;
    result.r = static_cast<float>((f.r > 0.0f) - (f.r < 0.0f));
    result.g = static_cast<float>((f.g > 0.0f) - (f.g < 0.0f));
    result.b = static_cast<float>((f.b > 0.0f) - (f.b < 0.0f));
    return result;
}

__host__ __device__ float sign_f(const float f)
{
    return static_cast<float>((f > 0.0f) - (f < 0.0f));
}

__host__ __device__ float dot_f3(const Float3 a, const Float3 b) {
    return a.r * b.r + a.g * b.g + a.b * b.b;
}

__host__ __device__ unsigned char clamp_to_255(const float v)
{
    if (v < 0.0f) return 0;
    if (v > 1.0f) return 255;
    return static_cast<unsigned char>(v * 255.0f);
}
