#include "pyrolyse/pyrrender.cuh"

#include "pyrolyse/pyrmaths.cuh"
#include "pyrolyse/pyrutils.cuh"

#include <curand_kernel.h>

Maf44 make_cam_matrix(const float3 pos, const float3 target)
{
    constexpr float3 upTemp = {0, -1, 0};

    const auto [fr, fg, fb] = norm_f3(sub_f3(target, pos));
    const auto [rr, rg, rb] = norm_f3({
        upTemp.y * fb - upTemp.z * fg,
        upTemp.z * fr - upTemp.x * fb,
        upTemp.x * fg - upTemp.y * fr
    });
    const float3 u = {
        fg * rb - fb * rg,
        fb * rr - fr * rb,
        fr * rg - fg * rr
    };

    const Maf44 M = {
        {
            {rr, u.x, fr, pos.x},
            {rg, u.y, fg, pos.y},
            {rb, u.z, fb, pos.z},
            {0, 0, 0, 1}
        }
    };

    return M;
}

ViewParams cook_view_params(const Transform& cameraTransform)
{
    const PyrConfig& config = get_config();
    const Camera camera = {
        cameraTransform, 1.0f, config.camera_fov,
        static_cast<float>(config.image_width) / static_cast<float>(config.image_height)
    };
    const Maf44 localToWorldMatrix = make_cam_matrix(cameraTransform.position, cameraTransform.lookat);
    const float planeHeight = camera.ncp * static_cast<float>(tan(camera.fov * 0.5f * DEG2RAD)) * 2.0f;
    const float planeWidth = planeHeight * camera.aspect;
    const ViewParams viewParams = {planeWidth, planeHeight, camera, localToWorldMatrix};
    return viewParams;
};

__device__ float3 frag(const float2 uv, const ViewParams* vp, const Material* materialBuffer,
                       const float3* triangleBuffer, const DeviceMesh* meshBuffer, const int nmesh, const long seed,
                       const int rayPerPixel, const int maxBounces, const int imageWidth)
{
    const auto [u, v] = uv;
    const float3 vpl = mul_f3({u, v, 1.0f}, {vp->pwidth, vp->pheight, vp->cam.ncp});
    const float4 vpl4 = f3_to_f4(vpl, 1.0f);
    const float3 vpw = f4_to_f3(mul_maf4_f4(vp->ltwm, vpl4));
    const float3 origin = vp->cam.t.position;
    const float3 camRight = {vp->ltwm.m[0][0], vp->ltwm.m[1][0], vp->ltwm.m[2][0]};
    const float3 camUp = {vp->ltwm.m[0][1], vp->ltwm.m[1][1], vp->ltwm.m[2][1]};

    float3 total = {.0f, .0f, .0f};

    for (int i = 0; i < rayPerPixel; i++)
    {
        const auto [jitterx, jittery] = mul_f2_f(rand_circle_point(seed + i), 3.0f / static_cast<float>(imageWidth));
        const float3 jvp = add_f3(add_f3(vpw, mul_f3_f(camRight, jitterx)), mul_f3_f(camUp, jittery));
        const Ray ray = {.origin = origin, .dir = norm_f3(sub_f3(jvp, origin))};
        total = add_f3(trace(ray, materialBuffer, triangleBuffer, meshBuffer, nmesh, seed + i, maxBounces), total);
    }

    return div_f3_f(total, static_cast<float>(rayPerPixel));
}

__device__ float3 trace(const Ray& ray, const Material* materialBuffer, const float3* triangleBuffer,
                        const DeviceMesh* meshBuffer, const int nmesh, const long seed, const int maxBounces)
{
    float3 light = {.0f, .0f, .0f};
    float3 col = {1.0f, 1.0f, 1.0f};
    Ray traced = ray;
    for (int i = 0; i < maxBounces; i++)
    {
        if (const Hit result = ray_collision(traced, materialBuffer, triangleBuffer, meshBuffer, nmesh); result.didHit)
        {
            traced.origin = result.location;
            traced.dir = norm_f3(add_f3(result.normal, rand_direction(seed + 7626 * i)));
            const auto [color, emissive, emstrength] = result.material;
            const float3 emitted = mul_f3_f(emissive, emstrength);
            light = add_f3(mul_f3(emitted, col), light);
            col = mul_f3(col, color);
        }
        else
        {
            light = mul_f3(add_f3(light, environmental_light(traced)), col);
            break;
        }
    }
    return light;
}

__global__ void render(float3* out, const Material* materialBuffer, const float3* triangleBuffer,
                       const DeviceMesh* meshBuffer, const ViewParams vp, const int nmesh, const long seed,
                       const int imageWidth, const int imageHeight, const int rayPerPixel, const int maxBounces)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imageWidth || y >= imageHeight) return;

    const unsigned int idx = y * imageWidth + x;
    float2 uv;
    uv.x = static_cast<float>(x) / static_cast<float>(imageWidth - 1);
    uv.y = static_cast<float>(y) / static_cast<float>(imageHeight - 1);

    const int2 numpixels = {imageWidth, imageHeight};
    const int2 pixcoord = {
        static_cast<int>(uv.x * static_cast<float>(numpixels.x)),
        static_cast<int>(uv.y * static_cast<float>(numpixels.y))
    };
    const long pixelindex = pixcoord.y * numpixels.x + pixcoord.x;

    out[idx] = frag(sub_f2_f(uv, .5f), &vp, materialBuffer, triangleBuffer, meshBuffer, nmesh, pixelindex * seed,
                    rayPerPixel, maxBounces, imageWidth);
}

__device__ Hit ray_collision(const Ray& ray, const Material* materialBuffer, const float3* triangleBuffer,
                             const DeviceMesh* meshBuffer, const int nmesh)
{
    Hit closestHit = {
        false,
        FLT_MAX,
        {.0f, .0f, .0f},
        {.0f, .0f, .0f},
        {{.0, 0.0, .0}}
    };

    for (int i = 0; i < nmesh; i++)
    {
        const auto [tindex, ntri, mindex] = meshBuffer[i];
        for (int tr = tindex; tr < tindex + ntri; tr++)
        {
            const auto triangle = Triangle{
                triangleBuffer[tr * 6],
                triangleBuffer[tr * 6 + 1],
                triangleBuffer[tr * 6 + 2],
                triangleBuffer[tr * 6 + 3],
                triangleBuffer[tr * 6 + 4],
                triangleBuffer[tr * 6 + 5]
            };

            if (const Hit hit = hit_triangle(ray, triangle); hit.didHit && hit.distance < closestHit.distance)
            {
                closestHit = hit;
                closestHit.material = materialBuffer[mindex];
            }
        }
    }
    return closestHit;
}

__device__ __forceinline__ Hit hit_triangle(const Ray& ray, const Triangle& triangle)
{
    constexpr Hit nullhit = {
        false,
        FLT_MAX,
        {0, 0, 0},
        {0, 0, 0},
        {
            {0.0f, 0.0f, 0.0f},
            {.0f, .0f, .0f},
            .0f
        }
    };
    const float3 edgeAB = sub_f3(triangle.posb, triangle.posa);
    const float3 edgeAC = sub_f3(triangle.posc, triangle.posa);
    const float3 normal = cross_f3(edgeAB, edgeAC);
    const float3 ao = sub_f3(ray.origin, triangle.posa);
    const float3 dao = cross_f3(ao, ray.dir);

    const float det = -dot_f3(ray.dir, normal);

    if (det < 1e-6f)
    {
        return nullhit;
    }

    const float invdet = 1.0f / det;
    const float dst = dot_f3(ao, normal) * invdet;

    if (dst < 0.0f)
    {
        return nullhit;
    }

    const float u = dot_f3(edgeAC, dao) * invdet;
    const float v = -dot_f3(edgeAB, dao) * invdet;
    const float w = 1.0f - u - v;

    if (u < 0.0f || v < 0.0f || w < 0.0f)
    {
        return nullhit;
    }

    const float3 tloc = add_f3(ray.origin, mul_f3_f(ray.dir, dst));
    const float3 tnorm = norm_f3(add_f3(add_f3(mul_f3_f(triangle.normb, u), mul_f3_f(triangle.normc, v)),
                                        mul_f3_f(triangle.norma, w)));
    return Hit{
        true,
        dst,
        tloc,
        tnorm,
        {{0.0f, 0.0f, 0.0f}, {.0f, .0f, .0f}, .0f}
    };
}

__device__ float rand_value(unsigned int state)
{
    state = state * 747796405 + 2891336453;
    const unsigned int result = (state >> ((state >> 28) + 4) ^ state) * 277803737;
    return static_cast<float>(result) / 4294967295.0f;
}

__device__ float rand_value_normal(const unsigned int state)
{
    const float theta = 2.0f * static_cast<float>(M_PI) * rand_value(state);
    const float rho = sqrtf(2 - log(rand_value(state)));
    return rho * cosf(theta);
}

__device__ float3 rand_direction(const unsigned int state)
{
    const float x = rand_value_normal(state);
    const float y = rand_value_normal(2 * state);
    const float z = rand_value_normal(6876 * state);
    return norm_f3({x, y, z});
}

__device__ float3 rand_normal_direction(const float3 normal, const unsigned int state)
{
    const float3 dir = rand_direction(state);
    return mul_f3_f(dir, sign_f(dot_f3(normal, dir)));
}

__device__ float2 rand_circle_point(const unsigned int state)
{
    const float angle = rand_value(state) * 2.0f * static_cast<float>(M_PI);
    const float2 pointOnCircle = {cos(angle), sin(angle)};
    return mul_f2_f(pointOnCircle, sqrtf(rand_value(state)));
}

__device__ float3 environmental_light(const Ray& ray)
{
    constexpr float3 HorizonColor = {.596f, .737f, .886f};
    constexpr float3 ZenithColor = {.956f, .937f, .823f};
    constexpr float3 GroundColor = {.1f, .1f, .3f};
    constexpr float3 SunLightDirection = {.8f, .5f, -.1f};
    constexpr float SunIntensity = 1.0f;
    constexpr float SunFocus = 5.0f;

    const float skyT = pow(smoothstep(.0f, .4f, ray.dir.y), .185f);
    const float3 sky = lerp_f3(ZenithColor, HorizonColor, skyT);
    const float sun = pow(maxf(0, dot_f3(ray.dir, mul_f3_f(SunLightDirection, -1.0f))), SunFocus) * SunIntensity;

    const float groundT = smoothstep(.01f, .0f, ray.dir.y);
    const float sunMask = groundT <= 0;
    return add_f3_f(lerp_f3(sky, GroundColor, groundT), sun * sunMask);
}
