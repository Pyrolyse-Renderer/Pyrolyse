#include "pyrolyse/pyrrender.cuh"

#include "pyrolyse/pyrutils.cuh"

Maf44 make_cam_matrix(const Float3 pos, const Float3 target)
{
    constexpr Float3 upTemp = {0, 1, 0};

    const auto [fr, fg, fb] = norm_f3(sub_f3(target, pos));
    const auto [rr, rg, rb] = norm_f3({
        upTemp.g * fb - upTemp.b * fg,
        upTemp.b * fr - upTemp.r * fb,
        upTemp.r * fg - upTemp.g * fr
    });
    const Float3 u = {
        fg * rb - fb * rg,
        fb * rr - fr * rb,
        fr * rg - fg * rr
    };

    const Maf44 M = {
        {
            {rr, u.r, fr, pos.r},
            {rg, u.g, fg, pos.g},
            {rb, u.b, fb, pos.b},
            {0, 0, 0, 1}
        }
    };

    return M;
}

ViewParams cook_view_params(const Transform& cameraTransform)
{
    const Camera camera = {cameraTransform, 1.0f, 90.0f, static_cast<float>(WIDTH) / static_cast<float>(HEIGHT)};
    const Maf44 localToWorldMatrix = make_cam_matrix(cameraTransform.position, cameraTransform.lookat);
    const float planeHeight = camera.ncp * static_cast<float>(tan(camera.fov * 0.5f * DEG2RAD)) * 2.0f;
    const float planeWidth = planeHeight * camera.aspect;
    const ViewParams viewParams = {planeWidth, planeHeight, camera, localToWorldMatrix};
    return viewParams;
};

__device__ Float3 frag(const Float2 uv, const ViewParams* vp, const Float3* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, const int nmesh)
{
    const auto [u, v] = sub_f2_f(uv, 0.5f);
    const Float3 vpl = mul_f3( { u, v, 1.0f }, {vp->pwidth, vp->pheight, vp->cam.ncp});
    const Float4 vpl4 = f3_to_f4(vpl, 1.0f);
    const Float3 vpw = f4_to_f3(mul_maf4_f4(vp->ltwm, vpl4));
    const Float3 origin = vp->cam.t.position;
    const Ray ray = { .origin = origin, .dir = norm_f3(sub_f3(vpw, origin))};

    const Hit result = ray_collision(ray, materialBuffer, triangleBuffer, meshBuffer, nmesh);

    return result.material.color;
}

__global__ void render(Float3* out, const Float3* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, const ViewParams vp, const int nmesh)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    const unsigned int idx = y * WIDTH + x;
    Float2 uv;
    uv.u = static_cast<float>(x) / static_cast<float>(WIDTH - 1);
    uv.v = static_cast<float>(y) / static_cast<float>(HEIGHT - 1);

    out[idx] = frag(uv, &vp, materialBuffer, triangleBuffer, meshBuffer, nmesh);
}

__device__ Hit ray_collision(const Ray& ray, const Float3* materialBuffer, const Float3* triangleBuffer, const DeviceMesh* meshBuffer, const int nmesh)
{
    Hit closestHit = {false, FLT_MAX, {.0f,.0f, .0f}, {.0f, .0f, .0f}, {{.0,1.0,.0}}};
    for (int i = 0; i < nmesh; i++)
    {
        const DeviceMesh currentMesh = meshBuffer[i];
        for (int tr = currentMesh.tindex; tr < currentMesh.tindex + currentMesh.ntri; tr++)
        {
            const auto triangle = Triangle{
                triangleBuffer[tr*6],
                triangleBuffer[tr*6+1],
                triangleBuffer[tr*6+2],
                triangleBuffer[tr*6+3],
                triangleBuffer[tr*6+4],
                triangleBuffer[tr*6+5]
            };

            if (const Hit hit = hit_triangle(ray, triangle); hit.didHit && hit.distance < closestHit.distance)
            {
                const Float3 color = materialBuffer[currentMesh.mindex];
                closestHit = hit;
                closestHit.material = {color};
            }
        }
    }
    return closestHit;
}

__device__ __forceinline__ Hit hit_triangle(const Ray& ray, const Triangle& triangle)
{
    const Float3 edgeAB = sub_f3(triangle.posb, triangle.posa);
    const Float3 edgeAC = sub_f3(triangle.posc, triangle.posa);
    const Float3 normal = cross_f3(edgeAB, edgeAC);
    const Float3 ao = sub_f3(ray.origin, triangle.posa);
    const Float3 dao = cross_f3(ao, ray.dir);

    const float det = -dot_f3(ray.dir, normal);

    if (det < 1e-6f) {
        return Hit{false, FLT_MAX, {0,0,0}, {0,0,0}, {{0.0f, 0.0f, 0.0f}}};
    }

    const float invdet = 1.0f / det;
    const float dst = dot_f3(ao, normal) * invdet;

    if (dst < 0.0f) {
        return Hit{false, FLT_MAX, {0,0,0}, {0,0,0}, {{0.0f, 0.0f, 0.0f}}};
    }

    const float u = dot_f3(edgeAC, dao) * invdet;
    const float v = -dot_f3(edgeAB, dao) * invdet;
    const float w = 1.0f - u - v;

    if (u < 0.0f || v < 0.0f || w < 0.0f) {
        return Hit{false, FLT_MAX, {0,0,0}, {0,0,0}, {{0.0f, 0.0f, 0.0f}}};
    }

    const Float3 tloc = add_f3(ray.origin, mul_f3_f(ray.dir, dst));
    const Float3 tnorm = norm_f3(add_f3(add_f3(mul_f3_f(triangle.normb, u), mul_f3_f(triangle.normc, v)), mul_f3_f(triangle.norma, w)));
    return Hit{
        true,
        dst,
        tloc,
        tnorm,
        {{0.0f, 0.0f, 0.0f}}
    };
}