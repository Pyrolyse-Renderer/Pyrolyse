#include "pyrolyse/pyrrender.cuh"
#include "pyrolyse/pyrutils.cuh"

Maf44 make_cam_matrix(const Float3 pos, const Float3 target)
{
    constexpr Float3 upTemp = {0, 1, 0};

    const Float3 f = norm_f3(sub_f3(target, pos));
    const Float3 r = norm_f3({
        upTemp.g * f.b - upTemp.b * f.g,
        upTemp.b * f.r - upTemp.r * f.b,
        upTemp.r * f.g - upTemp.g * f.r
    });
    const Float3 u = {
        f.g * r.b - f.b * r.g,
        f.b * r.r - f.r * r.b,
        f.r * r.g - f.g * r.r
    };

    const Maf44 M = {
        {
            {r.r, u.r, f.r, pos.r},
            {r.g, u.g, f.g, pos.g},
            {r.b, u.b, f.b, pos.b},
            {0, 0, 0, 1}
        }
    };

    return M;
}

__device__ Float3 frag(const Float2 uv, const ViewParams* vp)
{
    const auto [u, v] = sub_f2_f(uv, 0.5f);
    const Float3 vpl = mul_f3( { u, v, 1.0f }, {vp->pwidth, vp->pheight, vp->cam.ncp}); // ERROR
    const Float4 vpl4 = f3_to_f4(vpl, 1.0f);
    const Float3 vpw = f4_to_f3(mul_maf4_f4(&vp->ltwm, &vpl4)); // ERROR
    const Float3 origin = vp->cam.t.position;
    const Ray ray = { .origin = origin, .dir = norm_f3(sub_f3(vpw, origin))};
    return ray.dir;
}

__global__ void trace(Float3* out, const ViewParams viewParams)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    const unsigned int idx = y * WIDTH + x;
    Float2 uv;
    uv.u = static_cast<float>(x) / static_cast<float>(WIDTH - 1);
    uv.v = static_cast<float>(y) / static_cast<float>(HEIGHT - 1);

    out[idx] = frag(uv, &viewParams);
}