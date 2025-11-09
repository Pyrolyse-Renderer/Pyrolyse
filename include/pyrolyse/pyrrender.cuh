#pragma once

#include "pyrmaths.cuh"
#include "types.cuh"

Maf44 make_cam_matrix(Float3 pos, Float3 target);
__device__ Float3 frag(Float2 uv, const ViewParams* vp);
__global__ void trace(Float3* out, ViewParams viewParams);
