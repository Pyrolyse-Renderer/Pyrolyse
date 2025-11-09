#include "pyrolyse/pyrexport.cuh"
#include "pyrolyse/pyrrender.cuh"
#include "pyrolyse/pyrutils.cuh"
#include "pyrolyse/types.cuh"

int main()
{
    constexpr Transform cameraTransform = {{0.0f,0.0f,0.0f},{-1.0f,-1.0f,1.0f}};
    constexpr Camera camera = {cameraTransform, 1.0f, 90.0f, static_cast<float>(WIDTH) / static_cast<float>(HEIGHT)};
    const Maf44 localToWorldMatrix = make_cam_matrix(cameraTransform.position, cameraTransform.lookat);
    const float planeHeight = camera.ncp * static_cast<float>(tan(camera.fov * 0.5f * DEG2RAD)) * 2.0f;
    const float planeWidth = planeHeight * camera.aspect;
    const ViewParams viewParams = {planeWidth, planeHeight, camera, localToWorldMatrix};

    dim3 block(16,16);
    dim3 grid((WIDTH + block.x - 1) / block.x,(HEIGHT + block.y - 1) / block.y);
    Float3* d_out;
    cudaMalloc(&d_out, WIDTH * HEIGHT * sizeof(Float3));
    trace<<<grid, block>>>(d_out, viewParams);
    cudaDeviceSynchronize();
    auto* h_out = new Float3[WIDTH * HEIGHT];
    cudaMemcpy(h_out, d_out, WIDTH * HEIGHT * sizeof(Float3), cudaMemcpyDeviceToHost);
    const int result = write_bmp_on_file(h_out);
    delete[] h_out;
    cudaFree(d_out);
    return result;
}
