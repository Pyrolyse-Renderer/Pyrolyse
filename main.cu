#include <chrono>

#include "pyrolyse/pyrexport.cuh"
#include "pyrolyse/pyrimport.cuh"
#include "pyrolyse/pyrrender.cuh"
#include "pyrolyse/pyrutils.cuh"
#include "pyrolyse/types.cuh"

static ViewParams cook_view_params()
{
    constexpr Transform cameraTransform = {{0.0f,0.0f,2.0f},{0.0f,0.0f,0.0f}};
    constexpr Camera camera = {cameraTransform, 1.0f, 90.0f, static_cast<float>(WIDTH) / static_cast<float>(HEIGHT)};
    const Maf44 localToWorldMatrix = make_cam_matrix(cameraTransform.position, cameraTransform.lookat);
    const float planeHeight = camera.ncp * static_cast<float>(tan(camera.fov * 0.5f * DEG2RAD)) * 2.0f;
    const float planeWidth = planeHeight * camera.aspect;
    const ViewParams viewParams = {planeWidth, planeHeight, camera, localToWorldMatrix};
    return viewParams;
};

int main()
{
    const auto start = std::chrono::high_resolution_clock::now();

    const ViewParams viewParams = cook_view_params();

    int meshcount = 0;
    int trianglecount = 0;
    const Mesh* meshes = load_models(meshcount, trianglecount);

    auto* cpu_meshes_values = new DeviceMesh[meshcount];
    auto* cpu_materials_values = new Float3[meshcount*3];
    auto* cpu_triangles_values = new Float3[trianglecount*6];
    cook_buffers(meshes, cpu_triangles_values, cpu_materials_values, cpu_meshes_values, meshcount);

    dim3 block(16,16);
    dim3 grid((WIDTH + block.x - 1) / block.x,(HEIGHT + block.y - 1) / block.y);

    Float3* gpu_pixels;
    DeviceMesh* gpu_meshes;
    Float3* gpu_materials;
    Float3* gpu_triangles;

    cudaMalloc(&gpu_pixels, WIDTH * HEIGHT * sizeof(Float3));
    cudaMalloc(&gpu_materials, meshcount * 3 * sizeof(Float3));
    cudaMalloc(&gpu_triangles, trianglecount * 6 * sizeof(Float3));
    cudaMalloc(&gpu_meshes, meshcount * sizeof(DeviceMesh));

    cudaMemcpy(gpu_meshes, cpu_meshes_values, meshcount * sizeof(DeviceMesh), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_triangles, cpu_triangles_values, trianglecount * 6 * sizeof(Float3), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_materials, cpu_materials_values, meshcount * 3 * sizeof(Float3), cudaMemcpyHostToDevice);

    delete[] cpu_meshes_values;
    delete[] cpu_triangles_values;
    delete[] cpu_materials_values;

    render<<<grid, block>>>(gpu_pixels, gpu_materials, gpu_triangles, gpu_meshes, viewParams, meshcount);
    cudaDeviceSynchronize();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("Something went wrong : %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    auto* pixel_out = new Float3[WIDTH * HEIGHT];
    cudaMemcpy(pixel_out, gpu_pixels, WIDTH * HEIGHT * sizeof(Float3), cudaMemcpyDeviceToHost);
    const int result = write_bmp_on_file(pixel_out);

    delete[] pixel_out;
    cudaFree(gpu_pixels);
    cudaFree(gpu_meshes);
    cudaFree(gpu_materials);
    cudaFree(gpu_triangles);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Task ended in : %lld ms, eq to %f fps\n", duration.count(), 1000.0 / static_cast<double>(duration.count()));

    return result;
}