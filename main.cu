#include <chrono>
#include <filesystem>

#include "pyrolyse/pyrexport.cuh"
#include "pyrolyse/pyrimport.cuh"
#include "pyrolyse/pyrrender.cuh"
#include "pyrolyse/pyrutils.cuh"
#include "pyrolyse/types.cuh"

int main()
{
    const PyrConfig config = get_config();

    const auto start = std::chrono::high_resolution_clock::now();

    const ViewParams viewParams = cook_view_params({config.camera_start_pos, config.camera_start_lookat});

    int meshcount = 0;
    int trianglecount = 0;
    const Mesh* meshes = load_models(meshcount, trianglecount);

    if (!meshes || meshcount == 0 || trianglecount == 0)
    {
        fprintf(stderr, "Failed to load meshes\n");
        return 1;
    }

    auto* cpu_meshes_values = new DeviceMesh[meshcount];
    auto* cpu_materials_values = new Float3[meshcount*3];
    auto* cpu_triangles_values = new Float3[trianglecount*6];
    cook_buffers(meshes, cpu_triangles_values, cpu_materials_values, cpu_meshes_values, meshcount);

    dim3 block(16,16);
    dim3 grid((config.image_width + block.x - 1) / block.x,(config.image_height + block.y - 1) / block.y);

    Float3* gpu_pixels;
    DeviceMesh* gpu_meshes;
    Float3* gpu_materials;
    Float3* gpu_triangles;

    cudaMalloc(&gpu_pixels, config.image_width * config.image_height * sizeof(Float3));
    cudaMalloc(&gpu_materials, meshcount * sizeof(Float3));
    cudaMalloc(&gpu_triangles, trianglecount * 6 * sizeof(Float3));
    cudaMalloc(&gpu_meshes, meshcount * sizeof(DeviceMesh));

    cudaMemcpy(gpu_meshes, cpu_meshes_values, meshcount * sizeof(DeviceMesh), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_triangles, cpu_triangles_values, trianglecount * 6 * sizeof(Float3), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_materials, cpu_materials_values, meshcount * sizeof(Float3), cudaMemcpyHostToDevice);

    delete[] cpu_meshes_values;
    delete[] cpu_triangles_values;
    delete[] cpu_materials_values;

    render<<<grid, block>>>(gpu_pixels, gpu_materials, gpu_triangles, gpu_meshes, viewParams, meshcount, config.image_width, config.image_height);
    cudaDeviceSynchronize();

    if (cudaGetLastError() != cudaSuccess)
    {
        printf("Something went wrong : %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    auto* pixel_out = new Float3[config.image_width * config.image_height];
    cudaMemcpy(pixel_out, gpu_pixels, config.image_width * config.image_height * sizeof(Float3), cudaMemcpyDeviceToHost);
    const int result = write_bmp_on_file(pixel_out);

    delete[] pixel_out;
    cudaFree(gpu_pixels);
    cudaFree(gpu_meshes);
    cudaFree(gpu_materials);
    cudaFree(gpu_triangles);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Task ended in : %lld ms", duration.count());

    return result;
}