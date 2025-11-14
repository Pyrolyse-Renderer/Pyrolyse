#include "pyrolyse/pyrimport.cuh"

#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <vector>

Mesh load_obj(const char* path)
{
    std::ifstream ifs(path);
    std::vector<Triangle> tris;

    if (!ifs) return Mesh{nullptr, 0, {{.0f, .0f, .0f}}};

    int n = 0;
    ifs >> n;
    tris.reserve(n);

    float x1, y1, z1, nx1, ny1, nz1, x2, y2, z2, nx2, ny2, nz2, x3, y3, z3, nx3, ny3, nz3;
    for (std::size_t i = 0; i < n; ++i)
    {
        ifs
            >> x1 >> y1 >> z1 >> nx1 >> ny1 >> nz1
            >> x2 >> y2 >> z2 >> nx2 >> ny2 >> nz2
            >> x3 >> y3 >> z3 >> nx3 >> ny3 >> nz3;

        Triangle t{
            {x1, y1, z1},
            {x2, y2, z2},
            {x3, y3, z3},
            {nx1, ny1, nz1},
            {nx2, ny2, nz2},
            {nx3, ny3, nz3},
        };

        tris.push_back(t);
    }

    const auto ts = tris.size();
    auto* triarray = new Triangle[ts];
    for (int i = 0; i < ts; i++) triarray[i] = tris[i];

    return Mesh{triarray, n, {{1.0f, 1.0f, 1.0f}}};

}

Mesh* load_models(int& out_mesh_count, int& out_triangle_count)
{
    std::vector<Mesh> meshes = {};
    out_mesh_count = 0;
    const std::string& dir = "D:/Pyrolyse/samples";
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        fprintf(stderr, "Failed to find folder %s\n", dir.c_str());
        return nullptr;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".pyrobj") {
            const Mesh m = load_obj(entry.path().generic_string().c_str());
            meshes.push_back(m);
            out_triangle_count += m.ntri;
            out_mesh_count++;
        }
    }
    const auto ms = meshes.size();
    const auto meshesarray = new Mesh[ms];
    for (int i = 0; i < ms; i++) meshesarray[i] = meshes[i];
    return meshesarray;
}

void cook_buffers(const Mesh* meshes, Float3* out_triangles, Float3* out_materials, DeviceMesh* out_meshes, const int meshcount)
{
    int k = 0;
    int tri_idx = 0;
    int l = 0;
    int m = 0;

    for (int i = 0; i < meshcount; i++) {
        const int n = meshes[i].ntri;
        const int start_tri = tri_idx;
        for (int t = 0; t < n; t++) {
            out_triangles[k++] = meshes[i].triangles[t].posa;
            out_triangles[k++] = meshes[i].triangles[t].posb;
            out_triangles[k++] = meshes[i].triangles[t].posc;
            out_triangles[k++] = meshes[i].triangles[t].norma;
            out_triangles[k++] = meshes[i].triangles[t].normb;
            out_triangles[k++] = meshes[i].triangles[t].normc;
            tri_idx++;
        }

        out_materials[l] = {1.0f,0.0f,0.0f};
        out_meshes[m] = {start_tri, n, l};
        l++;
        m++;
    }
}