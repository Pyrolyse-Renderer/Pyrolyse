#pragma once

#include <string>

struct PyrConfig
{
    float3 camera_start_pos, camera_start_lookat;
    int image_width, image_height, rayPerPixel, maxBounces;
    float camera_fov;
    std::string outfile_path, meshes_dir;
};

static PyrConfig load_config(const std::string& path);

const PyrConfig& get_config();

static const char* config_path();