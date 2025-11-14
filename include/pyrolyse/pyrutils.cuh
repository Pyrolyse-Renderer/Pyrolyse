#pragma once

#include <string>

#include "types.cuh"

struct PyrConfig
{
    Float3 camera_start_pos;
    Float3 camera_start_lookat;
    int image_width;
    int image_height;
    float camera_fov;
};

static PyrConfig load_config(const std::string& path);

const PyrConfig& get_config();