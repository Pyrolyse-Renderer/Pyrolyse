#include "pyrolyse/pyrutils.cuh"

#include <fstream>
#include <nlohmann/json.hpp>

static PyrConfig load_config(const std::string& path){
    std::ifstream f(path);
    nlohmann::json j;
    f >> j;

    PyrConfig config {};
    const float posx = j["posx"].get<float>();
    const float posy = j["posy"].get<float>();
    const float posz = j["posz"].get<float>();
    config.camera_start_pos = {posx, posy, posz};
    const float lax = j["lax"].get<float>();
    const float lay = j["lay"].get<float>();
    const float laz = j["laz"].get<float>();
    config.camera_start_lookat = {lax, lay, laz};
    config.camera_fov = j["fov"].get<float>();
    config.image_width = j["image_width"].get<int>();
    config.image_height = j["image_height"].get<int>();
    config.rayPerPixel = j["ray_per_pixel"].get<int>();
    config.maxBounces = j["max_bounces"].get<int>();
    config.meshes_dir = j["meshes_dir"].get<std::string>();
    config.outfile_path = j["outfile_path"].get<std::string>();
    return config;
}

static const char* config_path()
{
    const static char* path = getenv("PYROLYSE_CONFIG");
    if (!path)
    {
        fprintf(stderr,"Invalid config path. Check 'PYROLYSE_CONFIG' environment variable.\n");
        exit(1);
    }
    fprintf(stdout,"Loading config from %s\n", path);
    return path;
}

const PyrConfig& get_config()
{
    static PyrConfig config = load_config(config_path());
    return config;
}