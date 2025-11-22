#include "pyrolyse/pyrexport.cuh"

#include <cstdio>

#include "pyrolyse/pyrutils.cuh"

int write_bmp_on_file(const float3* pixels)
{
    const PyrConfig& config = get_config();

    FILE* fptr;
    fopen_s(&fptr, config.outfile_path.c_str(), "wb");
    if (!fptr) return 1;

    const int rowSize  = config.image_width * 3 + 3 & ~3;
    const int imageSize = rowSize * config.image_height;
    const int fileSize  = 54 + imageSize;

    unsigned char header[54] = {0};
    header[0] = 'B';
    header[1] = 'M';
    *reinterpret_cast<int*>(&header[2])  = fileSize;
    *reinterpret_cast<int*>(&header[10]) = 54;
    *reinterpret_cast<int*>(&header[14]) = 40;
    *reinterpret_cast<int*>(&header[18]) = config.image_width;
    *reinterpret_cast<int*>(&header[22]) = config.image_height;
    *reinterpret_cast<short*>(&header[26]) = 1;
    *reinterpret_cast<short*>(&header[28]) = 24;
    *reinterpret_cast<int*>(&header[34]) = imageSize;

    fwrite(header, 1, 54, fptr);

    auto* row = new unsigned char[rowSize];

    for (int y = config.image_height - 1; y >= 0; y--)
    {
        for (int x = 0; x < config.image_width; x++)
        {
            const int idx = y * config.image_width + x;
            const int i   = x * 3;

            row[i + 0] = clamp_to_255(pixels[idx].z);
            row[i + 1] = clamp_to_255(pixels[idx].y);
            row[i + 2] = clamp_to_255(pixels[idx].x);
        }
        fwrite(row, 1, rowSize, fptr);
    }

    delete[] row;
    fclose(fptr);

    return 0;
}
