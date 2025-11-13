#include "pyrolyse/pyrexport.cuh"

#include <cstdio>

#include "pyrolyse/pyrutils.cuh"

int write_bmp_on_file(const Float3* pixels)
{
    char* outfilepath;
    size_t outfilepathlen;
    _dupenv_s(&outfilepath, &outfilepathlen, "PYROLYSE_OUT_FILE");
    if (outfilepathlen == 0) return 1;

    FILE* fptr;
    fopen_s(&fptr, outfilepath, "wb");
    if (!fptr) return 1;

    constexpr int rowSize  = WIDTH * 3 + 3 & ~3;
    constexpr int imageSize = rowSize * HEIGHT;
    constexpr int fileSize  = 54 + imageSize;

    unsigned char header[54] = {0};
    header[0] = 'B';
    header[1] = 'M';
    *reinterpret_cast<int*>(&header[2])  = fileSize;
    *reinterpret_cast<int*>(&header[10]) = 54;
    *reinterpret_cast<int*>(&header[14]) = 40;
    *reinterpret_cast<int*>(&header[18]) = WIDTH;
    *reinterpret_cast<int*>(&header[22]) = HEIGHT;
    *reinterpret_cast<short*>(&header[26]) = 1;
    *reinterpret_cast<short*>(&header[28]) = 24;
    *reinterpret_cast<int*>(&header[34]) = imageSize;

    fwrite(header, 1, 54, fptr);

    auto* row = new unsigned char[rowSize];

    for (int y = HEIGHT - 1; y >= 0; y--)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            const int idx = y * WIDTH + x;
            const int i   = x * 3;

            row[i + 0] = clamp_to_255(pixels[idx].b);
            row[i + 1] = clamp_to_255(pixels[idx].g);
            row[i + 2] = clamp_to_255(pixels[idx].r);
        }
        fwrite(row, 1, rowSize, fptr);
    }

    delete[] row;
    fclose(fptr);

    return 0;
}
