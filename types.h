#ifndef TYPES_H
#define TYPES_H

#include <cstddef>

constexpr size_t Resolution = 256;
constexpr size_t TileVolume = 1024;
constexpr size_t TileX = 16;
constexpr size_t TileY = 16;
constexpr size_t TileZ = TileVolume / (TileX * TileY);

struct PhysicalProperties {
    float speed_of_sound;
    float damping_factor;
};

#endif  // TYPES_H
