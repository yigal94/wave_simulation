#ifndef MEDIUM_H
#define MEDIUM_H
#include <iostream>

#include "types.h"
#include "utils.h"

class Medium {
   public:
    PhysicalProperties* props = nullptr;
    bool is_device = false;

    Medium();
    ~Medium();

    void allocate_host(const GridConfig& grid);
    void allocate_device(const GridConfig& grid);
    void free_host();
    void free_device();

    // New: initialize with absorbing boundary
    void initialize_with_absorbing(const MaterialConfig& material, const GridConfig& grid);

    template <typename Func>
    void initialize(const MaterialConfig& material, Func&& is_inner, const GridConfig& grid) {
        float absorbing_width_m = material.absorbing_width_cm * 0.01f;
        int absorbing_width = static_cast<int>(absorbing_width_m / grid.dx + 0.5f);
        std::cout << "Absorbing width in grid points: " << absorbing_width << std::endl;
        for (int z = 0; z < grid.size_z; ++z) {
            for (int y = 0; y < grid.size_y; ++y) {
                for (int x = 0; x < grid.size_x; ++x) {
                    int index = x + y * grid.size_x + z * grid.size_x * grid.size_y;
                    if (is_inner(x, y, z)) {
                        props[index] = material.inner;
                    } else {
                        bool is_absorbing =
                            (x < absorbing_width || x >= grid.size_x - absorbing_width ||
                             y < absorbing_width || y >= grid.size_y - absorbing_width ||
                             z < absorbing_width || z >= grid.size_z - absorbing_width);
                        if (is_absorbing) {
                            props[index] = material.absorbing;
                        } else {
                            props[index] = material.outer;
                        }
                    }
                }
            }
        }
    }

    void move_to_device(Medium& device_medium, const GridConfig& grid) const;
};

#endif  // MEDIUM_H
