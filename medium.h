#ifndef MEDIUM_H
#define MEDIUM_H

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

    template <typename Func>
    void initialize(const PhysicalProperties& inner_props, const PhysicalProperties& outer_props,
                    Func&& is_inner, const GridConfig& grid) {
        for (int z = 0; z < grid.size_z; ++z) {
            for (int y = 0; y < grid.size_y; ++y) {
                for (int x = 0; x < grid.size_x; ++x) {
                    int index = x + y * grid.size_x + z * grid.size_x * grid.size_y;
                    if (is_inner(x, y, z)) {
                        props[index] = inner_props;
                    } else {
                        props[index] = outer_props;
                    }
                }
            }
        }
    }

    void move_to_device(Medium& device_medium, const GridConfig& grid) const;
};

#endif  // MEDIUM_H
