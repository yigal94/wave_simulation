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

    void allocate_host();
    void allocate_device();
    void free_host();
    void free_device();
    void initialize(const PhysicalProperties& inner_props,
                    const PhysicalProperties& outer_props, size_t inner_block_size);
    void move_to_device(Medium& device_medium) const;
};

#endif // MEDIUM_H
