#include <cstring>

#include "medium.h"

Medium::Medium() {}

Medium::~Medium() {
    if (is_device) {
        free_device();
    } else {
        free_host();
    }
}

void Medium::allocate_host(const GridConfig& grid) {
    size_t size = grid.size_x * grid.size_y * grid.size_z;
    CUDA_CHECK(cudaMallocHost(&props, sizeof(PhysicalProperties) * size));
}

void Medium::allocate_device(const GridConfig& grid) {
    size_t size = grid.size_x * grid.size_y * grid.size_z;
    CUDA_CHECK(cudaMalloc(&props, sizeof(PhysicalProperties) * size));
}

void Medium::free_host() {
    if (props) CUDA_CHECK(cudaFreeHost(props));
    props = nullptr;
}

void Medium::free_device() {
    if (props) CUDA_CHECK(cudaFree(props));
    props = nullptr;
}

void Medium::move_to_device(Medium& device_medium, const GridConfig& grid) const {
    device_medium.is_device = true;
    device_medium.allocate_device(grid);
    size_t size = grid.size_x * grid.size_y * grid.size_z;
    CUDA_CHECK(cudaMemcpy(device_medium.props, props, sizeof(PhysicalProperties) * size,
                          cudaMemcpyHostToDevice));
}
