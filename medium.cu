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

void Medium::allocate_host() {
    size_t size = Resolution * Resolution * Resolution;
    CUDA_CHECK(cudaMallocHost(&props, sizeof(PhysicalProperties) * size));
}

void Medium::allocate_device() {
    size_t size = Resolution * Resolution * Resolution;
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

void Medium::initialize(const PhysicalProperties& inner_props,
                        const PhysicalProperties& outer_props, size_t inner_block_size) {
    size_t offset = (Resolution - inner_block_size) / 2;
    for (size_t z = 0; z < Resolution; ++z) {
        for (size_t y = 0; y < Resolution; ++y) {
            for (size_t x = 0; x < Resolution; ++x) {
                size_t index = x + y * Resolution + z * Resolution * Resolution;
                if (x >= offset && x < offset + inner_block_size && y >= offset &&
                    y < offset + inner_block_size && z >= offset && z < offset + inner_block_size) {
                    props[index] = inner_props;
                } else {
                    props[index] = outer_props;
                }
            }
        }
    }
}

void Medium::move_to_device(Medium& device_medium) const {
    device_medium.is_device = true;
    device_medium.allocate_device();
    size_t size = Resolution * Resolution * Resolution;
    CUDA_CHECK(cudaMemcpy(device_medium.props, props, sizeof(PhysicalProperties) * size,
                          cudaMemcpyHostToDevice));
}
