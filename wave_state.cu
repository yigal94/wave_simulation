#include <cstring>

#include "wave_state.h"

WaveState::WaveState() {}

WaveState::~WaveState() {
    if (is_device) {
        free_device();
    } else {
        free_host();
    }
}

void WaveState::allocate_host(const GridConfig& grid) {
    size_t size = grid.size_x * grid.size_y * grid.size_z;
    CUDA_CHECK(cudaMallocHost(&wave, sizeof(float) * size));
    CUDA_CHECK(cudaMallocHost(&wave_prev, sizeof(float) * size));
    CUDA_CHECK(cudaMallocHost(&wave_next, sizeof(float) * size));
}

void WaveState::allocate_device(const GridConfig& grid) {
    size_t size = grid.size_x * grid.size_y * grid.size_z;
    CUDA_CHECK(cudaMalloc(&wave, sizeof(float) * size));
    CUDA_CHECK(cudaMalloc(&wave_prev, sizeof(float) * size));
    CUDA_CHECK(cudaMalloc(&wave_next, sizeof(float) * size));
}

void WaveState::free_host() {
    if (wave) CUDA_CHECK(cudaFreeHost(wave));
    if (wave_prev) CUDA_CHECK(cudaFreeHost(wave_prev));
    if (wave_next) CUDA_CHECK(cudaFreeHost(wave_next));
    wave = wave_prev = wave_next = nullptr;
}

void WaveState::free_device() {
    if (wave) CUDA_CHECK(cudaFree(wave));
    if (wave_prev) CUDA_CHECK(cudaFree(wave_prev));
    if (wave_next) CUDA_CHECK(cudaFree(wave_next));
    wave = wave_prev = wave_next = nullptr;
}

void WaveState::initialize(const GridConfig& grid) {
    size_t size = grid.size_x * grid.size_y * grid.size_z;
    std::memset(wave, 0, sizeof(float) * size);
    std::memset(wave_prev, 0, sizeof(float) * size);
}

void WaveState::move_to_device(WaveState& device_state, const GridConfig& grid) const {
    device_state.is_device = true;
    device_state.allocate_device(grid);
    size_t size = grid.size_x * grid.size_y * grid.size_z;
    CUDA_CHECK(cudaMemcpy(device_state.wave, wave, sizeof(float) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_state.wave_prev, wave_prev, sizeof(float) * size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_state.wave_next, wave_next, sizeof(float) * size,
                          cudaMemcpyHostToDevice));
}

void WaveState::set_for_next_step() {
    float* temp = wave_prev;
    wave_prev = wave;
    wave = wave_next;
    wave_next = temp;
}

void WaveState::excite(const ExcitationConfig& excitation, const GridConfig& grid) {
    // Center in grid
    int center_x = grid.size_x / 2 + excitation.offset.x;
    int center_y = grid.size_y / 2 + excitation.offset.y;
    int center_z = grid.size_z / 2 + excitation.offset.z;
    float width = excitation.width > 0 ? excitation.width : grid.dx;  // meters
    int radius_grid = std::max(1, static_cast<int>(10 * width / grid.dx + 0.5f));
    for (int z = center_z - radius_grid; z <= center_z + radius_grid; ++z) {
        for (int y = center_y - radius_grid; y <= center_y + radius_grid; ++y) {
            for (int x = center_x - radius_grid; x <= center_x + radius_grid; ++x) {
                if (x < 0 || x >= grid.size_x || y < 0 || y >= grid.size_y || z < 0 ||
                    z >= grid.size_z)
                    continue;
                float dx = (x - center_x) * grid.dx;
                float dy = (y - center_y) * grid.dx;
                float dz = (z - center_z) * grid.dx;
                float r2 = dx * dx + dy * dy + dz * dz;
                float value = 0.0f;
                switch (excitation.shape) {
                    case ExcitationShape::Point:
                        if (x == center_x && y == center_y && z == center_z)
                            value = excitation.power;
                        break;
                    case ExcitationShape::Sphere:
                        if (r2 <= width * width) value = excitation.power;
                        break;
                    case ExcitationShape::Gaussian:
                        value = excitation.power * expf(-r2 / (2.0f * width * width));
                        break;
                    case ExcitationShape::Cube:
                        value = excitation.power;
                        break;
                }
                if (value != 0.0f) {
                    wave[x + y * grid.size_x + z * grid.size_x * grid.size_y] = value;
                    wave_prev[x + y * grid.size_x + z * grid.size_x * grid.size_y] = value;
                }
            }
        }
    }
}
