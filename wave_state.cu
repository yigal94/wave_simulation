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

void WaveState::allocate_host() {
    size_t size = Resolution * Resolution * Resolution;
    CUDA_CHECK(cudaMallocHost(&wave, sizeof(float) * size));
    CUDA_CHECK(cudaMallocHost(&wave_prev, sizeof(float) * size));
    CUDA_CHECK(cudaMallocHost(&wave_next, sizeof(float) * size));
}

void WaveState::allocate_device() {
    size_t size = Resolution * Resolution * Resolution;
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

void WaveState::initialize() {
    size_t size = Resolution * Resolution * Resolution;
    std::memset(wave, 0, sizeof(float) * size);
    std::memset(wave_prev, 0, sizeof(float) * size);
}

void WaveState::move_to_device(WaveState& device_state) const {
    device_state.is_device = true;
    device_state.allocate_device();
    size_t size = Resolution * Resolution * Resolution;
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

void WaveState::excite(float power, int3 offset) {
    size_t center_x = Resolution / 2 + offset.x;
    size_t center_y = Resolution / 2 + offset.y;
    size_t center_z = Resolution / 2 + offset.z;
    wave[center_x + center_y * Resolution + center_z * Resolution * Resolution] = power;
}
