#include <cstdio>

#include "simulation.h"
#include "wave_simulation_manager.h"

WaveSimulationManager::WaveSimulationManager(int steps, int slice_z)
    : num_steps(steps), slice_z(slice_z) {}

void WaveSimulationManager::setup(const PhysicalProperties& inner, const PhysicalProperties& outer,
                                  size_t inner_block_size) {
    host_state.allocate_host();
    host_state.initialize();
    host_state.excite(100.0f);
    host_medium.allocate_host();
    host_medium.initialize(inner, outer, inner_block_size);
    host_state.move_to_device(device_state);
    host_medium.move_to_device(device_medium);
}

void WaveSimulationManager::run() {
    constexpr size_t block_size = TileVolume;
    dim3 block(TileX, TileY, block_size / (TileX * TileY));
    dim3 grid(Resolution / TileX, Resolution / TileY, Resolution / TileZ);
    for (int step = 0; step < num_steps; ++step) {
        advance_wave_kernel_launcher(device_state.wave, device_state.wave_prev,
                                     device_state.wave_next, device_medium.props, Resolution,
                                     Resolution, Resolution, grid, block);
        if (step % 10 == 0) {
            // Copy slice from device to host and record
            CUDA_CHECK(cudaMemcpy(host_state.wave, device_state.wave,
                                  sizeof(float) * Resolution * Resolution * Resolution,
                                  cudaMemcpyDeviceToHost));
            recorder.record_slice(host_state, slice_z);
        }
        device_state.set_for_next_step();
        if (step % 100 == 0) {
            printf("Step %d completed.\n", step);
        }
    }
    CUDA_CHECK(cudaMemcpy(host_state.wave, device_state.wave,
                          sizeof(float) * Resolution * Resolution * Resolution,
                          cudaMemcpyDeviceToHost));
}
