#include <cstdio>

#include "simulation.h"
#include "wave_simulation_manager.h"

WaveSimulationManager::WaveSimulationManager(const GridConfig& grid, const RecordingConfig& rec)
    : recorder(grid), grid_config(grid), recording_config(rec) {}

void WaveSimulationManager::run() {
    dim3 block(grid_config.tile_x, grid_config.tile_y, grid_config.tile_z);
    dim3 grid_dim(grid_config.size_x / grid_config.tile_x, grid_config.size_y / grid_config.tile_y,
                  grid_config.size_z / grid_config.tile_z);
    for (int step = 0; step < grid_config.num_steps; ++step) {
        advance_wave_kernel_launcher(device_state.wave, device_state.wave_prev,
                                     device_state.wave_next, device_medium.props, grid_config,
                                     grid_dim, block);
        // Sample the point every step
        recorder.record_point_from_device(device_state.wave);
        if (step % recording_config.slice_sample_interval == 0) {
            // Record z-slice every N steps
            recorder.record_slice_from_device(device_state.wave, recording_config.slice_z);
        }
        device_state.set_for_next_step();
        if (step % 100 == 0) {
            printf("Step %d completed.\n", step);
        }
    }
    // Optionally record final state
    recorder.record_slice_from_device(device_state.wave, recording_config.slice_z);
    recorder.record_point_from_device(device_state.wave);
}
