#include <cmath>
#include <cstdio>
#include <ctime>

#include "simulation.h"
#include "wave_simulation_manager.h"

WaveSimulationManager::WaveSimulationManager(const GridConfig& grid, const RecordingConfig& rec)
    : recorder(grid), grid_config(grid), recording_config(rec) {}

void WaveSimulationManager::run() {
    dim3 block(grid_config.tile_x, grid_config.tile_y, grid_config.tile_z);
    dim3 grid_dim(grid_config.size_x / grid_config.tile_x, grid_config.size_y / grid_config.tile_y,
                  grid_config.size_z / grid_config.tile_z);

    int num_samples = recording_config.num_slice_samples;
    int total_steps = grid_config.num_steps;
    int snapshots_taken = 0;
    int log_samples = (int)(std::log((double)total_steps) * num_samples);
    // Precompute exponentially spaced snapshot steps
    std::vector<int> snapshot_steps;
    for (int i = 0; i < log_samples; ++i) {
        int step = (int)std::round(std::exp(i / (double)num_samples) *
                                   (total_steps / std::exp((double)log_samples / num_samples)));
        if (step >= total_steps) break;
        if (snapshot_steps.empty() || step > snapshot_steps.back()) snapshot_steps.push_back(step);
    }
    size_t next_snapshot_idx = 0;
    // --- ETA timing ---
    double start_time = clock();
    for (int step = 0; step < total_steps; ++step) {
        advance_wave_kernel_launcher(device_state.wave, device_state.wave_prev,
                                     device_state.wave_next, device_medium.props, grid_config,
                                     grid_dim, block);
        // Sample the point every step
        recorder.record_point_from_device(device_state.wave);
        // Take snapshot if this step matches the precomputed schedule
        if (next_snapshot_idx < snapshot_steps.size() &&
            step == snapshot_steps[next_snapshot_idx]) {
            recorder.record_slice_from_device(device_state.wave, recording_config.slice_z);
            ++snapshots_taken;
            ++next_snapshot_idx;
        }
        device_state.set_for_next_step();
        if (step % 100 == 0) {
            double elapsed = (clock() - start_time) / CLOCKS_PER_SEC;
            double eta = (step > 0) ? elapsed * (total_steps - step) / step : 0.0;
            int eta_min = (int)(eta / 60);
            int eta_sec = (int)(eta) % 60;
            double rate = (step > 0 && elapsed > 0) ? step / elapsed : 0.0;
            printf(
                "Step %d completed. Snapshots taken: %d | Elapsed: %.1fs | ETA: %dm %ds | Rate: "
                "%.2f steps/s\n",
                step, snapshots_taken, elapsed, eta_min, eta_sec, rate);
        }
    }
    // Optionally record final state
    recorder.record_slice_from_device(device_state.wave, recording_config.slice_z);
    recorder.record_point_from_device(device_state.wave);
}
