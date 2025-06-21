#include <cuda_runtime.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>

#include "wave_snapshot_recorder.h"
#include "wave_state.h"

void WaveSnapshotRecorder::save_all(const std::string& prefix) const {
    for (size_t i = 0; i < snapshots.size(); ++i) {
        std::ostringstream fname;
        fname << prefix << "_step_" << i << ".csv";
        std::ofstream out(fname.str());
        const auto& slice = snapshots[i];
        for (int y = 0; y < grid_config.size_y; ++y) {
            for (int x = 0; x < grid_config.size_x; ++x) {
                out << slice[x + y * grid_config.size_x];
                if (x + 1 < grid_config.size_x) out << ",";
            }
            out << "\n";
        }
    }
}

void WaveSnapshotRecorder::save_all_to_dir(const std::string& dir) const {
    std::filesystem::create_directories(dir);
    // Write metadata JSON with SI unit info
    nlohmann::json meta;
    meta["size_x"] = grid_config.size_x;
    meta["size_y"] = grid_config.size_y;
    meta["size_z"] = grid_config.size_z;
    meta["dx"] = grid_config.dx;
    meta["dt"] = grid_config.dt;
    std::ofstream meta_out(dir + "/metadata.json");
    meta_out << meta.dump(4) << std::endl;
    meta_out.close();
    for (size_t i = 0; i < snapshots.size(); ++i) {
        std::ostringstream fname;
        fname << dir << "/slice_step_" << i << ".csv";
        std::ofstream out(fname.str());
        // Write header with SI unit info
        out << "# 2D slice at z = " << grid_config.size_z / 2 << ", dx = " << grid_config.dx
            << " m\n";
        for (int y = 0; y < grid_config.size_y; ++y) {
            for (int x = 0; x < grid_config.size_x; ++x) {
                out << snapshots[i][x + y * grid_config.size_x];
                if (x + 1 < grid_config.size_x) out << ",";
            }
            out << "\n";
        }
    }
}

void WaveSnapshotRecorder::set_sample_point(int x, int y, int z) {
    sample_x = x;
    sample_y = y;
    sample_z = z;
}

void WaveSnapshotRecorder::record_slice_from_device(const float* device_wave, int slice_z) {
    std::vector<float> slice(grid_config.size_x * grid_config.size_y);
    size_t offset = slice_z * grid_config.size_x * grid_config.size_y;
    cudaMemcpy(slice.data(), device_wave + offset,
               sizeof(float) * grid_config.size_x * grid_config.size_y, cudaMemcpyDeviceToHost);
    snapshots.push_back(std::move(slice));
}

void WaveSnapshotRecorder::record_point_from_device(const float* device_wave) {
    if (sample_x < 0 || sample_y < 0 || sample_z < 0) return;
    size_t idx = sample_x + sample_y * grid_config.size_x +
                 sample_z * grid_config.size_x * grid_config.size_y;
    float value;
    cudaMemcpy(&value, device_wave + idx, sizeof(float), cudaMemcpyDeviceToHost);
    point_samples.push_back(value);
}

void WaveSnapshotRecorder::save_point_samples_csv(const std::string& filename) const {
    std::ofstream out(filename);
    out << "# Point sample at (x, y, z) = (" << sample_x << ", " << sample_y << ", " << sample_z
        << ")\n";
    out << "# Each row: amplitude at time t = step * dt (dt in metadata.json)\n";
    for (size_t i = 0; i < point_samples.size(); ++i) {
        out << point_samples[i] << "\n";
    }
}
