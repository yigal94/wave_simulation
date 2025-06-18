#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "wave_snapshot_recorder.h"
#include "wave_state.h"

void WaveSnapshotRecorder::record_slice(const WaveState& state, int slice_z) {
    std::vector<float> slice(Resolution * Resolution);
    for (size_t y = 0; y < Resolution; ++y) {
        for (size_t x = 0; x < Resolution; ++x) {
            size_t idx = x + y * Resolution + slice_z * Resolution * Resolution;
            slice[x + y * Resolution] = state.wave[idx];
        }
    }
    snapshots.push_back(std::move(slice));
}

void WaveSnapshotRecorder::save_all(const std::string& prefix) const {
    for (size_t i = 0; i < snapshots.size(); ++i) {
        std::ostringstream fname;
        fname << prefix << "_step_" << i << ".csv";
        std::ofstream out(fname.str());
        const auto& slice = snapshots[i];
        for (size_t y = 0; y < Resolution; ++y) {
            for (size_t x = 0; x < Resolution; ++x) {
                out << slice[x + y * Resolution];
                if (x + 1 < Resolution) out << ",";
            }
            out << "\n";
        }
    }
}

void WaveSnapshotRecorder::save_all_to_dir(const std::string& dir) const {
    std::filesystem::create_directories(dir);
    for (size_t i = 0; i < snapshots.size(); ++i) {
        std::ostringstream fname;
        fname << dir << "/slice_step_" << i << ".csv";
        std::ofstream out(fname.str());
        // std::cout << "Saving snapshot to: " << fname.str() << std::endl;
        const auto& slice = snapshots[i];
        for (size_t y = 0; y < Resolution; ++y) {
            for (size_t x = 0; x < Resolution; ++x) {
                out << slice[x + y * Resolution];
                if (x + 1 < Resolution) out << ",";
            }
            out << "\n";
        }
    }
}
