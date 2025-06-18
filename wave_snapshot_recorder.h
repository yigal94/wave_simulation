#ifndef WAVE_SNAPSHOT_RECORDER_H
#define WAVE_SNAPSHOT_RECORDER_H

#include <string>
#include <tuple>
#include <vector>

#include "types.h"
#include "wave_state.h"

class WaveSnapshotRecorder {
   public:
    std::vector<std::vector<float>> snapshots;        // Each snapshot is a 2D slice
    std::vector<float> point_samples;                 // Sampled value at a point for each snapshot
    int sample_x = -1, sample_y = -1, sample_z = -1;  // Coordinates for point sampling
    GridConfig grid_config;

    WaveSnapshotRecorder(const GridConfig& grid) : grid_config(grid) {}
    // Set the point to sample at each snapshot
    void set_sample_point(int x, int y, int z);

    // Record a z-slice from device memory
    void record_slice_from_device(const float* device_wave, int slice_z);

    // Record a point sample from device memory
    void record_point_from_device(const float* device_wave);

    void save_all(const std::string& prefix) const;
    void save_all_to_dir(const std::string& dir) const;  // new method
    void save_point_samples_csv(const std::string& filename) const;
};

#endif  // WAVE_SNAPSHOT_RECORDER_H
