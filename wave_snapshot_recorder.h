#ifndef WAVE_SNAPSHOT_RECORDER_H
#define WAVE_SNAPSHOT_RECORDER_H

#include <string>
#include <vector>

#include "types.h"
#include "wave_state.h"

class WaveSnapshotRecorder {
   public:
    std::vector<std::vector<float>> snapshots;  // Each snapshot is a 2D slice

    void record_slice(const WaveState& state, int slice_z);
    void save_all(const std::string& prefix) const;
    void save_all_to_dir(const std::string& dir) const;  // new method
};

#endif  // WAVE_SNAPSHOT_RECORDER_H
