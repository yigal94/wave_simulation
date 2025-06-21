#ifndef WAVE_SIMULATION_MANAGER_H
#define WAVE_SIMULATION_MANAGER_H

#include "medium.h"
#include "types.h"
#include "wave_snapshot_recorder.h"

class WaveSimulationManager {
   public:
    WaveState host_state, device_state;
    Medium host_medium, device_medium;
    WaveSnapshotRecorder recorder;

    GridConfig grid_config;
    RecordingConfig recording_config;
    int slice_z;

    WaveSimulationManager(const GridConfig& grid, const RecordingConfig& rec);
    template <typename Func>
    void setup(const MaterialConfig& material, const ExcitationConfig& excitation,
               Func&& is_inner) {
        host_state.allocate_host(grid_config);
        host_state.initialize(grid_config);
        host_state.excite(excitation, grid_config);
        host_medium.allocate_host(grid_config);
        // Use new unified initialization: inner > absorbing > outer
        host_medium.initialize(material, std::forward<Func>(is_inner), grid_config);
        host_state.move_to_device(device_state, grid_config);
        host_medium.move_to_device(device_medium, grid_config);
        recorder.set_sample_point(recording_config.sample_x, recording_config.sample_y,
                                  recording_config.sample_z);
    }
    void run();
};

#endif  // WAVE_SIMULATION_MANAGER_H
