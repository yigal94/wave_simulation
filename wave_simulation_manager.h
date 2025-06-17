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

    int num_steps;
    int slice_z;

    WaveSimulationManager(int steps, int slice_z);
    void setup(const PhysicalProperties& inner, const PhysicalProperties& outer,
               size_t inner_block_size);
    void run();
};

#endif  // WAVE_SIMULATION_MANAGER_H
