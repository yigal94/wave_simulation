#include <filesystem>
#include <iostream>

#include "types.h"
#include "wave_simulation_manager.h"

int main(int argc, char** argv) {
    std::string out_dir = (argc > 1) ? argv[1] : "snapshots";
    if (!std::filesystem::exists(out_dir)) {
        std::filesystem::create_directories(out_dir);
    }
    PhysicalProperties inner{0.01f, 0.0001f};
    PhysicalProperties outer{0.001f, 0.0002f};
    int num_steps = 500;
    int slice_z = Resolution / 2;
    size_t inner_block_size = Resolution / 2;
    WaveSimulationManager sim(num_steps, slice_z);
    sim.setup(inner, outer, inner_block_size);
    sim.run();
    sim.recorder.save_all_to_dir(out_dir);
    return 0;
}
