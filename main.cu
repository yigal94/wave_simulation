#include <filesystem>
#include <iostream>

#include "types.h"
#include "wave_simulation_manager.h"

SimulationConfig make_default_preset(const std::string& out_dir) {
    GridConfig grid_config{256, 256, 256, 8, 8, 8, 44100};
    MaterialConfig material_config{{0.3f, 0.00001f}, {0.03f, 0.0001f}};
    ExcitationConfig excitation_config{100.0f, {-10, 9, 0}};
    RecordingConfig recording_config{grid_config.size_x / 8,
                                     grid_config.size_y / 4,
                                     grid_config.size_z / 2,
                                     grid_config.size_z / 2,
                                     250,
                                     out_dir};
    GeometryConfig geometry_config{
        ShapeType::Cylinder,
        10.0f,
        {grid_config.size_x / 2.0f, grid_config.size_y / 2.0f, grid_config.size_z / 2.0f},
        nullptr};
    return SimulationConfig{grid_config, material_config, excitation_config, recording_config,
                            geometry_config};
}

auto make_shape_lambda(const GeometryConfig& geometry_config, const GridConfig& grid_config) {
    return [geometry_config, &grid_config](int x, int y, int z) -> bool {
        if (geometry_config.shape_type == ShapeType::Cylinder) {
            float dx = x - geometry_config.center.x;
            float dy = y - geometry_config.center.y;
            float dz = z - geometry_config.center.z;
            float half_length = grid_config.size_x / 4.0f;
            return (dx >= -half_length && dx <= half_length) &&
                   (dy * dy + dz * dz < geometry_config.radius * geometry_config.radius);
        } else if (geometry_config.shape_type == ShapeType::Sphere) {
            float dx = x - geometry_config.center.x;
            float dy = y - geometry_config.center.y;
            float dz = z - geometry_config.center.z;
            return (dx * dx + dy * dy + dz * dz < geometry_config.radius * geometry_config.radius);
        } else if (geometry_config.shape_type == ShapeType::Custom &&
                   geometry_config.custom_shape_func) {
            return geometry_config.custom_shape_func(x, y, z);
        }
        return false;
    };
}

int main(int argc, char** argv) {
    std::string out_dir = (argc > 1) ? argv[1] : "snapshots";
    if (!std::filesystem::exists(out_dir)) {
        std::filesystem::create_directories(out_dir);
    }
    SimulationConfig config = make_default_preset(out_dir);
    auto shape_lambda = make_shape_lambda(config.geometry, config.grid);
    WaveSimulationManager sim(config.grid, config.recording);
    sim.setup(config.material, config.excitation, shape_lambda);
    sim.run();
    sim.recorder.save_all_to_dir(config.recording.output_dir);
    sim.recorder.save_point_samples_csv(config.recording.output_dir + "/point_samples.csv");
    return 0;
}
