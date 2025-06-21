#include <filesystem>
#include <iostream>

#include "types.h"
#include "wave_simulation_manager.h"

SimulationConfig make_default_preset(const std::string& out_dir) {
    GridConfig grid_config{512, 96, 96, 8, 8, 8, 400000, 0.0005f, 1.0f / 882000.0f};
    MaterialConfig material_config{
        {200.0f, 0.01f},   // inner
        {10.0f, 2.0f},     // outer
        {10.0f, 2000.0f},  // absorbing: high damping
        0.5f,              // absorbing_width_cm: 1 cm
    };
    ExcitationConfig excitation_config;
    excitation_config.power = 100.0f;
    excitation_config.offset = {50, 0, 0};
    excitation_config.shape = ExcitationShape::Gaussian;
    excitation_config.width = 0.001f;  // 1cm Gaussian width
    RecordingConfig recording_config{grid_config.size_x / 8,
                                     grid_config.size_y / 2,
                                     grid_config.size_z / 2,
                                     grid_config.size_z / 2,
                                     20,
                                     out_dir};
    // Set geometry in SI units (meters), center offset from grid center for easier config
    float radius_m = 0.005f;               // 1cm radius in meters
    float3 center_m = {0.0f, 0.0f, 0.0f};  // Center at grid origin for simplicity
    GeometryConfig geometry_config{ShapeType::Cylinder, radius_m, center_m, nullptr};
    return SimulationConfig{grid_config, material_config, excitation_config, recording_config,
                            geometry_config};
}

auto make_shape_lambda(const GeometryConfig& geometry_config, const GridConfig& grid_config) {
    return [geometry_config, &grid_config](int x, int y, int z) -> bool {
        // Convert grid indices to meters
        float xm = (x - grid_config.size_x / 2) * grid_config.dx;
        float ym = (y - grid_config.size_y / 2) * grid_config.dx;
        float zm = (z - grid_config.size_z / 2) * grid_config.dx;
        if (geometry_config.shape_type == ShapeType::Cylinder) {
            float dx = xm - geometry_config.center.x;
            float dy = ym - geometry_config.center.y;
            float dz = zm - geometry_config.center.z;
            float half_length = (grid_config.size_x / 4.0f) * grid_config.dx;  // meters

            // Cylinder with spherical ends (capsule)
            if (dx >= -half_length && dx <= half_length) {
                // Inside the cylindrical section
                return (dy * dy + dz * dz < geometry_config.radius * geometry_config.radius);
            } else {
                // Check spherical caps at both ends
                float cap_center_x = (dx < -half_length) ? -half_length : half_length;
                float cap_dx = dx - cap_center_x;
                float dist2 = cap_dx * cap_dx + dy * dy + dz * dz;
                return (dist2 < geometry_config.radius * geometry_config.radius);
            }
        } else if (geometry_config.shape_type == ShapeType::Sphere) {
            float dx = xm - geometry_config.center.x;
            float dy = ym - geometry_config.center.y;
            float dz = zm - geometry_config.center.z;
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
