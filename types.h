#ifndef TYPES_H
#define TYPES_H

#include <cstddef>
#include <string>

struct GridConfig {
    int size_x;  // number of grid points in x
    int size_y;  // number of grid points in y
    int size_z;  // number of grid points in z
    int tile_x, tile_y, tile_z;
    int num_steps;  // number of time steps
    float dx;       // spatial step in meters (SI units)
    float dt;       // time step in seconds (SI units)
};

struct PhysicalProperties {
    float speed_of_sound;  // meters/second (SI units)
    float damping_factor;  // 1/second (SI units)
};

struct MaterialConfig {
    PhysicalProperties inner;
    PhysicalProperties outer;
    PhysicalProperties absorbing;  // Absorbing boundary material
    float absorbing_width_cm;      // Width of absorbing boundary (cm, SI: 0.01 m)
};

enum class ExcitationShape { Point, Sphere, Gaussian, Cube };

struct ExcitationConfig {
    float power;  // Initial amplitude at source (SI units: meters for displacement, Pascals for
                  // pressure)
    int3 offset;  // Offset from grid center in grid points (not meters)
    ExcitationShape shape = ExcitationShape::Gaussian;  // Default: Gaussian
    float width = 0.005f;                               // Width (meters), e.g. 5mm
    // Optionally, could add a position in meters for future extensibility
};

struct RecordingConfig {
    int sample_x, sample_y, sample_z;  // Sample point (grid indices)
    int slice_z;                       // z-index for 2D slice output
    int num_slice_samples;             // Number of slice samples to record
    std::string output_dir;            // Output directory
    // Note: Output files are in grid indices; see metadata for SI units
};

enum class ShapeType { Sphere, Cylinder, Custom };

struct GeometryConfig {
    ShapeType shape_type;
    float radius;   // meters (SI units)
    float3 center;  // meters (SI units)
    // For custom: a function pointer or lambda for host-side assignment
    bool (*custom_shape_func)(int, int, int) = nullptr;
};

struct SimulationConfig {
    GridConfig grid;
    MaterialConfig material;
    ExcitationConfig excitation;
    RecordingConfig recording;
    GeometryConfig geometry;
};

#endif  // TYPES_H
