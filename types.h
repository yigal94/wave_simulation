#ifndef TYPES_H
#define TYPES_H

#include <cstddef>
#include <string>

struct GridConfig {
    int size_x;
    int size_y;
    int size_z;
    int tile_x, tile_y, tile_z;
    int num_steps;
};

struct PhysicalProperties {
    float speed_of_sound;
    float damping_factor;
};

struct MaterialConfig {
    PhysicalProperties inner;
    PhysicalProperties outer;
};

struct ExcitationConfig {
    float power;
    int3 offset;
};

struct RecordingConfig {
    int sample_x, sample_y, sample_z;
    int slice_z;
    int slice_sample_interval;
    std::string output_dir;
};

enum class ShapeType { Sphere, Cylinder, Custom };

struct GeometryConfig {
    ShapeType shape_type;
    float radius;
    float3 center;
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
