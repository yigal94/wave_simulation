#ifndef WAVE_STATE_H
#define WAVE_STATE_H

#include "types.h"
#include "utils.h"

class WaveState {
   public:
    float* wave = nullptr;
    float* wave_prev = nullptr;
    float* wave_next = nullptr;
    bool is_device = false;

    WaveState();
    ~WaveState();

    void allocate_host(const GridConfig& grid);
    void allocate_device(const GridConfig& grid);
    void free_host();
    void free_device();
    void initialize(const GridConfig& grid);
    void move_to_device(WaveState& device_state, const GridConfig& grid) const;
    void set_for_next_step();
    void excite(float power, int3 offset, const GridConfig& grid);
};

#endif  // WAVE_STATE_H
