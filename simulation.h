#ifndef SIMULATION_H
#define SIMULATION_H

#include "types.h"

void advance_wave_kernel_launcher(const float* wave, const float* wave_prev, float* wave_next,
                                  const PhysicalProperties* props, size_t size_x, size_t size_y,
                                  size_t size_z, dim3 grid, dim3 block);

#endif  // SIMULATION_H
