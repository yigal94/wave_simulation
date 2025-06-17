#include <cuda_runtime.h>

#include "simulation.h"
#include "types.h"
#include "utils.h"

struct Int3 {
    int x, y, z;
};

__device__ __forceinline__ bool is_boundary(int thread, int dim, int tile, int size, int d) {
    return ((d == -1 && thread == 0) || (d == 1 && thread == dim - 1)) && (tile + d >= 0) &&
           (tile + d < size);
}

__device__ void load_ghost_cells(float (&wave_tile)[TileX + 2][TileY + 2][TileZ + 2],
                                 float (&wave_prev_tile)[TileX + 2][TileY + 2][TileZ + 2],
                                 const float* wave, const float* wave_prev, int tile_x, int tile_y,
                                 int tile_z, int x_thread, int y_thread, int z_thread, int size_x,
                                 int size_y, int size_z) {
    Int3 displacements[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int i = 0; i < 6; ++i) {
        int dx = displacements[i].x, dy = displacements[i].y, dz = displacements[i].z;
        bool should_load = is_boundary(x_thread, TileX, tile_x, size_x, dx) ||
                           is_boundary(y_thread, TileY, tile_y, size_y, dy) ||
                           is_boundary(z_thread, TileZ, tile_z, size_z, dz);
        if (should_load) {
            int adj_x = tile_x + dx, adj_y = tile_y + dy, adj_z = tile_z + dz;
            if (adj_x >= 0 && adj_x < size_x && adj_y >= 0 && adj_y < size_y && adj_z >= 0 &&
                adj_z < size_z) {
                size_t adj_index = adj_x * size_y * size_z + adj_y * size_z + adj_z;
                wave_tile[x_thread + 1 + dx][y_thread + 1 + dy][z_thread + 1 + dz] =
                    wave[adj_index];
                wave_prev_tile[x_thread + 1 + dx][y_thread + 1 + dy][z_thread + 1 + dz] =
                    wave_prev[adj_index];
            }
        }
    }
}

__global__ void advance_wave_kernel(const float* wave, const float* wave_prev, float* wave_next,
                                    const PhysicalProperties* props, size_t size_x, size_t size_y,
                                    size_t size_z) {
    auto [x_block, y_block, z_block] = blockIdx;
    auto [x_thread, y_thread, z_thread] = threadIdx;
    auto [x_dim, y_dim, z_dim] = blockDim;
    size_t tile_x = x_block * x_dim + x_thread;
    size_t tile_y = y_block * y_dim + y_thread;
    size_t tile_z = z_block * z_dim + z_thread;
    __shared__ float wave_tile[TileX + 2][TileY + 2][TileZ + 2];
    __shared__ float wave_prev_tile[TileX + 2][TileY + 2][TileZ + 2];
    size_t my_index = tile_x * size_y * size_z + tile_y * size_z + tile_z;
    if (tile_x < size_x && tile_y < size_y && tile_z < size_z) {
        wave_tile[x_thread + 1][y_thread + 1][z_thread + 1] = wave[my_index];
        wave_prev_tile[x_thread + 1][y_thread + 1][z_thread + 1] = wave_prev[my_index];
    }
    load_ghost_cells(wave_tile, wave_prev_tile, wave, wave_prev, tile_x, tile_y, tile_z, x_thread,
                     y_thread, z_thread, size_x, size_y, size_z);
    PhysicalProperties props_local = props[my_index];
    __syncthreads();
    if (tile_x == 0 || tile_x == size_x - 1 || tile_y == 0 || tile_y == size_y - 1 || tile_z == 0 ||
        tile_z == size_z - 1) {
        wave_next[my_index] = 0.0f;
    } else if (tile_x < size_x && tile_y < size_y && tile_z < size_z) {
        float laplacian = wave_tile[x_thread + 2][y_thread + 1][z_thread + 1] +
                          wave_tile[x_thread][y_thread + 1][z_thread + 1] +
                          wave_tile[x_thread + 1][y_thread + 2][z_thread + 1] +
                          wave_tile[x_thread + 1][y_thread][z_thread + 1] +
                          wave_tile[x_thread + 1][y_thread + 1][z_thread + 2] +
                          wave_tile[x_thread + 1][y_thread + 1][z_thread] -
                          6.0f * wave_tile[x_thread + 1][y_thread + 1][z_thread + 1];
        // Calculate the next wave value using the wave equation
        // u_next = 2 * u_curr - u_prev + c^2 * laplacian - damping * u_curr
        float u_curr = wave_tile[x_thread + 1][y_thread + 1][z_thread + 1];
        float u_prev = wave_prev_tile[x_thread + 1][y_thread + 1][z_thread + 1];
        float c2 = props_local.speed_of_sound * props_local.speed_of_sound;
        float damping = props_local.damping_factor;

        wave_next[my_index] =
            (2.0f - damping) * u_curr - (1.0f - damping) * u_prev + c2 * laplacian;
    }
}

void advance_wave_kernel_launcher(const float* wave, const float* wave_prev, float* wave_next,
                                  const PhysicalProperties* props, size_t size_x, size_t size_y,
                                  size_t size_z, dim3 grid, dim3 block) {
    advance_wave_kernel<<<grid, block>>>(wave, wave_prev, wave_next, props, size_x, size_y, size_z);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
