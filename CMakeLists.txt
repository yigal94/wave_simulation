cmake_minimum_required(VERSION 3.18)
project(HelloCUDA LANGUAGES CUDA CXX)

add_executable(hello
    main.cu
    simulation.cu
    medium.cu
    wave_state.cu
    wave_snapshot_recorder.cu
    wave_simulation_manager.cu
)

target_include_directories(hello PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(hello PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optimization flags
if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -lineinfo")
endif()
if (CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
endif()