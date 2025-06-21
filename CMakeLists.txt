cmake_minimum_required(VERSION 3.18)

# Prefer GCC 10 for C, C++, and CUDA host compiler if available
find_program(GCC_10_C_PATH NAMES gcc-10)
find_program(GCC_10_CXX_PATH NAMES g++-10)
if(GCC_10_C_PATH AND GCC_10_CXX_PATH)
    message(STATUS "Using gcc-10: ${GCC_10_C_PATH} and g++-10: ${GCC_10_CXX_PATH} for C/C++/CUDA host compiler")
    set(CMAKE_C_COMPILER ${GCC_10_C_PATH} CACHE FILEPATH "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER ${GCC_10_CXX_PATH} CACHE FILEPATH "C++ compiler" FORCE)
    set(CMAKE_CUDA_HOST_COMPILER ${GCC_10_CXX_PATH} CACHE FILEPATH "CUDA host compiler" FORCE)
else()
    message(WARNING "gcc-10/g++-10 not found. Using system default compilers.")
endif()

project(HelloCUDA LANGUAGES CUDA CXX)


# Optimization flags
if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -lineinfo --std=c++17 -allow-unsupported-compiler")
endif()
if (CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g --std=c++17")
endif()


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


# To override, run: cmake -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-XX ...