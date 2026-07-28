cmake_minimum_required(VERSION 3.28)

project(
    MetalRobo
    VERSION 0.1.0
    DESCRIPTION "Metal-native robotics simulation and reinforcement learning"
    LANGUAGES C CXX OBJCXX
)

if(NOT APPLE)
    message(FATAL_ERROR "MetalRobo 0.1 targets Apple Silicon macOS.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

set(METALROBO_SHADER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/Physics.metal"
)
set(METALROBO_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(METALROBO_AIR "${METALROBO_SHADER_DIR}/MetalRobo.air")
set(METALROBO_METALLIB "${METALROBO_SHADER_DIR}/MetalRobo.metallib")

add_custom_command(
    OUTPUT "${METALROBO_METALLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${METALROBO_SHADER_DIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_SHADER_SOURCE}"
            -o "${METALROBO_AIR}"
    COMMAND xcrun -sdk macosx metallib
            "${METALROBO_AIR}"
            -o "${METALROBO_METALLIB}"
    DEPENDS
        "${METALROBO_SHADER_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/gpu_types.h"
    VERBATIM
)

add_custom_target(metalrobo_shaders ALL DEPENDS "${METALROBO_METALLIB}")

add_library(
    metalrobo
    SHARED
    src/core/Model.cpp
    src/core/Franka.cpp
    src/core/CpuDynamics.cpp
    src/metal/MetalRuntime.mm
    src/c_api.cpp
)
add_dependencies(metalrobo metalrobo_shaders)

target_include_directories(
    metalrobo
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_definitions(
    metalrobo
    PRIVATE
        METALROBO_BUILDING_LIBRARY=1
        METALROBO_DEFAULT_METALLIB="${METALROBO_METALLIB}"
)
target_compile_options(
    metalrobo
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc;-Wall;-Wextra;-Wpedantic>"
)
target_link_libraries(
    metalrobo
    PRIVATE
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
        "${QUARTZCORE_FRAMEWORK}"
)
set_target_properties(
    metalrobo
    PROPERTIES
        OUTPUT_NAME metalrobo
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
        BUILD_RPATH "@loader_path/../lib"
        INSTALL_RPATH "@loader_path"
)

add_executable(metalrobo_bench apps/benchmark.cpp)
target_link_libraries(metalrobo_bench PRIVATE metalrobo)
set_target_properties(
    metalrobo_bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(metalrobo_cpu_probe apps/cpu_probe.cpp)
target_link_libraries(metalrobo_cpu_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_cpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(metalrobo_parity_probe apps/parity_probe.cpp)
target_link_libraries(metalrobo_parity_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_parity_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

install(TARGETS metalrobo metalrobo_bench)
install(DIRECTORY include/ DESTINATION include)
install(FILES "${METALROBO_METALLIB}" DESTINATION lib/metalrobo)
