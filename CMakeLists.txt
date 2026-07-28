cmake_minimum_required(VERSION 3.28)

project(
    MetalRobo
    VERSION 0.2.0
    DESCRIPTION "Metal-native robotics simulation and reinforcement learning"
    LANGUAGES C CXX OBJCXX
)

if(NOT APPLE)
    message(FATAL_ERROR "MetalRobo 0.2 targets Apple Silicon macOS.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

set(METALROBO_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(METALROBO_PHYSICS_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/Physics.metal"
)
set(METALROBO_CONSTRAINT_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ConstraintSolver.metal"
)
set(METALROBO_FREE_BODY_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/FreeBodyDynamics.metal"
)
set(METALROBO_COLLISION_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/Collision.metal"
)
set(METALROBO_PHYSICS_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboPhysics.air"
)
set(METALROBO_CONSTRAINT_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboConstraintSolver.air"
)
set(METALROBO_FREE_BODY_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboFreeBodyDynamics.air"
)
set(METALROBO_COLLISION_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboCollision.air"
)
set(METALROBO_METALLIB "${METALROBO_SHADER_DIR}/MetalRobo.metallib")

add_custom_command(
    OUTPUT "${METALROBO_METALLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${METALROBO_SHADER_DIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_PHYSICS_SOURCE}"
            -o "${METALROBO_PHYSICS_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_CONSTRAINT_SOURCE}"
            -o "${METALROBO_CONSTRAINT_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_FREE_BODY_SOURCE}"
            -o "${METALROBO_FREE_BODY_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_COLLISION_SOURCE}"
            -o "${METALROBO_COLLISION_AIR}"
    COMMAND xcrun -sdk macosx metallib
            "${METALROBO_PHYSICS_AIR}"
            "${METALROBO_CONSTRAINT_AIR}"
            "${METALROBO_FREE_BODY_AIR}"
            "${METALROBO_COLLISION_AIR}"
            -o "${METALROBO_METALLIB}"
    DEPENDS
        "${METALROBO_PHYSICS_SOURCE}"
        "${METALROBO_CONSTRAINT_SOURCE}"
        "${METALROBO_FREE_BODY_SOURCE}"
        "${METALROBO_COLLISION_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/gpu_types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/engine_types.h"
    VERBATIM
)

add_custom_target(metalrobo_shaders ALL DEPENDS "${METALROBO_METALLIB}")

add_library(
    metalrobo
    SHARED
    src/core/Model.cpp
    src/core/Franka.cpp
    src/core/CpuDynamics.cpp
    src/core/EngineModel.cpp
    src/core/G1.cpp
    src/core/Collision.cpp
    src/core/FreeBodyDynamics.cpp
    src/core/ArticulatedDynamics.cpp
    src/core/ReferenceConicSolver.cpp
    src/core/QualityContactSolver.cpp
    src/core/ConstraintSolver.cpp
    src/core/RigidBodyWorld.cpp
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

add_executable(metalrobo_engine_model_probe apps/engine_model_probe.cpp)
target_link_libraries(metalrobo_engine_model_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_engine_model_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(metalrobo_g1_model_probe apps/g1_model_probe.cpp)
target_link_libraries(metalrobo_g1_model_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_g1_model_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(metalrobo_collision_probe apps/collision_probe.cpp)
target_link_libraries(metalrobo_collision_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_collision_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(metalrobo_free_body_probe apps/free_body_probe.cpp)
target_link_libraries(metalrobo_free_body_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_free_body_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_rigid_body_world_probe
    apps/rigid_body_world_probe.cpp
)

add_executable(
    metalrobo_articulated_dynamics_probe
    apps/articulated_dynamics_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_dynamics_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_dynamics_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)
target_link_libraries(metalrobo_rigid_body_world_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_rigid_body_world_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_reference_conic_probe
    apps/reference_conic_probe.cpp
)

add_executable(
    metalrobo_quality_contact_probe
    apps/quality_contact_probe.cpp
)
target_link_libraries(metalrobo_quality_contact_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_quality_contact_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)
target_link_libraries(metalrobo_reference_conic_probe PRIVATE metalrobo)
set_target_properties(
    metalrobo_reference_conic_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(metalrobo_contact_gpu_probe apps/contact_gpu_probe.mm)
add_dependencies(metalrobo_contact_gpu_probe metalrobo_shaders)
target_compile_definitions(
    metalrobo_contact_gpu_probe
    PRIVATE
        METALROBO_DEFAULT_METALLIB="${METALROBO_METALLIB}"
)
target_compile_options(
    metalrobo_contact_gpu_probe
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
        -Wpedantic
)
target_link_libraries(
    metalrobo_contact_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_contact_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(metalrobo_free_body_gpu_probe apps/free_body_gpu_probe.mm)
add_dependencies(metalrobo_free_body_gpu_probe metalrobo_shaders)
target_compile_definitions(
    metalrobo_free_body_gpu_probe
    PRIVATE
        METALROBO_DEFAULT_METALLIB="${METALROBO_METALLIB}"
)
target_compile_options(
    metalrobo_free_body_gpu_probe
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
        -Wpedantic
)
target_link_libraries(
    metalrobo_free_body_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_free_body_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(metalrobo_collision_gpu_probe apps/collision_gpu_probe.mm)
add_dependencies(metalrobo_collision_gpu_probe metalrobo_shaders)
target_compile_definitions(
    metalrobo_collision_gpu_probe
    PRIVATE
        METALROBO_COLLISION_METALLIB="${METALROBO_METALLIB}"
)
target_compile_options(
    metalrobo_collision_gpu_probe
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
        -Wpedantic
)
target_link_libraries(
    metalrobo_collision_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_collision_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

install(TARGETS metalrobo metalrobo_bench)
install(DIRECTORY include/ DESTINATION include)
install(FILES "${METALROBO_METALLIB}" DESTINATION lib/metalrobo)
