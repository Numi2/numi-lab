cmake_minimum_required(VERSION 3.28)

project(
    MetalRobo
    VERSION 0.4.0
    DESCRIPTION "Metal-native robotics simulation and reinforcement learning"
    LANGUAGES C CXX OBJCXX
)

if(NOT APPLE)
    message(FATAL_ERROR "MetalRobo 0.4 targets Apple Silicon macOS.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(LibXml2 REQUIRED)

set(METALROBO_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(METALROBO_PHYSICS_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/Physics.metal"
)
set(METALROBO_CONSTRAINT_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ConstraintSolver.metal"
)
set(METALROBO_QUALITY_CONTACT_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalQualityContactSolver.metal"
)
set(METALROBO_UNIFIED_QUALITY_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalUnifiedQualitySolver.metal"
)
set(METALROBO_FREE_BODY_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/FreeBodyDynamics.metal"
)
set(METALROBO_COLLISION_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/Collision.metal"
)
set(METALROBO_BROADPHASE_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/DeterministicBroadphase.metal"
)
set(METALROBO_ARTICULATED_OPERATOR_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ArticulatedOperator.metal"
)
set(METALROBO_ARTICULATED_ABA_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ArticulatedABA.metal"
)
set(METALROBO_ARTICULATED_INVERSE_MASS_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ArticulatedInverseMass.metal"
)
set(METALROBO_WORLD_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalWorld.metal"
)
set(METALROBO_WORLD_CONTACT_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalWorldContact.metal"
)
set(METALROBO_WORLD_COMPILER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/WorldCompiler.metal"
)
set(METALROBO_HYBRID_RENDERER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/HybridRenderer.metal"
)
set(METALROBO_PHYSICS_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboPhysics.air"
)
set(METALROBO_CONSTRAINT_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboConstraintSolver.air"
)
set(METALROBO_QUALITY_CONTACT_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboQualityContactSolver.air"
)
set(METALROBO_UNIFIED_QUALITY_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboUnifiedQualitySolver.air"
)
set(METALROBO_FREE_BODY_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboFreeBodyDynamics.air"
)
set(METALROBO_COLLISION_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboCollision.air"
)
set(METALROBO_BROADPHASE_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboDeterministicBroadphase.air"
)
set(METALROBO_ARTICULATED_OPERATOR_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboArticulatedOperator.air"
)
set(METALROBO_ARTICULATED_ABA_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboArticulatedABA.air"
)
set(METALROBO_ARTICULATED_ABA_SMALL_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboArticulatedABASmall.air"
)
set(METALROBO_MULTI_ARTICULATED_ABA_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboMultiArticulatedABA.air"
)
set(METALROBO_DISCRETE_ELASTIC_ROD_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/DiscreteElasticRod.metal"
)
set(METALROBO_DISCRETE_ELASTIC_ROD_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboDiscreteElasticRod.air"
)
set(METALROBO_GENERALIZED_CONSTRAINT_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalGeneralizedConstraints.metal"
)
set(METALROBO_MULTI_ARTICULATED_CONTACT_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalMultiArticulatedContact.metal"
)
set(METALROBO_GENERALIZED_CONSTRAINT_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboGeneralizedConstraints.air"
)
set(METALROBO_MULTI_ARTICULATED_CONTACT_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboMultiArticulatedContact.air"
)
set(METALROBO_ARTICULATED_INVERSE_MASS_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboArticulatedInverseMass.air"
)
set(METALROBO_MULTI_ARTICULATED_INVERSE_MASS_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboMultiArticulatedInverseMass.air"
)
set(METALROBO_WORLD_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboWorld.air"
)
set(METALROBO_WORLD_CONTACT_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboWorldContact.air"
)
set(METALROBO_WORLD_COMPILER_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboWorldCompiler.air"
)
set(METALROBO_HYBRID_RENDERER_AIR
    "${METALROBO_SHADER_DIR}/MetalRoboHybridRenderer.air"
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
            -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_QUALITY_CONTACT_SOURCE}"
            -o "${METALROBO_QUALITY_CONTACT_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_UNIFIED_QUALITY_SOURCE}"
            -o "${METALROBO_UNIFIED_QUALITY_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_FREE_BODY_SOURCE}"
            -o "${METALROBO_FREE_BODY_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_COLLISION_SOURCE}"
            -o "${METALROBO_COLLISION_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_BROADPHASE_SOURCE}"
            -o "${METALROBO_BROADPHASE_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_ARTICULATED_OPERATOR_SOURCE}"
            -o "${METALROBO_ARTICULATED_OPERATOR_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_ARTICULATED_ABA_SOURCE}"
            -o "${METALROBO_ARTICULATED_ABA_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -DMR_ABA_KERNEL_NAME=mr_articulated_aba_step_small
            -DMR_ABA_COMPILED_MAX_BODIES=12
            -DMR_ABA_COMPILED_MAX_DOFS=16
            -DMR_ABA_COMPILED_MAX_Q=17
            -c "${METALROBO_ARTICULATED_ABA_SOURCE}"
            -o "${METALROBO_ARTICULATED_ABA_SMALL_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -DMR_ABA_KERNEL_NAME=mr_multi_articulated_aba_step
            -DMR_ABA_MULTI_ARTICULATION=1
            -c "${METALROBO_ARTICULATED_ABA_SOURCE}"
            -o "${METALROBO_MULTI_ARTICULATED_ABA_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_DISCRETE_ELASTIC_ROD_SOURCE}"
            -o "${METALROBO_DISCRETE_ELASTIC_ROD_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_GENERALIZED_CONSTRAINT_SOURCE}"
            -o "${METALROBO_GENERALIZED_CONSTRAINT_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_MULTI_ARTICULATED_CONTACT_SOURCE}"
            -o "${METALROBO_MULTI_ARTICULATED_CONTACT_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_ARTICULATED_INVERSE_MASS_SOURCE}"
            -o "${METALROBO_ARTICULATED_INVERSE_MASS_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -DMR_INVERSE_MASS_KERNEL_NAME=mr_multi_articulated_inverse_mass
            -DMR_INVERSE_MASS_MULTI_ARTICULATION=1
            -c "${METALROBO_ARTICULATED_INVERSE_MASS_SOURCE}"
            -o "${METALROBO_MULTI_ARTICULATED_INVERSE_MASS_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_WORLD_SOURCE}"
            -o "${METALROBO_WORLD_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_WORLD_CONTACT_SOURCE}"
            -o "${METALROBO_WORLD_CONTACT_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_WORLD_COMPILER_SOURCE}"
            -o "${METALROBO_WORLD_COMPILER_AIR}"
    COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${METALROBO_HYBRID_RENDERER_SOURCE}"
            -o "${METALROBO_HYBRID_RENDERER_AIR}"
    COMMAND xcrun -sdk macosx metallib
            "${METALROBO_PHYSICS_AIR}"
            "${METALROBO_CONSTRAINT_AIR}"
            "${METALROBO_QUALITY_CONTACT_AIR}"
            "${METALROBO_UNIFIED_QUALITY_AIR}"
            "${METALROBO_FREE_BODY_AIR}"
            "${METALROBO_COLLISION_AIR}"
            "${METALROBO_BROADPHASE_AIR}"
            "${METALROBO_ARTICULATED_OPERATOR_AIR}"
            "${METALROBO_ARTICULATED_ABA_AIR}"
            "${METALROBO_ARTICULATED_ABA_SMALL_AIR}"
            "${METALROBO_MULTI_ARTICULATED_ABA_AIR}"
            "${METALROBO_DISCRETE_ELASTIC_ROD_AIR}"
            "${METALROBO_GENERALIZED_CONSTRAINT_AIR}"
            "${METALROBO_MULTI_ARTICULATED_CONTACT_AIR}"
            "${METALROBO_ARTICULATED_INVERSE_MASS_AIR}"
            "${METALROBO_MULTI_ARTICULATED_INVERSE_MASS_AIR}"
            "${METALROBO_WORLD_AIR}"
            "${METALROBO_WORLD_CONTACT_AIR}"
            "${METALROBO_WORLD_COMPILER_AIR}"
            "${METALROBO_HYBRID_RENDERER_AIR}"
            -o "${METALROBO_METALLIB}"
    DEPENDS
        "${METALROBO_PHYSICS_SOURCE}"
        "${METALROBO_CONSTRAINT_SOURCE}"
        "${METALROBO_QUALITY_CONTACT_SOURCE}"
        "${METALROBO_UNIFIED_QUALITY_SOURCE}"
        "${METALROBO_FREE_BODY_SOURCE}"
        "${METALROBO_COLLISION_SOURCE}"
        "${METALROBO_BROADPHASE_SOURCE}"
        "${METALROBO_ARTICULATED_OPERATOR_SOURCE}"
        "${METALROBO_ARTICULATED_ABA_SOURCE}"
        "${METALROBO_DISCRETE_ELASTIC_ROD_SOURCE}"
        "${METALROBO_GENERALIZED_CONSTRAINT_SOURCE}"
        "${METALROBO_MULTI_ARTICULATED_CONTACT_SOURCE}"
        "${METALROBO_ARTICULATED_INVERSE_MASS_SOURCE}"
        "${METALROBO_WORLD_SOURCE}"
        "${METALROBO_WORLD_CONTACT_SOURCE}"
        "${METALROBO_WORLD_COMPILER_SOURCE}"
        "${METALROBO_HYBRID_RENDERER_SOURCE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/gpu_types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/engine_types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/constraint_ir_shared.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/quality_solver_shared.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/unified_quality_shared.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/parallel_aba_shared.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/rod_gpu_shared.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/generalized_constraint_shared.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/multi_contact_shared.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/world_compiler_types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/r2s2r_types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/hybrid_renderer_types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/visual_platform_types.h"
    VERBATIM
)

add_custom_target(metalrobo_shaders ALL DEPENDS "${METALROBO_METALLIB}")

add_library(
    metalrobo
    SHARED
    src/core/Model.cpp
    src/core/Franka.cpp
    src/core/FrankaEngine.cpp
    src/core/FrankaHand.cpp
    src/core/FrankaWorld.cpp
    src/core/CpuDynamics.cpp
    src/core/EngineModel.cpp
    src/core/EngineModelComposer.cpp
    src/core/ParallelABASchedule.cpp
    src/core/CompiledMetalMultiArticulatedProgram.cpp
    src/core/CompiledMetalMultiArticulatedContactProgram.cpp
    src/core/WorldCompiler.cpp
    src/core/WorldPack.cpp
    src/core/EpisodeTwinCompiler.cpp
    src/core/R2S2R.cpp
    src/core/VisualPlatform.cpp
    src/core/RobotDescriptionCooker.cpp
    src/core/GeometryCooker.cpp
    src/core/G1.cpp
    src/core/SurgicalPSM.cpp
    src/core/SurgicalAssets.cpp
    src/core/SurgicalWorld.cpp
    src/core/HeterogeneousWorld.cpp
    src/core/DiscreteElasticRod.cpp
    src/core/Collision.cpp
    src/core/FreeBodyDynamics.cpp
    src/core/ArticulatedDynamics.cpp
    src/core/ArticulatedActuation.cpp
    src/core/ArticulatedJointLimits.cpp
    src/core/ArticulatedContact.cpp
    src/core/CoupledArticulatedRigidContact.cpp
    src/core/ArticulatedRigidCollision.cpp
    src/core/ArticulatedRigidWorld.cpp
    src/core/ArticulatedMixedConstraints.cpp
    src/core/ArticulatedCollision.cpp
    src/core/ArticulatedWorld.cpp
    src/core/MultiArticulatedWorld.cpp
    src/core/MultiArticulatedContact.cpp
    src/core/ConstraintIR.cpp
    src/core/ReferenceConicSolver.cpp
    src/core/QualityContactSolver.cpp
    src/core/ConstraintSolver.cpp
    src/core/RigidBodyWorld.cpp
    src/apple/CaptureManifestJSON.mm
    src/apple/HardwareOutcomeJSON.mm
    src/metal/MetalArticulatedOperator.mm
    src/metal/MetalArticulatedABA.mm
    src/metal/MetalMultiArticulatedABA.mm
    src/metal/MetalDiscreteElasticRod.mm
    src/metal/MetalArticulatedInverseMass.mm
    src/metal/MetalMultiArticulatedInverseMass.mm
    src/metal/MetalMultiArticulatedConstraints.mm
    src/metal/MetalMultiArticulatedContact.mm
    src/metal/MetalQualityContactSolver.mm
    src/metal/MetalUnifiedQualitySolver.mm
    src/metal/MetalWorld.mm
    src/metal/MetalWorldFamily.mm
    src/metal/MetalHybridRenderer.mm
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
        "${ACCELERATE_FRAMEWORK}"
        SQLite::SQLite3
        LibXml2::LibXml2
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
        INSTALL_RPATH "@loader_path/../lib"
)

add_executable(
    metalrobo_metal_world_contact_probe
    apps/metal_world_contact_probe.cpp
)
target_link_libraries(
    metalrobo_metal_world_contact_probe
    PRIVATE metalrobo
)
set_target_properties(
    metalrobo_metal_world_contact_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
        INSTALL_RPATH "@loader_path/../lib"
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

add_executable(
    metalrobo_franka_engine_model_probe
    apps/franka_engine_model_probe.cpp
)
target_link_libraries(
    metalrobo_franka_engine_model_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_franka_engine_model_probe
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

add_executable(
    metalrobo_articulated_actuation_probe
    apps/articulated_actuation_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_actuation_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_actuation_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_joint_limits_probe
    apps/articulated_joint_limits_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_joint_limits_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_joint_limits_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_contact_probe
    apps/articulated_contact_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_contact_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_contact_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_mixed_constraints_probe
    apps/articulated_mixed_constraints_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_mixed_constraints_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_mixed_constraints_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_multi_articulated_world_probe
    apps/multi_articulated_world_probe.cpp
)
target_link_libraries(
    metalrobo_multi_articulated_world_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_multi_articulated_world_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_multi_articulated_contact_probe
    apps/multi_articulated_contact_probe.cpp
)
target_link_libraries(
    metalrobo_multi_articulated_contact_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_multi_articulated_contact_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_metal_multi_articulated_contact_probe
    apps/metal_multi_articulated_contact_probe.cpp
)
target_link_libraries(
    metalrobo_metal_multi_articulated_contact_probe
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_metal_multi_articulated_contact_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_metal_multi_articulated_contact_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_parallel_aba_schedule_probe
    apps/parallel_aba_schedule_probe.cpp
)
target_link_libraries(
    metalrobo_parallel_aba_schedule_probe
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_parallel_aba_schedule_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_parallel_aba_schedule_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_multi_articulated_aba_gpu_probe
    apps/multi_articulated_aba_gpu_probe.mm
)
target_link_libraries(
    metalrobo_multi_articulated_aba_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_multi_articulated_aba_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_multi_articulated_inverse_mass_gpu_probe
    apps/multi_articulated_inverse_mass_gpu_probe.mm
)
target_link_libraries(
    metalrobo_multi_articulated_inverse_mass_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_multi_articulated_inverse_mass_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_multi_articulated_constraint_gpu_probe
    apps/multi_articulated_constraint_gpu_probe.mm
)
target_link_libraries(
    metalrobo_multi_articulated_constraint_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_multi_articulated_constraint_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_metal_discrete_elastic_rod_probe
    apps/metal_discrete_elastic_rod_probe.mm
)
target_link_libraries(
    metalrobo_metal_discrete_elastic_rod_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_metal_discrete_elastic_rod_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_metal_unified_quality_probe
    apps/metal_unified_quality_probe.cpp
)
target_link_libraries(
    metalrobo_metal_unified_quality_probe
    PRIVATE metalrobo
)
target_compile_options(
    metalrobo_metal_unified_quality_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_metal_unified_quality_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
        INSTALL_RPATH "@loader_path/../lib"
)

add_executable(
    metalrobo_robot_description_cooker_probe
    apps/robot_description_cooker_probe.cpp
)
target_link_libraries(
    metalrobo_robot_description_cooker_probe
    PRIVATE
        metalrobo
)
target_compile_definitions(
    metalrobo_robot_description_cooker_probe
    PRIVATE
        METALROBO_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
set_target_properties(
    metalrobo_robot_description_cooker_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_dual_psm_world_probe
    apps/dual_psm_world_probe.cpp
)
target_link_libraries(
    metalrobo_dual_psm_world_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_dual_psm_world_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_discrete_elastic_rod_probe
    apps/discrete_elastic_rod_probe.cpp
)
target_link_libraries(
    metalrobo_discrete_elastic_rod_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_discrete_elastic_rod_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_engine_model_composer_probe
    apps/engine_model_composer_probe.cpp
)
target_link_libraries(
    metalrobo_engine_model_composer_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_engine_model_composer_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_heterogeneous_world_probe
    apps/heterogeneous_world_probe.cpp
)
target_link_libraries(
    metalrobo_heterogeneous_world_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_heterogeneous_world_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_metal_quality_contact_probe
    apps/metal_quality_contact_probe.mm
)
target_link_libraries(
    metalrobo_metal_quality_contact_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_metal_quality_contact_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_g1_collision_contact_probe
    apps/g1_collision_contact_probe.cpp
)
target_link_libraries(
    metalrobo_g1_collision_contact_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_g1_collision_contact_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_world_probe
    apps/articulated_world_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_world_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_world_probe
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

add_executable(
    metalrobo_constraint_ir_probe
    apps/constraint_ir_probe.cpp
)
target_link_libraries(
    metalrobo_constraint_ir_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_constraint_ir_probe
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

add_executable(
    metalrobo_deterministic_broadphase_probe
    apps/deterministic_broadphase_probe.mm
)
add_dependencies(
    metalrobo_deterministic_broadphase_probe
    metalrobo_shaders
)
target_compile_definitions(
    metalrobo_deterministic_broadphase_probe
    PRIVATE
        METALROBO_DEFAULT_METALLIB="${METALROBO_METALLIB}"
)
target_compile_options(
    metalrobo_deterministic_broadphase_probe
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
        -Wpedantic
)
target_link_libraries(
    metalrobo_deterministic_broadphase_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_deterministic_broadphase_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_operator_gpu_probe
    apps/articulated_operator_gpu_probe.mm
)
add_dependencies(
    metalrobo_articulated_operator_gpu_probe
    metalrobo_shaders
)
target_compile_definitions(
    metalrobo_articulated_operator_gpu_probe
    PRIVATE
        METALROBO_DEFAULT_METALLIB="${METALROBO_METALLIB}"
)
target_compile_options(
    metalrobo_articulated_operator_gpu_probe
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
        -Wpedantic
)
target_link_libraries(
    metalrobo_articulated_operator_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_articulated_operator_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_aba_gpu_probe
    apps/articulated_aba_gpu_probe.mm
)
add_dependencies(
    metalrobo_articulated_aba_gpu_probe
    metalrobo_shaders
)
target_compile_definitions(
    metalrobo_articulated_aba_gpu_probe
    PRIVATE
        METALROBO_ABA_METALLIB="${METALROBO_METALLIB}"
)
target_compile_options(
    metalrobo_articulated_aba_gpu_probe
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
        -Wpedantic
)
target_link_libraries(
    metalrobo_articulated_aba_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_articulated_aba_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_inverse_mass_gpu_probe
    apps/articulated_inverse_mass_gpu_probe.mm
)
add_dependencies(
    metalrobo_articulated_inverse_mass_gpu_probe
    metalrobo_shaders
)
target_compile_definitions(
    metalrobo_articulated_inverse_mass_gpu_probe
    PRIVATE
        METALROBO_INVERSE_MASS_METALLIB="${METALROBO_METALLIB}"
)
target_compile_options(
    metalrobo_articulated_inverse_mass_gpu_probe
    PRIVATE
        -fobjc-arc
        -Wall
        -Wextra
        -Wpedantic
)
target_link_libraries(
    metalrobo_articulated_inverse_mass_gpu_probe
    PRIVATE
        metalrobo
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
set_target_properties(
    metalrobo_articulated_inverse_mass_gpu_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_operator_host_probe
    apps/articulated_operator_host_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_operator_host_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_operator_host_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_operator_context_probe
    apps/articulated_operator_context_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_operator_context_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_operator_context_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_aba_host_probe
    apps/articulated_aba_host_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_aba_host_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_aba_host_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_metal_world_probe
    apps/metal_world_probe.cpp
)
target_link_libraries(
    metalrobo_metal_world_probe
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_metal_world_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_metal_world_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_inverse_mass_host_probe
    apps/articulated_inverse_mass_host_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_inverse_mass_host_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_inverse_mass_host_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_surgical_psm_probe
    apps/surgical_psm_probe.cpp
)
target_link_libraries(
    metalrobo_surgical_psm_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_surgical_psm_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_surgical_assets_probe
    apps/surgical_assets_probe.cpp
)
target_link_libraries(
    metalrobo_surgical_assets_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_surgical_assets_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_surgical_metal_operator_probe
    apps/surgical_metal_operator_probe.cpp
)
target_link_libraries(
    metalrobo_surgical_metal_operator_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_surgical_metal_operator_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_coupled_articulated_rigid_contact_probe
    apps/coupled_articulated_rigid_contact_probe.cpp
)
target_link_libraries(
    metalrobo_coupled_articulated_rigid_contact_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_coupled_articulated_rigid_contact_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_rigid_collision_probe
    apps/articulated_rigid_collision_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_rigid_collision_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_rigid_collision_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_articulated_rigid_world_probe
    apps/articulated_rigid_world_probe.cpp
)
target_link_libraries(
    metalrobo_articulated_rigid_world_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_articulated_rigid_world_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_supported_needle_pickup_probe
    apps/supported_needle_pickup_probe.cpp
)
target_link_libraries(
    metalrobo_supported_needle_pickup_probe
    PRIVATE
        metalrobo
)
set_target_properties(
    metalrobo_supported_needle_pickup_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_world_compiler_probe
    apps/world_compiler_probe.cpp
)
target_link_libraries(
    metalrobo_world_compiler_probe
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_world_compiler_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_world_compiler_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_metal_world_family_probe
    apps/metal_world_family_probe.cpp
)
target_link_libraries(
    metalrobo_metal_world_family_probe
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_metal_world_family_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_metal_world_family_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_r2s2r_loop_probe
    apps/r2s2r_loop_probe.cpp
)
target_link_libraries(
    metalrobo_r2s2r_loop_probe
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_r2s2r_loop_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_r2s2r_loop_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_world_pack
    apps/world_pack_tool.cpp
)
target_link_libraries(
    metalrobo_world_pack
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_world_pack
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_world_pack
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_episode_compile
    apps/episode_twin_compile.cpp
)
target_link_libraries(
    metalrobo_episode_compile
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_episode_compile
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_episode_compile
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_episode_twin_product_probe
    apps/episode_twin_product_probe.cpp
)
target_link_libraries(
    metalrobo_episode_twin_product_probe
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_episode_twin_product_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_episode_twin_product_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

add_executable(
    metalrobo_hybrid_renderer_probe
    apps/hybrid_renderer_probe.cpp
)
target_link_libraries(
    metalrobo_hybrid_renderer_probe
    PRIVATE
        metalrobo
)
target_compile_options(
    metalrobo_hybrid_renderer_probe
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
set_target_properties(
    metalrobo_hybrid_renderer_probe
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

install(
    TARGETS
        metalrobo
        metalrobo_bench
        metalrobo_world_pack
        metalrobo_episode_compile
)
install(DIRECTORY include/ DESTINATION include)
install(FILES "${METALROBO_METALLIB}" DESTINATION lib/metalrobo)
install(
    FILES schemas/capture_manifest.schema.json
    DESTINATION share/metalrobo/schemas
)
install(
    FILES
        THIRD_PARTY_NOTICES.md
        licenses/CISST_LICENSE.txt
    DESTINATION share/metalrobo
)
