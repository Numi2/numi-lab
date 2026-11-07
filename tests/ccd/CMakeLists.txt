cmake_minimum_required(VERSION 3.28)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(NumiCCDProbe LANGUAGES CXX OBJCXX)
    include(CTest)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
endif()
get_filename_component(NUMI_CCD_ROOT "../.." ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(NUMI_CCD_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Focused production CCD instrument: compiling one owner shader avoids building
# the complete renderer/training shader graph for a collision-bound experiment.
set(NUMI_CCD_PROBE_AIR "${NUMI_CCD_SHADER_DIR}/NumiCCDProbe.air")
set(NUMI_CCD_PROBE_METALLIB "${NUMI_CCD_SHADER_DIR}/NumiCCDProbe.metallib")
add_custom_command(
    OUTPUT "${NUMI_CCD_PROBE_AIR}" "${NUMI_CCD_PROBE_METALLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_CCD_SHADER_DIR}"
    COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3
        -I "${NUMI_CCD_ROOT}/include"
        -c "${NUMI_CCD_ROOT}/src/metal/Collision.metal" -o "${NUMI_CCD_PROBE_AIR}"
    COMMAND xcrun -sdk macosx metallib "${NUMI_CCD_PROBE_AIR}"
        -o "${NUMI_CCD_PROBE_METALLIB}"
    DEPENDS "${NUMI_CCD_ROOT}/src/metal/Collision.metal"
        
        "${NUMI_CCD_ROOT}/include/metalrobo/engine_types.h"
        "${NUMI_CCD_ROOT}/include/metalrobo/rod_gpu_shared.h"
        "${NUMI_CCD_ROOT}/include/metalrobo/gpu_types.h"
        "${NUMI_CCD_ROOT}/include/metalrobo/compensated_translation_gpu.h"
        "${NUMI_CCD_ROOT}/include/metalrobo/constraint_ir_shared.h"
    VERBATIM)
add_custom_target(numi_ccd_probe_shader DEPENDS "${NUMI_CCD_PROBE_METALLIB}")
add_executable(metalrobo_ccd_advancement_probe "${NUMI_CCD_ROOT}/apps/ccd_advancement_probe.mm")
target_include_directories(metalrobo_ccd_advancement_probe PRIVATE "${NUMI_CCD_ROOT}/include")
target_compile_options(metalrobo_ccd_advancement_probe PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(metalrobo_ccd_advancement_probe PRIVATE
    "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
add_dependencies(metalrobo_ccd_advancement_probe numi_ccd_probe_shader)
set_target_properties(metalrobo_ccd_advancement_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
if(BUILD_TESTING)
    add_test(NAME metalrobo.ccd.directional_advancement
        COMMAND metalrobo_ccd_advancement_probe "${NUMI_CCD_PROBE_METALLIB}" 1 32 3)
    set_tests_properties(metalrobo.ccd.directional_advancement PROPERTIES
        LABELS "metalrobo;collision;ccd;metal;experimental" TIMEOUT 120)
    add_test(NAME metalrobo.ccd.small_scale_regression
        COMMAND metalrobo_ccd_advancement_probe "${NUMI_CCD_PROBE_METALLIB}" 0.1 32 3)
    set_tests_properties(metalrobo.ccd.small_scale_regression PROPERTIES
        LABELS "metalrobo;collision;ccd;metal;experimental" TIMEOUT 120)
endif()
