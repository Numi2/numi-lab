cmake_minimum_required(VERSION 3.28)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(
        NumiMatter
        VERSION 0.2.0
        DESCRIPTION "Compiled heterogeneous matter runtime for Numi Lab"
        LANGUAGES CXX
    )
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(
    NUMI_MATTER_COMPILER_SOURCES
    src/matter.cpp
    src/language_impl.cpp
    src/constitutive_impl.cpp
    src/compiler_impl.cpp
    src/fingerprint_impl.cpp
    src/validation_impl.cpp
    src/package_impl.cpp
    src/surgical_tissue.cpp
)

# One canonical object set feeds both the standalone compiler library and the
# parent MetalRobo dylib. This prevents the integrated runtime from depending
# on archive extraction heuristics that can otherwise drop unreferenced public
# Matter entry points from a shared library.
add_library(numi_matter_compiler_objects OBJECT ${NUMI_MATTER_COMPILER_SOURCES})
target_include_directories(
    numi_matter_compiler_objects
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)
target_compile_options(
    numi_matter_compiler_objects
    PRIVATE -Wall -Wextra -Wpedantic
)

add_library(
    numi_matter_compiler
    STATIC
    $<TARGET_OBJECTS:numi_matter_compiler_objects>
)
target_include_directories(
    numi_matter_compiler
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

add_executable(numi-matterc tools/matterc.cpp)
target_link_libraries(numi-matterc PRIVATE numi_matter_compiler)
target_compile_options(numi-matterc PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(CTest)
add_executable(numi-matter-stateful-check tools/stateful_check.cpp)
target_link_libraries(numi-matter-stateful-check PRIVATE numi_matter_compiler)
target_compile_definitions(
    numi-matter-stateful-check
    PRIVATE
        NUMI_MATTER_STATEFUL_MATERIAL="${CMAKE_CURRENT_SOURCE_DIR}/materials/damageable_silicone.nmatter"
)
target_compile_options(
    numi-matter-stateful-check
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
if(BUILD_TESTING)
    add_test(
        NAME matter.compiler.stateful_roundtrip
        COMMAND $<TARGET_FILE:numi-matter-stateful-check>
    )
    set_tests_properties(
        matter.compiler.stateful_roundtrip
        PROPERTIES LABELS "matter;compiler;stateful" TIMEOUT 60
    )
endif()

if(APPLE)
    enable_language(OBJCXX)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)

    set(NUMI_MATTER_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(NUMI_MATTER_AIR "${NUMI_MATTER_SHADER_DIR}/NumiMatter.air")
    set(NUMI_MATTER_METALLIB "${NUMI_MATTER_SHADER_DIR}/NumiMatter.metallib")

    add_custom_command(
        OUTPUT "${NUMI_MATTER_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/src/matter_kernels.metal"
            -o "${NUMI_MATTER_AIR}"
        COMMAND xcrun -sdk macosx metallib
            "${NUMI_MATTER_AIR}"
            -o "${NUMI_MATTER_METALLIB}"
        DEPENDS
            src/matter_kernels.metal
            src/metal/common.metalinc
            src/metal/mpm.metalinc
            src/metal/fem.metalinc
            src/metal/fgmres.metalinc
            src/metal/mixed_fem.metalinc
            src/metal/topology_mutation.metalinc
            src/metal/contact.metalinc
            src/metal/adaptive.metalinc
            src/metal/identification_scheduler.metalinc
            include/numi/matter/shared.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM
    )
    add_custom_target(numi_matter_shaders ALL DEPENDS "${NUMI_MATTER_METALLIB}")

    add_library(
        numi_matter_runtime_objects
        OBJECT
        src/runtime.mm
        src/metal_world.cpp
    )
    add_dependencies(numi_matter_runtime_objects numi_matter_shaders)
    target_include_directories(
        numi_matter_runtime_objects
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    )
    target_compile_options(
        numi_matter_runtime_objects
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
            "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc;-Wall;-Wextra;-Wpedantic>"
    )
    target_compile_definitions(
        numi_matter_runtime_objects
        PRIVATE NUMI_MATTER_DEFAULT_METALLIB=\"${NUMI_MATTER_METALLIB}\"
    )

    add_library(
        numi_matter_runtime
        STATIC
        $<TARGET_OBJECTS:numi_matter_runtime_objects>
    )
    target_include_directories(
        numi_matter_runtime
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
    )
    target_link_libraries(
        numi_matter_runtime
        PUBLIC numi_matter_compiler
        PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}"
    )

    install(FILES "${NUMI_MATTER_METALLIB}" DESTINATION lib/numi-matter)
    install(TARGETS numi_matter_runtime ARCHIVE DESTINATION lib)
endif()

install(
    TARGETS numi_matter_compiler numi-matterc
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY materials/ DESTINATION share/numi-matter/materials)
