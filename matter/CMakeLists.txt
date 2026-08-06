cmake_minimum_required(VERSION 3.28)

project(
    NumiMatter
    VERSION 0.1.0
    DESCRIPTION "Compiled heterogeneous matter runtime for Numi Lab"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(numi_matter_compiler STATIC src/matter.cpp)
target_include_directories(
    numi_matter_compiler
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)
target_compile_options(
    numi_matter_compiler
    PRIVATE -Wall -Wextra -Wpedantic
)

add_executable(numi-matterc tools/matterc.cpp)
target_link_libraries(numi-matterc PRIVATE numi_matter_compiler)
target_compile_options(numi-matterc PRIVATE -Wall -Wextra -Wpedantic -Werror)

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
            include/numi/matter/shared.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
        VERBATIM
    )
    add_custom_target(numi_matter_shaders ALL DEPENDS "${NUMI_MATTER_METALLIB}")

    add_library(numi_matter_runtime STATIC src/runtime.mm)
    add_dependencies(numi_matter_runtime numi_matter_shaders)
    target_include_directories(
        numi_matter_runtime
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    )
    target_compile_options(
        numi_matter_runtime
        PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
    )
    target_link_libraries(
        numi_matter_runtime
        PUBLIC numi_matter_compiler
        PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}"
    )
    target_compile_definitions(
        numi_matter_runtime
        PRIVATE NUMI_MATTER_DEFAULT_METALLIB=\"${NUMI_MATTER_METALLIB}\"
    )

    install(FILES "${NUMI_MATTER_METALLIB}" DESTINATION lib/numi-matter)
    install(TARGETS numi_matter_runtime ARCHIVE DESTINATION lib)
endif()

install(TARGETS numi_matter_compiler numi-matterc
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY materials/ DESTINATION share/numi-matter/materials)
