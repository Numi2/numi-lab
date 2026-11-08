cmake_minimum_required(VERSION 3.28)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(NumiPlaneContactProbe LANGUAGES CXX OBJCXX)
    include(CTest)
endif()
get_filename_component(NUMI_CONTACT_ROOT "../.." ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
set(_matter "${NUMI_CONTACT_ROOT}/matter")
set(_shader "${CMAKE_CURRENT_BINARY_DIR}/NumiMatter.metallib")
add_custom_command(OUTPUT "${_shader}"
    COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
        -I "${_matter}/include" -I "${NUMI_CONTACT_ROOT}/include"
        -MMD -MF "${CMAKE_CURRENT_BINARY_DIR}/matter.d" -MT "${_shader}"
        -c "${_matter}/src/matter_kernels.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/matter.air"
    COMMAND xcrun -sdk macosx metallib "${CMAKE_CURRENT_BINARY_DIR}/matter.air" -o "${_shader}"
    DEPENDS "${_matter}/src/matter_kernels.metal"
    DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/matter.d" VERBATIM)
add_custom_target(numi_plane_contact_shader DEPENDS "${_shader}")
set(_compiler matter language_impl constitutive_impl compiler_impl vascular_impl
    fingerprint_impl validation_impl package_impl surgical_tissue)
set(_sources)
foreach(_source IN LISTS _compiler)
    list(APPEND _sources "${_matter}/src/${_source}.cpp")
endforeach()
add_library(numi_plane_contact_owner STATIC ${_sources}
    "${_matter}/src/runtime.mm" "${_matter}/src/human_physiology.mm"
    "${_matter}/src/numi_human.mm")
target_include_directories(numi_plane_contact_owner PUBLIC
    "${_matter}/include" "${NUMI_CONTACT_ROOT}/include")
target_compile_options(numi_plane_contact_owner PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
target_link_libraries(numi_plane_contact_owner PUBLIC
    "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
add_executable(metalrobo_plane_contact_probe "${NUMI_CONTACT_ROOT}/apps/plane_contact_probe.mm")
target_compile_options(metalrobo_plane_contact_probe PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(metalrobo_plane_contact_probe PRIVATE numi_plane_contact_owner)
add_dependencies(metalrobo_plane_contact_probe numi_plane_contact_shader)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(numi_contact_policy policy.cpp)
    target_link_libraries(numi_contact_policy PRIVATE numi_plane_contact_owner)
    add_executable(numi_contact_plane_area plane_area.cpp)
    target_link_libraries(numi_contact_plane_area PRIVATE numi_plane_contact_owner)
    add_test(NAME numi.contact.plane_area COMMAND numi_contact_plane_area)
    # Independent FP64 fixtures are reference-only. They do not execute or
    # qualify a Matter midpoint integration path.
    add_executable(numi_contact_fem_fp64_reference fem_plane_fp64_reference.cpp)
    add_test(NAME numi.contact.fem_fp64_energy_gradient
        COMMAND numi_contact_fem_fp64_reference --selftest)
    add_executable(numi_contact_fem_fp64_midpoint fem_plane_fp64_midpoint.cpp)
    add_test(NAME numi.contact.fem_fp64_midpoint_discrete
        COMMAND numi_contact_fem_fp64_midpoint --discrete-check)
    add_executable(numi_contact_midpoint_transaction midpoint_transaction_cpu.cpp)
    add_test(NAME numi.contact.midpoint_transaction
        COMMAND numi_contact_midpoint_transaction)
    add_test(NAME numi.contact.fem_fp64_midpoint_reference
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/midpoint_regression.py"
            "$<TARGET_FILE:numi_contact_fem_fp64_midpoint>")
    add_test(NAME numi.contact.physical_policy COMMAND numi_contact_policy)
    add_test(NAME numi.contact.residual_restore COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/control_regression.py"
        "$<TARGET_FILE:metalrobo_plane_contact_probe>" "${_shader}"
        "${CMAKE_CURRENT_BINARY_DIR}/residual-regression-evidence")
    set_tests_properties(numi.contact.residual_restore PROPERTIES TIMEOUT 180)
endif()
