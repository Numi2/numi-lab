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
add_executable(metalrobo_plane_contact_benchmark "${NUMI_CONTACT_ROOT}/apps/plane_contact_benchmark.mm")
target_compile_options(metalrobo_plane_contact_benchmark PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(metalrobo_plane_contact_benchmark PRIVATE numi_plane_contact_owner)
add_dependencies(metalrobo_plane_contact_benchmark numi_plane_contact_shader)
add_executable(numi_plane_contact_probe_scaled "${NUMI_CONTACT_ROOT}/apps/plane_contact_probe.mm")
target_compile_options(numi_plane_contact_probe_scaled PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(numi_plane_contact_probe_scaled PRIVATE numi_plane_contact_owner)
add_dependencies(numi_plane_contact_probe_scaled numi_plane_contact_shader)

add_executable(numi_midpoint_rounding_iteration "${NUMI_CONTACT_ROOT}/apps/midpoint_rounding_iteration.mm")
target_compile_options(numi_midpoint_rounding_iteration PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(numi_midpoint_rounding_iteration PRIVATE numi_plane_contact_owner)
add_executable(numi_midpoint_capture_analysis "${NUMI_CONTACT_ROOT}/apps/midpoint_capture_analysis.mm")
target_compile_options(numi_midpoint_capture_analysis PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(numi_midpoint_capture_analysis PRIVATE numi_plane_contact_owner)
add_executable(numi_midpoint_failure_capture "${NUMI_CONTACT_ROOT}/apps/midpoint_failure_capture.mm")
target_compile_options(numi_midpoint_failure_capture PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(numi_midpoint_failure_capture PRIVATE numi_plane_contact_owner)
add_dependencies(numi_midpoint_failure_capture numi_plane_contact_shader)
add_executable(numi_midpoint_dynamic_guard "${NUMI_CONTACT_ROOT}/apps/midpoint_dynamic_guard_probe.mm")
target_compile_options(numi_midpoint_dynamic_guard PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(numi_midpoint_dynamic_guard PRIVATE numi_plane_contact_owner)
add_dependencies(numi_midpoint_dynamic_guard numi_plane_contact_shader)
add_executable(numi_midpoint_retry_order "${NUMI_CONTACT_ROOT}/apps/midpoint_retry_order_probe.mm")
target_compile_options(numi_midpoint_retry_order PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(numi_midpoint_retry_order PRIVATE numi_plane_contact_owner)
add_dependencies(numi_midpoint_retry_order numi_plane_contact_shader)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(numi_be_precision_scope be_precision_scope.mm)
    target_compile_options(numi_be_precision_scope PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_link_libraries(numi_be_precision_scope PRIVATE numi_plane_contact_owner)
    add_dependencies(numi_be_precision_scope numi_plane_contact_shader)
    add_test(NAME numi.contact.be_precision_scope COMMAND numi_be_precision_scope "${_shader}")
    set_tests_properties(numi.contact.be_precision_scope PROPERTIES TIMEOUT 120)
    add_test(NAME numi.contact.be_precision_step COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/be_precision_step_regression.py"
        "$<TARGET_FILE:numi_plane_contact_probe_scaled>" "${_shader}")
    set_tests_properties(numi.contact.be_precision_step PROPERTIES TIMEOUT 180)
    add_executable(numi_midpoint_diagnostic_scope midpoint_diagnostic_scope.mm)
    target_compile_options(numi_midpoint_diagnostic_scope PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_link_libraries(numi_midpoint_diagnostic_scope PRIVATE numi_plane_contact_owner)
    add_dependencies(numi_midpoint_diagnostic_scope numi_plane_contact_shader)
    add_test(NAME numi.contact.midpoint_diagnostic_scope COMMAND numi_midpoint_diagnostic_scope "${_shader}")
    set_tests_properties(numi.contact.midpoint_diagnostic_scope PROPERTIES TIMEOUT 120)
    add_executable(numi_contact_policy policy.cpp)
    target_link_libraries(numi_contact_policy PRIVATE numi_plane_contact_owner)
    add_executable(numi_contact_plane_area plane_area.cpp)
    target_link_libraries(numi_contact_plane_area PRIVATE numi_plane_contact_owner)
    add_test(NAME numi.contact.plane_area COMMAND numi_contact_plane_area)
    # Independent FP64 fixtures are reference-only. They do not execute or
    # qualify a Matter midpoint integration path.
    add_executable(numi_contact_fem_fp64_reference fem_plane_fp64_reference.cpp)
    add_executable(numi_contact_fem_slider_fp64_reference fem_slider_fp64_reference.cpp)
    add_executable(numi_contact_fem_slider_fp64_midpoint fem_slider_fp64_midpoint.cpp)
    add_test(NAME numi.contact.slider_fp64_midpoint COMMAND numi_contact_fem_slider_fp64_midpoint 0.000015625 0.01)
    add_test(NAME numi.contact.slider_fp64_midpoint_near COMMAND numi_contact_fem_slider_fp64_midpoint 0.000015625 0.01 0.000095)
    add_executable(numi_contact_slider_midpoint_discrete fem_slider_midpoint_discrete.cpp)
    add_test(NAME numi.contact.slider_midpoint_discrete COMMAND numi_contact_slider_midpoint_discrete)
    add_executable(numi_contact_fem_fp64_reference_compact fem_plane_fp64_reference_compact.cpp)
    add_test(NAME numi.contact.reference_compact_parity COMMAND numi_contact_fem_fp64_reference_compact --parity)
    add_executable(numi_contact_fem_fp64_reference_parallel fem_plane_fp64_reference_parallel.cpp)
    target_compile_definitions(numi_contact_fem_fp64_reference_parallel PRIVATE NUMI_FEM_REFERENCE_MAX_DIVISIONS=64)
    add_executable(numi_contact_fem_fp64_reference_n128 fem_plane_fp64_reference_parallel.cpp)
    target_compile_definitions(numi_contact_fem_fp64_reference_n128 PRIVATE NUMI_FEM_REFERENCE_MAX_DIVISIONS=128)
    add_executable(numi_contact_fem_fp64_reference_n128_8 fem_plane_fp64_reference_parallel.cpp)
    target_compile_definitions(numi_contact_fem_fp64_reference_n128_8 PRIVATE NUMI_FEM_REFERENCE_MAX_DIVISIONS=128 NUMI_REFERENCE_WORKERS=8)
    add_executable(numi_contact_fem_fp64_reference_n64 fem_plane_fp64_reference.cpp)
    target_compile_definitions(numi_contact_fem_fp64_reference_n64 PRIVATE NUMI_FEM_REFERENCE_MAX_DIVISIONS=64)
    add_test(NAME numi.contact.fem_fp64_energy_gradient
        COMMAND numi_contact_fem_fp64_reference --selftest)
    add_executable(numi_contact_fem_fp64_midpoint fem_plane_fp64_midpoint.cpp)
    add_test(NAME numi.contact.fem_fp64_midpoint_discrete
        COMMAND numi_contact_fem_fp64_midpoint --discrete-check)
    add_executable(numi_contact_midpoint_balance midpoint_balance_test.cpp)
    add_test(NAME numi.contact.midpoint_balance COMMAND numi_contact_midpoint_balance)
    add_executable(numi_contact_midpoint_endpoint_metal midpoint_endpoint_metal.mm)
    target_include_directories(numi_contact_midpoint_endpoint_metal PRIVATE "${_matter}/include")
    target_compile_options(numi_contact_midpoint_endpoint_metal PRIVATE -fobjc-arc)
    target_link_libraries(numi_contact_midpoint_endpoint_metal PRIVATE
        "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    add_dependencies(numi_contact_midpoint_endpoint_metal numi_plane_contact_shader)
    add_test(NAME numi.contact.midpoint_endpoint_metal
        COMMAND numi_contact_midpoint_endpoint_metal "${_shader}")
    set_tests_properties(numi.contact.midpoint_endpoint_metal PROPERTIES TIMEOUT 120)
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
    add_test(NAME numi.contact.midpoint_precontact_repair COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/midpoint_precontact_repair_regression.py"
        "$<TARGET_FILE:numi_plane_contact_probe_scaled>" "${_shader}")
    set_tests_properties(numi.contact.midpoint_precontact_repair PROPERTIES TIMEOUT 180 ENVIRONMENT "NM_MATTER_MIDPOINT_PRECONTACT_POLISH=1")
    add_test(NAME numi.contact.midpoint_native_transaction COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/midpoint_native_regression.py"
        "$<TARGET_FILE:metalrobo_plane_contact_probe>" "${_shader}")
    set_tests_properties(numi.contact.midpoint_native_transaction PROPERTIES TIMEOUT 180)
    add_test(NAME numi.contact.midpoint_dynamic_guard COMMAND numi_midpoint_dynamic_guard "${_shader}")
    set_tests_properties(numi.contact.midpoint_dynamic_guard PROPERTIES TIMEOUT 180)
    add_test(NAME numi.contact.midpoint_retry_order COMMAND numi_midpoint_retry_order "${_shader}")
    set_tests_properties(numi.contact.midpoint_retry_order PROPERTIES TIMEOUT 180)
    add_test(NAME numi.contact.midpoint_stage_retry_order COMMAND numi_midpoint_retry_order "${_shader}")
    set_tests_properties(numi.contact.midpoint_stage_retry_order PROPERTIES TIMEOUT 180 ENVIRONMENT "NM_MATTER_MIDPOINT_VELOCITY_COMPENSATION=1;NM_MATTER_MIDPOINT_STAGE_COMPENSATION=1")
endif()

add_executable(numi_midpoint_stage_correction_diagnostic "${NUMI_CONTACT_ROOT}/apps/midpoint_stage_correction_diagnostic.mm")
target_compile_options(numi_midpoint_stage_correction_diagnostic PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(numi_midpoint_stage_correction_diagnostic PRIVATE numi_plane_contact_owner)

add_executable(numi_be_fp64_capture_analysis "${NUMI_CONTACT_ROOT}/apps/be_fp64_capture_analysis.mm")
target_compile_options(numi_be_fp64_capture_analysis PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_link_libraries(numi_be_fp64_capture_analysis PRIVATE numi_plane_contact_owner)
