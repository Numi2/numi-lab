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
    src/vascular_impl.cpp
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
add_executable(numi-matter-cardiac-material-check tools/cardiac_material_check.cpp)
target_link_libraries(numi-matter-cardiac-material-check PRIVATE numi_matter_compiler)
target_compile_options(numi-matter-cardiac-material-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardiac_material_source
        COMMAND $<TARGET_FILE:numi-matter-cardiac-material-check>
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_ventricular_guccione_passive.nmatter"
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter")
    set_tests_properties(matter.compiler.cardiac_material_source PROPERTIES
        LABELS "matter;compiler;human;cardiac;material" TIMEOUT 120)
endif()

add_executable(numi-matter-fem-material-frame-compiler-check tools/fem_material_frame_compiler_check.cpp)
target_link_libraries(numi-matter-fem-material-frame-compiler-check PRIVATE numi_matter_compiler)
target_compile_options(numi-matter-fem-material-frame-compiler-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.fem_material_frame COMMAND numi-matter-fem-material-frame-compiler-check)
    set_tests_properties(matter.compiler.fem_material_frame PROPERTIES LABELS "matter;material;human" TIMEOUT 60)
endif()

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
            src/metal/vascular.metalinc
            src/metal/human_equality.metalinc
            src/metal/human_limits.metalinc
            src/metal/mixed_fem.metalinc
            src/metal/topology_mutation.metalinc
            src/metal/contact.metalinc
            src/metal/adaptive.metalinc
            src/metal/identification_scheduler.metalinc
            src/metal/numi_human.metalinc
            src/metal/accepted_state_proof.metalinc
            include/numi/matter/accepted_state_proof_gpu.h
            include/numi/matter/shared.h
            include/numi/matter/fiber_exp_linear.h
            include/numi/matter/human_equality_gpu.h
            include/numi/matter/human_limits_gpu.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM
    )
    add_custom_target(numi_matter_shaders ALL DEPENDS "${NUMI_MATTER_METALLIB}")

    add_library(
        numi_matter_runtime_objects
        OBJECT
        src/runtime.mm
        src/human_physiology.mm
        src/metal_world.cpp
        src/numi_human.mm
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

# Source-fibre law: FP64 independent oracle and actual Metal bytecode parity.
if(APPLE)
    set(NUMI_FIBER_CHECK_AIR "${CMAKE_CURRENT_BINARY_DIR}/shaders/fiber_exp_linear_check.air")
    set(NUMI_FIBER_CHECK_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/shaders/fiber_exp_linear_check.metallib")
    add_custom_command(OUTPUT "${NUMI_FIBER_CHECK_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/fiber_exp_linear_check.metal" -o "${NUMI_FIBER_CHECK_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_FIBER_CHECK_AIR}" -o "${NUMI_FIBER_CHECK_METALLIB}"
        DEPENDS tools/fiber_exp_linear_check.metal src/metal/common.metalinc
            include/numi/matter/shared.h include/numi/matter/fiber_exp_linear.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM)
    add_custom_target(numi_fiber_check_shaders DEPENDS "${NUMI_FIBER_CHECK_METALLIB}")
    add_executable(numi-matter-fiber-check tools/fiber_exp_linear_check.mm)
    add_dependencies(numi-matter-fiber-check numi_fiber_check_shaders)
    target_link_libraries(numi-matter-fiber-check PRIVATE numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-fiber-check PRIVATE
        NUMI_FIBER_CHECK_METALLIB="${NUMI_FIBER_CHECK_METALLIB}"
        NUMI_FIBER_SOURCE_MATERIAL="${CMAKE_CURRENT_SOURCE_DIR}/materials/open_knee_ligament_febio_exp_linear.nmatter")
    target_compile_options(numi-matter-fiber-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.source_fiber COMMAND numi-matter-fiber-check --cpu-only)
        add_test(NAME matter.metal.source_fiber COMMAND numi-matter-fiber-check)
        set_tests_properties(matter.compiler.source_fiber matter.metal.source_fiber
            PROPERTIES LABELS "matter;material;human" TIMEOUT 60)
    endif()
endif()

# HumanPack physiology uses the existing Matter package/runtime authority.
add_executable(numi-matter-vascular-compiler-check tools/vascular_compiler_check.cpp)
target_link_libraries(numi-matter-vascular-compiler-check PRIVATE numi_matter_compiler)
target_compile_options(numi-matter-vascular-compiler-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.vascular COMMAND numi-matter-vascular-compiler-check)
    set_tests_properties(matter.compiler.vascular PROPERTIES LABELS "matter;compiler;human;vascular" TIMEOUT 60)
endif()
if(APPLE)
    add_executable(numi-matter-physiologyc tools/physiologyc.mm)
    target_link_libraries(numi-matter-physiologyc PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}")
    target_compile_options(numi-matter-physiologyc PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    install(TARGETS numi-matter-physiologyc RUNTIME DESTINATION bin)
    add_executable(numi-matter-vascular-check tools/vascular_check.mm)
    target_link_libraries(numi-matter-vascular-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-vascular-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-vascular-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}"
        NUMI_MATTER_MATERIAL="${CMAKE_CURRENT_SOURCE_DIR}/materials/silicone.nmatter")
    target_compile_options(numi-matter-vascular-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.vascular COMMAND numi-matter-vascular-check)
        set_tests_properties(matter.metal.vascular PROPERTIES LABELS "matter;metal;human;vascular" TIMEOUT 60)
        add_test(NAME matter.metal.human_physiology_payload COMMAND numi-matter-vascular-check
            --human-input "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/human_physiology.native.v1.json")
        set_tests_properties(matter.metal.human_physiology_payload PROPERTIES LABELS "matter;metal;human;vascular" TIMEOUT 60)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME matter.compiler.human_physiology_admission
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/physiology_input_check.py"
                "$<TARGET_FILE:numi-matter-physiologyc>" "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/human_physiology.native.v1.json"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/shi-hose.native.v2.json")
            set_tests_properties(matter.compiler.human_physiology_admission PROPERTIES LABELS "matter;compiler;human;vascular" TIMEOUT 60)
        endif()
    endif()
endif()

if(APPLE)
    add_executable(numi-matter-cardiac-check tools/cardiac_check.mm)
    target_link_libraries(numi-matter-cardiac-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardiac-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardiac-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardiac-check PRIVATE -fobjc-arc -Wall -Wextra -Werror)
endif()

if(APPLE)
    add_executable(numi-matter-cvsim-check tools/cvsim_check.mm)
    target_link_libraries(numi-matter-cvsim-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cvsim-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cvsim-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cvsim-check PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cvsim_source COMMAND numi-matter-cvsim-check
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/cvsim21.native.v3.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/cvsim21-continuous-reference-64ms.csv")
        set_tests_properties(matter.metal.cvsim_source PROPERTIES LABELS "matter;metal;human;vascular" TIMEOUT 60)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(NAME matter.compiler.cvsim_admission
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cvsim_input_check.py"
                "$<TARGET_FILE:numi-matter-physiologyc>" "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/cvsim21.native.v3.json"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/shi-hose.native.v2.json")
            set_tests_properties(matter.compiler.cvsim_admission PROPERTIES LABELS "matter;compiler;human;vascular" TIMEOUT 60)
        endif()
    endif()
endif()

if(APPLE)
    add_executable(numi-matter-cardiac-transaction-check tools/cardiac_transaction_check.mm)
    target_link_libraries(numi-matter-cardiac-transaction-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardiac-transaction-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardiac-transaction-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardiac-transaction-check PRIVATE -fobjc-arc -Wall -Wextra -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardiac_transaction COMMAND numi-matter-cardiac-transaction-check)
        set_tests_properties(matter.metal.cardiac_transaction PROPERTIES LABELS "matter;metal;human;vascular" TIMEOUT 60)
        add_test(NAME matter.metal.cardiac_source COMMAND numi-matter-cardiac-check
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/shi-hose.native.v2.json" --steps 64 --dt .001)
        set_tests_properties(matter.metal.cardiac_source PROPERTIES LABELS "matter;metal;human;vascular" TIMEOUT 120)
    endif()
endif()

if(APPLE)
    add_executable(numi-matter-fem-material-frame-check tools/fem_material_frame_check.mm)
    target_link_libraries(numi-matter-fem-material-frame-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-fem-material-frame-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-fem-material-frame-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fem-material-frame-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.fem_material_frame COMMAND numi-matter-fem-material-frame-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_ventricular_guccione_passive.nmatter")
        set_tests_properties(matter.metal.fem_material_frame PROPERTIES LABELS "matter;material;human" TIMEOUT 180)
    endif()

    add_executable(numi-matter-vascular-cavity-check tools/vascular_cavity_check.mm)
    target_link_libraries(numi-matter-vascular-cavity-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-vascular-cavity-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-vascular-cavity-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-vascular-cavity-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.vascular_cavity COMMAND numi-matter-vascular-cavity-check)
        set_tests_properties(matter.metal.vascular_cavity PROPERTIES LABELS "matter;metal;human;vascular;cavity" TIMEOUT 120)
    endif()
endif()
