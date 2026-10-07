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
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME matter.analysis.cardboard_convergence
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_convergence.py")
        add_test(NAME matter.analysis.cardboard_tooling
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_tooling_analysis.py")
        add_test(NAME matter.analysis.cardboard_newton_trace
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_newton_trace.py")
        set_tests_properties(matter.analysis.cardboard_convergence matter.analysis.cardboard_tooling matter.analysis.cardboard_newton_trace
            PROPERTIES LABELS "matter;cardboard;analysis" TIMEOUT 60)
    endif()
endif()
add_executable(numi-matter-cardboard-tooling-check tools/cardboard_tooling_check.cpp)
target_link_libraries(numi-matter-cardboard-tooling-check PRIVATE numi_matter_compiler)
target_include_directories(numi-matter-cardboard-tooling-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-tooling-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_tooling COMMAND numi-matter-cardboard-tooling-check)
    set_tests_properties(matter.compiler.cardboard_tooling PROPERTIES LABELS "matter;cardboard;geometry" TIMEOUT 30)
endif()
add_executable(numi-matter-cardboard-barrier-scale-check tools/cardboard_prescribed_barrier_scale_check.cpp)
target_link_libraries(numi-matter-cardboard-barrier-scale-check PRIVATE numi_matter_compiler)
target_include_directories(numi-matter-cardboard-barrier-scale-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-barrier-scale-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_barrier_scale
        COMMAND numi-matter-cardboard-barrier-scale-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/damageable_silicone.nmatter")
    set_tests_properties(matter.compiler.cardboard_barrier_scale
        PROPERTIES LABELS "matter;cardboard;contact;fingerprint" TIMEOUT 30)
endif()
add_executable(numi-matter-cardboard-profile-check tools/cardboard_profile_check.cpp)
target_compile_options(numi-matter-cardboard-profile-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_profile COMMAND numi-matter-cardboard-profile-check)
    set_tests_properties(matter.compiler.cardboard_profile PROPERTIES LABELS "matter;cardboard;geometry" TIMEOUT 30)
endif()
add_executable(numi-matter-cardboard-glue-mesh-check tools/cardboard_glue_mesh_check.cpp)
target_compile_options(numi-matter-cardboard-glue-mesh-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_glue_mesh COMMAND numi-matter-cardboard-glue-mesh-check)
    set_tests_properties(matter.compiler.cardboard_glue_mesh PROPERTIES LABELS "matter;cardboard;geometry" TIMEOUT 60)
endif()
add_executable(numi-matter-cardboard-box-blank-check tools/cardboard_box_blank_check.cpp)
target_compile_options(numi-matter-cardboard-box-blank-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_box_blank COMMAND numi-matter-cardboard-box-blank-check)
    set_tests_properties(matter.compiler.cardboard_box_blank PROPERTIES LABELS "matter;cardboard;geometry" TIMEOUT 60)
endif()
add_executable(numi-matter-cardboard-paper-material-check tools/cardboard_paper_material_check.cpp)
target_link_libraries(numi-matter-cardboard-paper-material-check PRIVATE numi_matter_compiler)
target_compile_options(numi-matter-cardboard-paper-material-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_paper COMMAND numi-matter-cardboard-paper-material-check)
    set_tests_properties(matter.compiler.cardboard_paper PROPERTIES LABELS "matter;cardboard;material" TIMEOUT 120)
endif()
add_executable(numi-matter-cardboard-delamination-reference-check
    tools/cardboard_bleached_delamination_check.cpp)
target_compile_options(numi-matter-cardboard-delamination-reference-check
    PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_delamination_reference
        COMMAND numi-matter-cardboard-delamination-reference-check)
    set_tests_properties(matter.compiler.cardboard_delamination_reference
        PROPERTIES LABELS "matter;cardboard;delamination;reference" TIMEOUT 30)
endif()
add_executable(numi-matter-cardboard-material-check tools/cardboard_material_check.cpp)
target_link_libraries(numi-matter-cardboard-material-check PRIVATE numi_matter_compiler)
target_compile_options(numi-matter-cardboard-material-check PRIVATE -Wall -Wextra -Wpedantic)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_material
        COMMAND numi-matter-cardboard-material-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_liner_orthotropic.nmatter"
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter"
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_rct_elastic.nmatter")
    set_tests_properties(matter.compiler.cardboard_material PROPERTIES
        LABELS "matter;compiler;cardboard;material" TIMEOUT 120)
endif()

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
    set(NUMI_MATTER_PHYSICAL_DIGEST_AIR
        "${NUMI_MATTER_SHADER_DIR}/NumiMatterPhysicalStateDigest.air")
    set(NUMI_MATTER_PHYSICAL_DIGEST_METALLIB
        "${NUMI_MATTER_SHADER_DIR}/NumiMatterPhysicalStateDigest.metallib")

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
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_translation_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_geometry_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/numi/matter/human_support_precision_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${NUMI_MATTER_PHYSICAL_DIGEST_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal
            -std=metal4.0
            -O3
            -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/src/physical_state_digest_kernels.metal"
            -o "${NUMI_MATTER_PHYSICAL_DIGEST_AIR}"
        COMMAND xcrun -sdk macosx metallib
            "${NUMI_MATTER_PHYSICAL_DIGEST_AIR}"
            -o "${NUMI_MATTER_PHYSICAL_DIGEST_METALLIB}"
        DEPENDS
            src/physical_state_digest_kernels.metal
            src/metal/physical_state_digest.metalinc
            src/metal/sha256.metalinc
            include/numi/matter/physical_state_digest_gpu.h
            include/numi/matter/sha256_gpu.h
            include/numi/matter/accepted_state_proof_gpu.h
            include/numi/matter/shared.h
        VERBATIM
    )
    add_custom_target(numi_matter_shaders ALL
        DEPENDS
            "${NUMI_MATTER_METALLIB}"
            "${NUMI_MATTER_PHYSICAL_DIGEST_METALLIB}")

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
    add_executable(numi-matter-fem-kinematic-release-check tools/fem_kinematic_release_check.mm)
    target_link_libraries(numi-matter-fem-kinematic-release-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-fem-kinematic-release-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-fem-kinematic-release-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fem-kinematic-release-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.fem_kinematic_release COMMAND numi-matter-fem-kinematic-release-check)
        set_tests_properties(matter.metal.fem_kinematic_release PROPERTIES LABELS "matter;fem;release" TIMEOUT 120)
    endif()
    add_executable(numi-matter-cardboard-tooling-native-check tools/cardboard_tooling_native_check.mm)
    target_link_libraries(numi-matter-cardboard-tooling-native-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-tooling-native-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-tooling-native-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-tooling-native-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_tooling COMMAND numi-matter-cardboard-tooling-native-check)
        set_tests_properties(matter.metal.cardboard_tooling PROPERTIES LABELS "matter;cardboard;contact" TIMEOUT 180)
    endif()
    add_executable(numi-matter-cardboard-probe tools/cardboard_probe.mm)
    target_link_libraries(numi-matter-cardboard-probe PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_options(numi-matter-cardboard-probe PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_compile_definitions(numi-matter-cardboard-probe PRIVATE
        NUMI_CARDBOARD_MATERIAL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/materials"
        NUMI_CARDBOARD_METALLIB="${NUMI_MATTER_METALLIB}")
    add_executable(numi-matter-cardboard-gpu-check tools/cardboard_gpu_check.mm)
    target_link_libraries(numi-matter-cardboard-gpu-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-gpu-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_options(numi-matter-cardboard-gpu-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_compile_definitions(numi-matter-cardboard-gpu-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_material
            COMMAND numi-matter-cardboard-gpu-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_liner_orthotropic.nmatter"
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter")
        set_tests_properties(matter.metal.cardboard_material PROPERTIES
            LABELS "matter;metal;cardboard;material" TIMEOUT 120)
    endif()
    set(NUMI_HUMAN_RESPIRATION_METALLIB "${NUMI_MATTER_SHADER_DIR}/HumanRespiration.metallib")
    set(NUMI_BRAIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../numi-brain" CACHE PATH "NumiBrain source for the accepted respiratory chemoreflex")
    add_custom_command(
        OUTPUT "${NUMI_HUMAN_RESPIRATION_METALLIB}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../src/metal"
            -I "${NUMI_BRAIN_ROOT}/Sources/NumiBrainABI/include"
            -I "${NUMI_BRAIN_ROOT}/Sources/NumiBrainMetal/Shaders"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/src/human_respiration.metal"
            -o "${NUMI_MATTER_SHADER_DIR}/HumanRespiration.air"
        COMMAND xcrun -sdk macosx metallib
            "${NUMI_MATTER_SHADER_DIR}/HumanRespiration.air"
            -o "${NUMI_HUMAN_RESPIRATION_METALLIB}"
        DEPENDS src/human_respiration.metal include/numi/matter/human_respiration.h
            src/NumiHumanRestingCommonField.metalinc
            ../src/metal/PairedSourceGeometry.metalinc
            ../include/metalrobo/compensated_geometry_gpu.h
            include/numi/matter/shared.h ../src/metal/MujocoMuscleReference.metal
            ../include/metalrobo/mujoco_muscle_gpu.h
            ../include/metalrobo/numi_human_stand_gpu.h
            ../include/metalrobo/numi_human_resting_visual_gpu.h
            ../include/metalrobo/numi_human_support_geometry_gpu.h
            ../src/metal/NumiHumanRestingSupportGeometry.metalinc
            "${NUMI_BRAIN_ROOT}/Sources/NumiBrainABI/include/NumiBrainRespiratoryChemoreflexV1.h"
            "${NUMI_BRAIN_ROOT}/Sources/NumiBrainMetal/Shaders/RespiratoryChemoreflexV1.metal"
        VERBATIM
    )
    add_custom_target(numi_human_respiration_shaders DEPENDS "${NUMI_HUMAN_RESPIRATION_METALLIB}")
    # Explicit cross-suite target: ordinary Matter builds do not require a
    # sibling Brain checkout. Launchers select and bind NUMI_BRAIN_ROOT.
    add_executable(numi-human-resting EXCLUDE_FROM_ALL tools/human_resting.mm)
    add_dependencies(numi-human-resting numi_human_respiration_shaders)
    target_include_directories(numi-human-resting PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_include_directories(numi-human-resting PRIVATE "${NUMI_BRAIN_ROOT}/Sources/NumiBrainABI/include")
    target_compile_options(numi-human-resting PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_compile_definitions(numi-human-resting PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}"
        NUMI_HUMAN_RESPIRATION_METALLIB="${NUMI_HUMAN_RESPIRATION_METALLIB}")
    target_link_libraries(numi-human-resting PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    # Explicit bounded GPU regression fixture for the existing RestingRun owner.
    add_executable(numi-human-gas-transport-fixture EXCLUDE_FROM_ALL
        tools/human_resting_gas_transport_fixture.mm)
    add_dependencies(numi-human-gas-transport-fixture numi_human_respiration_shaders)
    target_include_directories(numi-human-gas-transport-fixture PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_include_directories(numi-human-gas-transport-fixture PRIVATE "${NUMI_BRAIN_ROOT}/Sources/NumiBrainABI/include")
    target_compile_options(numi-human-gas-transport-fixture PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_compile_definitions(numi-human-gas-transport-fixture PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}"
        NUMI_HUMAN_RESPIRATION_METALLIB="${NUMI_HUMAN_RESPIRATION_METALLIB}")
    target_link_libraries(numi-human-gas-transport-fixture PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
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

    install(FILES
        "${NUMI_MATTER_METALLIB}"
        "${NUMI_MATTER_PHYSICAL_DIGEST_METALLIB}"
        DESTINATION lib/numi-matter)
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
    set(NUMI_CARDBOARD_PAPER_AIR "${NUMI_MATTER_SHADER_DIR}/cardboard_paper_check.air")
    set(NUMI_CARDBOARD_PAPER_METALLIB "${NUMI_MATTER_SHADER_DIR}/cardboard_paper_check.metallib")
    add_custom_command(OUTPUT "${NUMI_CARDBOARD_PAPER_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_paper_check.metal" -o "${NUMI_CARDBOARD_PAPER_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_PAPER_AIR}" -o "${NUMI_CARDBOARD_PAPER_METALLIB}"
        DEPENDS tools/cardboard_paper_check.metal src/metal/common.metalinc
            include/numi/matter/shared.h include/numi/matter/fiber_exp_linear.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM)
    add_custom_target(numi_cardboard_paper_shaders DEPENDS "${NUMI_CARDBOARD_PAPER_METALLIB}")
    add_executable(numi-matter-cardboard-paper-gpu-check tools/cardboard_paper_gpu_check.mm)
    add_dependencies(numi-matter-cardboard-paper-gpu-check numi_cardboard_paper_shaders)
    target_link_libraries(numi-matter-cardboard-paper-gpu-check PRIVATE numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-paper-gpu-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-paper-gpu-check PRIVATE NUMI_CARDBOARD_PAPER_METALLIB="${NUMI_CARDBOARD_PAPER_METALLIB}")
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_paper COMMAND numi-matter-cardboard-paper-gpu-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_liner_hill_ideal.nmatter"
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_medium_hill_ideal.nmatter")
        set_tests_properties(matter.metal.cardboard_paper PROPERTIES LABELS "matter;cardboard;plasticity" TIMEOUT 180)
    endif()
    set(NUMI_CARDBOARD_DELAMINATION_AIR
        "${NUMI_MATTER_SHADER_DIR}/cardboard_bleached_delamination_check.air")
    set(NUMI_CARDBOARD_DELAMINATION_METALLIB
        "${NUMI_MATTER_SHADER_DIR}/cardboard_bleached_delamination_check.metallib")
    add_custom_command(OUTPUT "${NUMI_CARDBOARD_DELAMINATION_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_bleached_delamination_check.metal"
            -o "${NUMI_CARDBOARD_DELAMINATION_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_DELAMINATION_AIR}"
            -o "${NUMI_CARDBOARD_DELAMINATION_METALLIB}"
        DEPENDS tools/cardboard_bleached_delamination_check.metal
            src/metal/common.metalinc include/numi/matter/shared.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM)
    add_custom_target(numi_cardboard_delamination_shaders
        DEPENDS "${NUMI_CARDBOARD_DELAMINATION_METALLIB}")
    add_executable(numi-matter-cardboard-delamination-gpu-check
        tools/cardboard_bleached_delamination_gpu_check.mm)
    add_dependencies(numi-matter-cardboard-delamination-gpu-check
        numi_cardboard_delamination_shaders)
    target_link_libraries(numi-matter-cardboard-delamination-gpu-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-delamination-gpu-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-delamination-gpu-check PRIVATE
        NUMI_CARDBOARD_BLEACHED_DELAMINATION_METALLIB="${NUMI_CARDBOARD_DELAMINATION_METALLIB}")
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_delamination_coupon
            COMMAND numi-matter-cardboard-delamination-gpu-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/tryding2023_bleached_paperboard_modeI_coupon.nmatter")
        set_tests_properties(matter.metal.cardboard_delamination_coupon
            PROPERTIES LABELS "matter;cardboard;delamination;material" TIMEOUT 180)
    endif()
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
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_translation_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_geometry_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/numi/matter/human_support_precision_gpu.h"
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

# Allocation-free incremental SHA-256 used by exact physical-state evidence.
# The vector probe compiles and executes the production Metal implementation;
# it intentionally does not claim that any runtime state has been captured.
if(APPLE)
    set(NUMI_SHA256_CHECK_AIR
        "${CMAKE_CURRENT_BINARY_DIR}/shaders/sha256_gpu_check.air")
    set(NUMI_SHA256_CHECK_METALLIB
        "${CMAKE_CURRENT_BINARY_DIR}/shaders/sha256_gpu_check.metallib")
    add_custom_command(OUTPUT "${NUMI_SHA256_CHECK_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/sha256_gpu_check.metal"
            -o "${NUMI_SHA256_CHECK_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_SHA256_CHECK_AIR}"
            -o "${NUMI_SHA256_CHECK_METALLIB}"
        DEPENDS
            tools/sha256_gpu_check.metal
            src/metal/sha256.metalinc
            src/metal/sha256_kernels.metalinc
            include/numi/matter/sha256_gpu.h
            include/numi/matter/shared.h
        VERBATIM)
    add_custom_target(numi_sha256_check_shaders
        DEPENDS "${NUMI_SHA256_CHECK_METALLIB}")
    add_executable(numi-matter-sha256-gpu-check tools/sha256_gpu_check.mm)
    add_dependencies(numi-matter-sha256-gpu-check numi_sha256_check_shaders)
    target_include_directories(numi-matter-sha256-gpu-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(numi-matter-sha256-gpu-check PRIVATE
        "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-sha256-gpu-check PRIVATE
        NUMI_SHA256_CHECK_METALLIB="${NUMI_SHA256_CHECK_METALLIB}")
    target_compile_options(numi-matter-sha256-gpu-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.sha256
            COMMAND numi-matter-sha256-gpu-check)
        set_tests_properties(matter.metal.sha256 PROPERTIES
            LABELS "matter;metal;sha256;physical-state" TIMEOUT 60)
    endif()
endif()

# Canonical physical-state digest schema parity. This drives all production
# digest kernels over controlled bytes and compares every owner/root SHA-256
# against an independent CommonCrypto oracle. It is algorithm/schema evidence,
# not a claim that live physical state has been captured or qualified.
if(APPLE)
    add_custom_target(numi_physical_state_digest_check_shaders
        DEPENDS "${NUMI_MATTER_PHYSICAL_DIGEST_METALLIB}")
    add_executable(numi-matter-physical-state-digest-gpu-check
        tools/physical_state_digest_gpu_check.mm)
    add_dependencies(numi-matter-physical-state-digest-gpu-check
        numi_physical_state_digest_check_shaders)
    target_include_directories(numi-matter-physical-state-digest-gpu-check
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(numi-matter-physical-state-digest-gpu-check PRIVATE
        "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-physical-state-digest-gpu-check
        PRIVATE
            NUMI_PHYSICAL_DIGEST_CHECK_METALLIB="${NUMI_MATTER_PHYSICAL_DIGEST_METALLIB}")
    target_compile_options(numi-matter-physical-state-digest-gpu-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.physical_state_digest_schema
            COMMAND numi-matter-physical-state-digest-gpu-check)
        set_tests_properties(matter.metal.physical_state_digest_schema
            PROPERTIES
                LABELS "matter;metal;sha256;schema;synthetic"
                TIMEOUT 60)
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
        add_test(NAME matter.metal.cvsim_source_dense45 COMMAND numi-matter-cvsim-check
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/cvsim21.native.v3.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/cvsim21-continuous-reference-64ms.csv"
            --vascular-dense45)
        set_tests_properties(matter.metal.cvsim_source_dense45 PROPERTIES LABELS "matter;metal;human;vascular" TIMEOUT 60)
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
    add_executable(numi-matter-fem-regional-material-check tools/fem_regional_material_check.mm)
    target_link_libraries(numi-matter-fem-regional-material-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-fem-regional-material-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-fem-regional-material-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fem-regional-material-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.fem_regional_material COMMAND numi-matter-fem-regional-material-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_ventricular_guccione_passive.nmatter"
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter" --cpu-only)
        add_test(NAME matter.metal.fem_regional_material COMMAND numi-matter-fem-regional-material-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_ventricular_guccione_passive.nmatter"
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter")
        set_tests_properties(matter.compiler.fem_regional_material matter.metal.fem_regional_material
            PROPERTIES LABELS "matter;material;human" TIMEOUT 180)
    endif()

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

    add_executable(numi-matter-vascular-human-binding-check tools/vascular_human_binding_check.mm)
    target_link_libraries(numi-matter-vascular-human-binding-check PRIVATE "${FOUNDATION_FRAMEWORK}" numi_matter_compiler)
    target_include_directories(numi-matter-vascular-human-binding-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_options(numi-matter-vascular-human-binding-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.vascular_human_binding
            COMMAND numi-matter-vascular-human-binding-check
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/human_organ_blood_bridge_20260913.json")
        set_tests_properties(matter.compiler.vascular_human_binding
            PROPERTIES LABELS "matter;compiler;human;vascular;provenance" TIMEOUT 120)
    endif()

    add_executable(numi-matter-vessel-registration-check tools/vessel_registration_check.mm)
    target_link_libraries(numi-matter-vessel-registration-check PRIVATE "${FOUNDATION_FRAMEWORK}")
    target_compile_options(numi-matter-vessel-registration-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.vessel_registration
            COMMAND numi-matter-vessel-registration-check
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/human_organ_vessel_registration_20260913.json")
        set_tests_properties(matter.compiler.vessel_registration
            PROPERTIES LABELS "matter;compiler;human;vascular;provenance" TIMEOUT 120)
        add_test(NAME matter.compiler.vessel_body_links
            COMMAND numi-matter-vessel-registration-check
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixtures/human_organ_vessel_body_link_20260913.json")
        set_tests_properties(matter.compiler.vessel_body_links
            PROPERTIES LABELS "matter;compiler;human;vascular;provenance" TIMEOUT 120)
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

if(APPLE)
    add_executable(numi-matter-cardboard-preconditioner-gpu-check
        tools/cardboard_fgmres_preconditioner_gpu_check.mm)
    target_link_libraries(numi-matter-cardboard-preconditioner-gpu-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-cardboard-preconditioner-gpu-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-preconditioner-gpu-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_preconditioner
            COMMAND numi-matter-cardboard-preconditioner-gpu-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_liner_hill_ideal.nmatter"
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_medium_hill_ideal.nmatter")
        set_tests_properties(matter.metal.cardboard_preconditioner PROPERTIES
            LABELS "matter;cardboard;numerics" TIMEOUT 180)
    endif()
    add_executable(numi-matter-fem-reference-check tools/fem_reference_check.mm)
    target_link_libraries(numi-matter-fem-reference-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-fem-reference-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-fem-reference-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fem-reference-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.fem_reference COMMAND numi-matter-fem-reference-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_ventricular_guccione_passive.nmatter"
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter" --cpu-only)
        add_test(NAME matter.metal.fem_reference COMMAND numi-matter-fem-reference-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_ventricular_guccione_passive.nmatter"
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter")
        set_tests_properties(matter.compiler.fem_reference matter.metal.fem_reference
            PROPERTIES LABELS "matter;material;human" TIMEOUT 180)
    endif()

endif()

if(APPLE)
    add_executable(numi-matter-directional-valve-check tools/directional_valve_check.mm)
    target_link_libraries(numi-matter-directional-valve-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-directional-valve-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-directional-valve-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-directional-valve-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.directional_valve COMMAND numi-matter-directional-valve-check --cpu-only)
        add_test(NAME matter.metal.directional_valve COMMAND numi-matter-directional-valve-check)
        set_tests_properties(matter.compiler.directional_valve matter.metal.directional_valve
            PROPERTIES LABELS "matter;human;vascular" TIMEOUT 180)
    endif()
endif()

if(APPLE)
    add_executable(numi-matter-wet-lab tools/wet_lab_geometry.mm)
    target_include_directories(numi-matter-wet-lab PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_link_libraries(numi-matter-wet-lab PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-wet-lab PRIVATE -fobjc-arc -Wall -Wextra -Wno-deprecated-declarations)
    target_compile_definitions(numi-matter-wet-lab PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    install(TARGETS numi-matter-wet-lab RUNTIME DESTINATION bin)
endif()
