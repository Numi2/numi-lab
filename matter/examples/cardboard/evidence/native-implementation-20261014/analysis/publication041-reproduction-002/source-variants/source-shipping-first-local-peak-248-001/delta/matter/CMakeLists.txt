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
add_executable(numi-matter-cardboard-constituent-volume-damage-check
    tools/cardboard_constituent_volume_damage_check.cpp)
target_compile_options(numi-matter-cardboard-constituent-volume-damage-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_constituent_volume_damage
        COMMAND numi-matter-cardboard-constituent-volume-damage-check)
    set_tests_properties(matter.compiler.cardboard_constituent_volume_damage PROPERTIES
        TIMEOUT 30 LABELS "matter;cardboard;damage;cpu;synthetic")
endif()
add_executable(numi-matter-regional-graph-schwarz-layout-check
    tools/regional_graph_schwarz_layout_check.cpp)
target_compile_options(numi-matter-regional-graph-schwarz-layout-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.regional_graph_schwarz_layout
        COMMAND numi-matter-regional-graph-schwarz-layout-check)
    set_tests_properties(matter.compiler.regional_graph_schwarz_layout PROPERTIES
        TIMEOUT 30 LABELS "matter;cardboard;solver;cpu;synthetic")
endif()
add_executable(numi-matter-cardboard-refined-graph-layout-check
    tools/cardboard_refined_graph_layout_check.cpp)
target_include_directories(numi-matter-cardboard-refined-graph-layout-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(numi-matter-cardboard-refined-graph-layout-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_refined_graph_layout
        COMMAND numi-matter-cardboard-refined-graph-layout-check)
    set_tests_properties(matter.compiler.cardboard_refined_graph_layout PROPERTIES
        TIMEOUT 120 LABELS "matter;cardboard;solver;cpu;mesh;arena")
endif()
add_executable(numi-matter-regional-graph-completed-tail-check
    tools/regional_graph_completed_tail_check.cpp)
target_compile_options(numi-matter-regional-graph-completed-tail-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.regional_graph_completed_tail
        COMMAND numi-matter-regional-graph-completed-tail-check)
    set_tests_properties(matter.compiler.regional_graph_completed_tail PROPERTIES
        TIMEOUT 30 LABELS "matter;cardboard;solver;cpu;synthetic")
endif()
add_executable(numi-matter-cardboard-contact-node-incidence-sort-check
    tools/cardboard_contact_node_incidence_sort_check.cpp)
target_compile_options(numi-matter-cardboard-contact-node-incidence-sort-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_contact_node_incidence_sort
        COMMAND numi-matter-cardboard-contact-node-incidence-sort-check)
    set_tests_properties(matter.compiler.cardboard_contact_node_incidence_sort PROPERTIES
        TIMEOUT 30 LABELS "matter;cardboard;contact;cpu;synthetic")
endif()
add_executable(numi-matter-mortar-exact-tiling-check
    tools/mortar_exact_tiling_check.cpp)
target_include_directories(numi-matter-mortar-exact-tiling-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(numi-matter-mortar-exact-tiling-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.mortar_exact_tiling
        COMMAND numi-matter-mortar-exact-tiling-check)
    set_tests_properties(matter.compiler.mortar_exact_tiling PROPERTIES
        TIMEOUT 30 LABELS "matter;cardboard;contact;cpu;synthetic")
endif()
add_executable(numi-matter-regional-restart-augmentation-admission-check
    tools/regional_restart_augmentation_admission_check.cpp)
target_link_libraries(numi-matter-regional-restart-augmentation-admission-check PRIVATE
    numi_matter_compiler)
target_include_directories(numi-matter-regional-restart-augmentation-admission-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-regional-restart-augmentation-admission-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.regional_restart_augmentation_admission
        COMMAND numi-matter-regional-restart-augmentation-admission-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter")
    set_tests_properties(matter.compiler.regional_restart_augmentation_admission PROPERTIES
        TIMEOUT 30 LABELS "matter;cardboard;solver;cpu;synthetic")
endif()
add_executable(numi-matter-cardboard-cohesive-armijo-tiles-check
    tools/cardboard_cohesive_armijo_tiles_check.cpp)
target_compile_options(numi-matter-cardboard-cohesive-armijo-tiles-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
target_include_directories(numi-matter-cardboard-cohesive-armijo-tiles-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_cohesive_armijo_tiles
        COMMAND numi-matter-cardboard-cohesive-armijo-tiles-check)
    set_tests_properties(matter.compiler.cardboard_cohesive_armijo_tiles PROPERTIES
        TIMEOUT 30 LABELS "matter;cardboard;cohesive;cpu;synthetic")
endif()
add_executable(numi-matter-cardboard-solver-replay tools/cardboard_solver_replay.cpp)
target_compile_options(numi-matter-cardboard-solver-replay PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(numi-matter-cardboard-solver-pair-replay tools/cardboard_solver_pair_replay.cpp)
target_link_libraries(numi-matter-cardboard-solver-pair-replay PRIVATE numi_matter_compiler)
target_include_directories(numi-matter-cardboard-solver-pair-replay PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CURRENT_SOURCE_DIR}/tools")
target_compile_options(numi-matter-cardboard-solver-pair-replay PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_solver_reference
        COMMAND numi-matter-cardboard-solver-replay --check)
    set_tests_properties(matter.compiler.cardboard_solver_reference
        PROPERTIES LABELS "matter;cardboard;solver;reference" TIMEOUT 30)
endif()
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME matter.analysis.fgmres_true_residual_contract
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_true_residual_certification_contract.py")
        set_tests_properties(matter.analysis.fgmres_true_residual_contract
            PROPERTIES TIMEOUT 30 LABELS "matter;fgmres;true-residual;cpu;contract")
        add_executable(numi-matter-true-residual-arena-plan-check
            tools/true_residual_arena_plan_check.cpp)
        target_include_directories(numi-matter-true-residual-arena-plan-check PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_options(numi-matter-true-residual-arena-plan-check PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME matter.analysis.true_residual_arena_plan
            COMMAND numi-matter-true-residual-arena-plan-check)
        set_tests_properties(matter.analysis.true_residual_arena_plan
            PROPERTIES TIMEOUT 30 LABELS "matter;fgmres;true-residual;cpu;admission")
        add_executable(numi-matter-pure-fem-armijo-cgls-arena-plan-check
            tools/pure_fem_armijo_cgls_arena_plan_check.cpp)
        target_include_directories(numi-matter-pure-fem-armijo-cgls-arena-plan-check PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_options(numi-matter-pure-fem-armijo-cgls-arena-plan-check PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME matter.analysis.pure_fem_armijo_cgls_arena_plan
            COMMAND numi-matter-pure-fem-armijo-cgls-arena-plan-check)
        set_tests_properties(matter.analysis.pure_fem_armijo_cgls_arena_plan
            PROPERTIES TIMEOUT 30 LABELS "matter;fgmres;armijo;cgls;cpu;admission")
        set(NUMI_FEM_CGLS_ARENA_8192_CHECK_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
        set(NUMI_FEM_CGLS_ARENA_8192_CHECK_AIR "${NUMI_FEM_CGLS_ARENA_8192_CHECK_DIR}/FEMPureCGLSArena8192Check.air")
        set(NUMI_FEM_CGLS_ARENA_8192_CHECK_METALLIB "${NUMI_FEM_CGLS_ARENA_8192_CHECK_DIR}/FEMPureCGLSArena8192Check.metallib")
        add_custom_command(
            OUTPUT "${NUMI_FEM_CGLS_ARENA_8192_CHECK_METALLIB}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_FEM_CGLS_ARENA_8192_CHECK_DIR}"
            COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
                -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src/metal"
                -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/pure_fem_cgls_arena_8192_check.metal"
                -o "${NUMI_FEM_CGLS_ARENA_8192_CHECK_AIR}"
            COMMAND xcrun -sdk macosx metallib "${NUMI_FEM_CGLS_ARENA_8192_CHECK_AIR}"
                -o "${NUMI_FEM_CGLS_ARENA_8192_CHECK_METALLIB}"
            DEPENDS tools/pure_fem_cgls_arena_8192_check.metal
                src/metal/common.metalinc src/metal/fem.metalinc
                src/metal/base_cgls.metalinc src/metal/base_cgls_runtime.metalinc
                src/metal/fgmres.metalinc include/numi/matter/shared.h
            VERBATIM)
        add_custom_target(numi_fem_cgls_arena_8192_check_shaders
            DEPENDS "${NUMI_FEM_CGLS_ARENA_8192_CHECK_METALLIB}")
        add_executable(numi-matter-fem-cgls-arena-8192-check
            tools/pure_fem_cgls_arena_8192_check.mm)
        add_dependencies(numi-matter-fem-cgls-arena-8192-check
            numi_fem_cgls_arena_8192_check_shaders)
        target_include_directories(numi-matter-fem-cgls-arena-8192-check PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
        find_library(NUMI_FEM_CGLS_CHECK_FOUNDATION Foundation REQUIRED)
        find_library(NUMI_FEM_CGLS_CHECK_METAL Metal REQUIRED)
        target_link_libraries(numi-matter-fem-cgls-arena-8192-check PRIVATE
            "${NUMI_FEM_CGLS_CHECK_FOUNDATION}" "${NUMI_FEM_CGLS_CHECK_METAL}")
        target_compile_definitions(numi-matter-fem-cgls-arena-8192-check PRIVATE
            NUMI_FEM_CGLS_ARENA_METALLIB="${NUMI_FEM_CGLS_ARENA_8192_CHECK_METALLIB}")
        target_compile_options(numi-matter-fem-cgls-arena-8192-check PRIVATE
            -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
        if(BUILD_TESTING)
            add_test(NAME matter.metal.fem_cgls_arena_8192_native_check
                COMMAND numi-matter-fem-cgls-arena-8192-check)
            set_tests_properties(matter.metal.fem_cgls_arena_8192_native_check
                PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE
                    LABELS "matter;fgmres;cgls;metal;synthetic;arena")
        endif()
        add_test(NAME matter.analysis.cardboard_source_data
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_source_data_test.py")
        add_test(NAME matter.analysis.cardboard_pin_source_observables
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_pin_adhesion_source_observables_test.py")
        set_tests_properties(matter.analysis.cardboard_pin_source_observables
            PROPERTIES TIMEOUT 60 LABELS "matter;cardboard;sources;identifiability")
        add_test(NAME matter.analysis.cardboard_bct_capacity_producer
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_bct_capacity_history_producer.py")
        add_test(NAME matter.analysis.cardboard_bct_source_bound_harness
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_bct_source_bound_harness.py")
        set_tests_properties(matter.analysis.cardboard_bct_capacity_producer
            matter.analysis.cardboard_bct_source_bound_harness PROPERTIES
            TIMEOUT 60 LABELS "matter;cardboard;analysis;synthetic")
        add_test(NAME matter.analysis.cardboard_calibration_ensemble
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_calibration_ensemble_test.py")
        add_test(NAME matter.analysis.cardboard_resolution_study
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_resolution_study.py")
        set_tests_properties(matter.analysis.cardboard_calibration_ensemble
            matter.analysis.cardboard_resolution_study PROPERTIES
            TIMEOUT 60 LABELS "matter;cardboard;analysis;synthetic")
        set_tests_properties(matter.analysis.cardboard_source_data
            PROPERTIES TIMEOUT 60 LABELS "matter;cardboard;sources")
        add_test(NAME matter.analysis.cardboard_convergence
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_convergence.py")
        add_test(NAME matter.analysis.cardboard_tooling
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_tooling_analysis.py")
        add_test(NAME matter.analysis.cardboard_newton_trace
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_newton_trace.py")
        add_test(NAME matter.analysis.cardboard_accepted_animation
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_accepted_animation.py")
        add_test(NAME matter.analysis.cardboard_capsule_side_derivative
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_rigid_contact_derivative_replay.PortableCapsuleSideDerivativeTests)
        set_tests_properties(matter.analysis.cardboard_capsule_side_derivative PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tools"
            TIMEOUT 30 LABELS "matter;cardboard;contact;derivative;cpu")
        add_test(NAME matter.analysis.cardboard_deformable_derivative
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_deformable_contact_derivative_replay)
        add_test(NAME matter.analysis.cardboard_retained_capture_audit
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_retained_damage_capture_audit)
        add_test(NAME matter.analysis.cardboard_stage_resolution
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_stage_resolution_study)
        add_test(NAME matter.analysis.cardboard_regional_conditioning
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_regional_conditioning_audit)
        add_test(NAME matter.analysis.cardboard_cached_operator_audit
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_cached_operator_audit)
        add_test(NAME matter.analysis.cardboard_bilinear_capture_layers
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_bilinear_capture_layers)
        add_test(NAME matter.analysis.cardboard_reference_fb_jacobian
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_reference_fb_jacobian)
        add_test(NAME matter.analysis.cardboard_compensated_fp64_reference
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_compensated_fp64_reference)
        add_test(NAME matter.analysis.cardboard_offline_conditioning_study
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_offline_conditioning_study)
        add_test(NAME matter.analysis.cardboard_diagnostic_history
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_diagnostic_history_compare)
        add_test(NAME matter.analysis.cardboard_solver_pair
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_solver_pair_analyzer)
        add_test(NAME matter.analysis.cardboard_solver_assembly
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_solver_assembly_audit)
        add_test(NAME matter.analysis.cardboard_full_contact_history
            COMMAND "${Python3_EXECUTABLE}" -m unittest
                test_cardboard_full_contact_history_compare)
        set_tests_properties(matter.analysis.cardboard_deformable_derivative
            matter.analysis.cardboard_retained_capture_audit
            matter.analysis.cardboard_stage_resolution
            matter.analysis.cardboard_regional_conditioning
            matter.analysis.cardboard_cached_operator_audit
            matter.analysis.cardboard_bilinear_capture_layers
            matter.analysis.cardboard_reference_fb_jacobian
            matter.analysis.cardboard_compensated_fp64_reference
            matter.analysis.cardboard_offline_conditioning_study
            matter.analysis.cardboard_diagnostic_history
            matter.analysis.cardboard_solver_pair
            matter.analysis.cardboard_solver_assembly
            matter.analysis.cardboard_full_contact_history PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tools"
            TIMEOUT 60 LABELS "matter;cardboard;analysis;cpu")
        set_tests_properties(matter.analysis.cardboard_accepted_animation PROPERTIES
            TIMEOUT 30 LABELS "matter;cardboard;analysis;geometry")
        set_tests_properties(matter.analysis.cardboard_convergence matter.analysis.cardboard_tooling matter.analysis.cardboard_newton_trace
            PROPERTIES LABELS "matter;cardboard;analysis" TIMEOUT 60)
    endif()
endif()
add_executable(numi-matter-cardboard-tooling-check tools/cardboard_tooling_check.cpp)
add_executable(numi-matter-cardboard-contact-candidate-domain-check
    tools/cardboard_contact_candidate_domain_check.cpp)
target_include_directories(numi-matter-cardboard-contact-candidate-domain-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(numi-matter-cardboard-contact-candidate-domain-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_contact_candidate_domain
        COMMAND numi-matter-cardboard-contact-candidate-domain-check)
    set_tests_properties(matter.compiler.cardboard_contact_candidate_domain PROPERTIES
        LABELS "matter;cardboard;contact;candidate-domain;cpu" TIMEOUT 30)
endif()
add_executable(numi-matter-cardboard-reaction-observables-check tools/cardboard_reaction_observables_check.cpp)
target_include_directories(numi-matter-cardboard-reaction-observables-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-reaction-observables-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_reaction_observables
        COMMAND numi-matter-cardboard-reaction-observables-check)
    set_tests_properties(matter.compiler.cardboard_reaction_observables PROPERTIES
        LABELS "matter;cardboard;reaction;cpu" TIMEOUT 30)
endif()
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
foreach(cardboardGeometry IN ITEMS fracture_regularization starch_interface_mesh)
    string(REPLACE "_" "-" cardboardGeometryTarget "${cardboardGeometry}")
    add_executable("numi-matter-cardboard-${cardboardGeometryTarget}-check"
        "tools/cardboard_${cardboardGeometry}_check.cpp")
    target_compile_options("numi-matter-cardboard-${cardboardGeometryTarget}-check"
        PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME "matter.compiler.cardboard_${cardboardGeometry}"
            COMMAND "numi-matter-cardboard-${cardboardGeometryTarget}-check")
        set_tests_properties("matter.compiler.cardboard_${cardboardGeometry}"
            PROPERTIES TIMEOUT 60 LABELS "matter;cardboard;geometry")
    endif()
endforeach()
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
add_executable(numi-matter-cardboard-box-assembly-controller-check
    tools/cardboard_box_assembly_controller_check.cpp)
add_executable(numi-matter-cardboard-shipping-protocol-check
    tools/cardboard_shipping_protocol_check.cpp)
target_include_directories(numi-matter-cardboard-shipping-protocol-check
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-shipping-protocol-check
    PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_shipping_protocol
        COMMAND numi-matter-cardboard-shipping-protocol-check)
    set_tests_properties(matter.compiler.cardboard_shipping_protocol
        PROPERTIES TIMEOUT 60 LABELS "compiler;cardboard;shipping;geometry")
endif()
target_include_directories(numi-matter-cardboard-box-assembly-controller-check
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-box-assembly-controller-check
    PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_box_assembly_controller
        COMMAND numi-matter-cardboard-box-assembly-controller-check)
    set_tests_properties(matter.compiler.cardboard_box_assembly_controller
        PROPERTIES LABELS "matter;cardboard;assembly;geometry" TIMEOUT 60)
endif()
add_executable(numi-matter-cardboard-paper-material-check tools/cardboard_paper_material_check.cpp)
add_executable(numi-matter-cardboard-damage-builder-check tools/cardboard_damage_builder_check.cpp)
add_executable(numi-matter-cardboard-retained-matrix-damage-check
    tools/cardboard_retained_matrix_damage_check.cpp)
target_link_libraries(numi-matter-cardboard-retained-matrix-damage-check
    PRIVATE numi_matter_compiler)
target_include_directories(numi-matter-cardboard-retained-matrix-damage-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-retained-matrix-damage-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(numi-matter-cardboard-damage-builder-check PRIVATE numi_matter_compiler)
target_include_directories(numi-matter-cardboard-damage-builder-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-damage-builder-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_damage_builder
        COMMAND numi-matter-cardboard-damage-builder-check)
    set_tests_properties(matter.compiler.cardboard_damage_builder PROPERTIES
        TIMEOUT 60 LABELS "matter;cardboard;material;synthetic")
    add_test(NAME matter.compiler.cardboard_retained_matrix_damage
        COMMAND numi-matter-cardboard-retained-matrix-damage-check)
    set_tests_properties(matter.compiler.cardboard_retained_matrix_damage PROPERTIES
        TIMEOUT 120 LABELS "matter;cardboard;material;synthetic")
endif()
target_include_directories(numi-matter-cardboard-paper-material-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_executable(numi-matter-cardboard-compensated-deformation-check
    tools/cardboard_compensated_deformation_check.cpp)
target_include_directories(numi-matter-cardboard-compensated-deformation-check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_compile_options(numi-matter-cardboard-compensated-deformation-check PRIVATE
    -ffp-contract=off -fno-fast-math)
if(BUILD_TESTING)
    add_test(NAME cardboard-compensated-deformation
        COMMAND numi-matter-cardboard-compensated-deformation-check)
endif()

add_executable(numi-matter-fem-regional-galerkin-compiler-check
    tools/fem_regional_galerkin_compiler_check.cpp)
add_executable(numi-matter-fem-static-surface-map-check
    tools/fem_static_surface_map_check.cpp)
target_link_libraries(numi-matter-fem-static-surface-map-check PRIVATE numi_matter_compiler)
target_compile_options(numi-matter-fem-static-surface-map-check PRIVATE -Wall -Wextra -Wpedantic)
add_executable(numi-matter-cardboard-pin-adhesion-fixture-check
    tools/cardboard_pin_adhesion_fixture_check.cpp)
target_include_directories(numi-matter-cardboard-pin-adhesion-fixture-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-pin-adhesion-fixture-check PRIVATE -Wall -Wextra -Wpedantic)
add_executable(numi-matter-cardboard-retained-damage-fp64-oracle-check
    tools/cardboard_retained_damage_fp64_oracle_check.cpp)
target_include_directories(numi-matter-cardboard-retained-damage-fp64-oracle-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(numi-matter-cardboard-retained-damage-fp64-oracle-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(numi-matter-cardboard-generic-compensated-projection-check
    tools/cardboard_generic_compensated_projection_check.cpp)
target_link_libraries(numi-matter-cardboard-generic-compensated-projection-check PRIVATE numi_matter_compiler)
target_include_directories(numi-matter-cardboard-generic-compensated-projection-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-generic-compensated-projection-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_executable(numi-matter-cardboard-bilinear15-projection-policy-check
    tools/cardboard_bilinear15_projection_policy_check.cpp)
target_link_libraries(numi-matter-cardboard-bilinear15-projection-policy-check
    PRIVATE numi_matter_compiler)
target_include_directories(numi-matter-cardboard-bilinear15-projection-policy-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(numi-matter-cardboard-bilinear15-projection-policy-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.fem_static_surface_map COMMAND numi-matter-fem-static-surface-map-check)
    set_tests_properties(matter.compiler.fem_static_surface_map PROPERTIES LABELS "matter;compiler;contact;cpu")
    add_test(NAME matter.compiler.cardboard_pin_adhesion_fixture COMMAND numi-matter-cardboard-pin-adhesion-fixture-check)
    set_tests_properties(matter.compiler.cardboard_pin_adhesion_fixture PROPERTIES LABELS "matter;cardboard;source;cpu")
    add_test(NAME matter.compiler.cardboard_retained_damage_fp64_oracle COMMAND numi-matter-cardboard-retained-damage-fp64-oracle-check)
    set_tests_properties(matter.compiler.cardboard_retained_damage_fp64_oracle PROPERTIES LABELS "matter;cardboard;reference;cpu")
    add_test(NAME matter.compiler.cardboard_generic_compensated_projection COMMAND numi-matter-cardboard-generic-compensated-projection-check)
    set_tests_properties(matter.compiler.cardboard_generic_compensated_projection PROPERTIES LABELS "matter;cardboard;projection;cpu" TIMEOUT 120)
    add_test(NAME matter.compiler.cardboard_bilinear15_projection_policy
        COMMAND numi-matter-cardboard-bilinear15-projection-policy-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials")
    set_tests_properties(matter.compiler.cardboard_bilinear15_projection_policy PROPERTIES
        LABELS "matter;cardboard;projection;cpu;synthetic" TIMEOUT 60)
endif()
target_link_libraries(numi-matter-fem-regional-galerkin-compiler-check PRIVATE
    numi_matter_compiler)
target_compile_options(numi-matter-fem-regional-galerkin-compiler-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.fem_regional_galerkin
        COMMAND numi-matter-fem-regional-galerkin-compiler-check)
    set_tests_properties(matter.compiler.fem_regional_galerkin PROPERTIES
        TIMEOUT 60 LABELS "matter;fem;preconditioner;synthetic")
endif()

add_executable(numi-matter-fixed-convex-contact-step-check
    tools/fixed_convex_contact_step_check.cpp)
target_include_directories(numi-matter-fixed-convex-contact-step-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(numi-matter-fixed-convex-contact-step-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.fixed_convex_contact_step
        COMMAND numi-matter-fixed-convex-contact-step-check)
    set_tests_properties(matter.compiler.fixed_convex_contact_step PROPERTIES
        TIMEOUT 60 LABELS "matter;contact;ccd;synthetic")
endif()

add_executable(numi-matter-cardboard-hill-scalar-return-map-check
    tools/cardboard_hill_scalar_return_map_check.cpp)
target_include_directories(numi-matter-cardboard-hill-scalar-return-map-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(numi-matter-cardboard-hill-scalar-return-map-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off)
foreach(cardboard_helper cohesive-capacity-area shipping-driver shipping-memory shipping-closure-mesh shipping-snapshot coupon-resume inside-clearance shipping-clearance-memory regional-consumer-capture)
    string(REPLACE "-" "_" cardboard_helper_file "${cardboard_helper}")
    add_executable(numi-matter-cardboard-${cardboard_helper}-check
        tools/cardboard_${cardboard_helper_file}_check.cpp)
    target_include_directories(numi-matter-cardboard-${cardboard_helper}-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_options(numi-matter-cardboard-${cardboard_helper}-check PRIVATE
        -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
    if(BUILD_TESTING)
        set(cardboard_helper_arguments)
        if(cardboard_helper STREQUAL "regional-consumer-capture")
            set(cardboard_helper_arguments "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime.mm")
        endif()
        add_test(NAME matter.compiler.cardboard_${cardboard_helper_file}
            COMMAND numi-matter-cardboard-${cardboard_helper}-check ${cardboard_helper_arguments})
        set_tests_properties(matter.compiler.cardboard_${cardboard_helper_file}
            PROPERTIES TIMEOUT 60 LABELS "matter;cardboard;shipping;cpu")
    endif()
endforeach()
add_executable(numi-matter-cardboard-mixed-cohesive-check tools/cardboard_mixed_cohesive_check.cpp)
add_executable(numi-matter-cardboard-mixed-cohesive-frame-check tools/cardboard_mixed_cohesive_frame_check.cpp)
target_compile_options(numi-matter-cardboard-mixed-cohesive-frame-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_mixed_cohesive_frame COMMAND numi-matter-cardboard-mixed-cohesive-frame-check)
    set_tests_properties(matter.compiler.cardboard_mixed_cohesive_frame PROPERTIES LABELS "matter;cardboard;cohesive;cpu" TIMEOUT 60)
endif()
target_compile_options(numi-matter-cardboard-mixed-cohesive-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_mixed_cohesive COMMAND numi-matter-cardboard-mixed-cohesive-check)
    set_tests_properties(matter.compiler.cardboard_mixed_cohesive PROPERTIES LABELS "matter;cardboard;cohesive;cpu" TIMEOUT 60)
endif()
target_link_libraries(numi-matter-cardboard-paper-material-check PRIVATE numi_matter_compiler)
target_compile_options(numi-matter-cardboard-paper-material-check PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_paper COMMAND numi-matter-cardboard-paper-material-check)
    set_tests_properties(matter.compiler.cardboard_paper PROPERTIES LABELS "matter;cardboard;material" TIMEOUT 120)
endif()
add_executable(numi-matter-cardboard-expression-cache-check
    tools/cardboard_expression_cache_check.cpp)
target_link_libraries(numi-matter-cardboard-expression-cache-check PRIVATE numi_matter_compiler)
target_include_directories(numi-matter-cardboard-expression-cache-check PRIVATE tools)
target_compile_options(numi-matter-cardboard-expression-cache-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_expression_cache
        COMMAND numi-matter-cardboard-expression-cache-check)
    set_tests_properties(matter.compiler.cardboard_expression_cache
        PROPERTIES LABELS "matter;cardboard;abi;expression-cache;cpu" TIMEOUT 60)
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
    set(NUMI_MATTER_DEPFILE "${NUMI_MATTER_SHADER_DIR}/NumiMatter.d")
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
            -MMD -MF "${NUMI_MATTER_DEPFILE}" -MT "${NUMI_MATTER_METALLIB}"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/src/matter_kernels.metal"
            -o "${NUMI_MATTER_AIR}"
        COMMAND xcrun -sdk macosx metallib
            "${NUMI_MATTER_AIR}"
            -o "${NUMI_MATTER_METALLIB}"
        DEPENDS
            src/matter_kernels.metal
            src/metal/common.metalinc
            include/numi/matter/paper_compensated_bytecode.h
            include/numi/matter/fem_compensated_deformation.h
            include/numi/matter/hill_scalar_initializer.h
            src/metal/mpm.metalinc
            src/metal/fem.metalinc
            src/metal/fgmres.metalinc
            src/metal/base_cgls.metalinc
            src/metal/base_cgls_runtime.metalinc
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
            include/numi/matter/mortar_shared_edge_guard.h
            include/numi/matter/human_equality_gpu.h
            include/numi/matter/human_limits_gpu.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_translation_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_geometry_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/numi/matter/human_support_precision_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        DEPFILE "${NUMI_MATTER_DEPFILE}"
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
    set(NUMI_CARDBOARD_WARM_CHECK_AIR "${NUMI_MATTER_SHADER_DIR}/CardboardWarmStartCheck.air")
    set(NUMI_CARDBOARD_WARM_CHECK_METALLIB "${NUMI_MATTER_SHADER_DIR}/CardboardWarmStartCheck.metallib")
    add_custom_command(
        OUTPUT "${NUMI_CARDBOARD_WARM_CHECK_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/metal"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_base_cgls_warm_start_check.metal"
            -o "${NUMI_CARDBOARD_WARM_CHECK_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_WARM_CHECK_AIR}"
            -o "${NUMI_CARDBOARD_WARM_CHECK_METALLIB}"
        DEPENDS tools/cardboard_base_cgls_warm_start_check.metal
            src/metal/common.metalinc src/metal/fem.metalinc
            src/metal/base_cgls.metalinc src/metal/base_cgls_runtime.metalinc
            src/metal/fgmres.metalinc include/numi/matter/shared.h
        VERBATIM)
    add_custom_target(numi_cardboard_warm_start_check_shaders
        DEPENDS "${NUMI_CARDBOARD_WARM_CHECK_METALLIB}")
    add_executable(numi-matter-cardboard-warm-start-check
        tools/cardboard_base_cgls_warm_start_check.mm)
    add_dependencies(numi-matter-cardboard-warm-start-check numi_cardboard_warm_start_check_shaders)
    target_include_directories(numi-matter-cardboard-warm-start-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(numi-matter-cardboard-warm-start-check PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-cardboard-warm-start-check PRIVATE
        NUMI_WARM_START_METALLIB="${NUMI_CARDBOARD_WARM_CHECK_METALLIB}")
    target_compile_options(numi-matter-cardboard-warm-start-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_warm_start_kernel_contract
            COMMAND numi-matter-cardboard-warm-start-check)
        set_tests_properties(matter.metal.cardboard_warm_start_kernel_contract PROPERTIES
            LABELS "matter;cardboard;solver;metal;synthetic" TIMEOUT 60 RUN_SERIAL TRUE)
    endif()
    add_executable(numi-matter-fem-static-surface-map-native-check tools/fem_static_surface_map_native_check.mm)
    add_dependencies(numi-matter-fem-static-surface-map-native-check numi_matter_shaders)
    target_link_libraries(numi-matter-fem-static-surface-map-native-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-fem-static-surface-map-native-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fem-static-surface-map-native-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.fem_static_surface_map_native_fixture COMMAND numi-matter-fem-static-surface-map-native-check --cpu-only)
        set_tests_properties(matter.compiler.fem_static_surface_map_native_fixture PROPERTIES LABELS "matter;compiler;contact;cpu")
        add_test(NAME matter.metal.fem_static_surface_map COMMAND numi-matter-fem-static-surface-map-native-check)
        set_tests_properties(matter.metal.fem_static_surface_map PROPERTIES LABELS "matter;contact;surface;synthetic" TIMEOUT 120 RUN_SERIAL TRUE)
    endif()
    add_executable(numi-matter-fgmres-true-residual-native-check tools/fgmres_true_residual_native_check.mm)
    add_dependencies(numi-matter-fgmres-true-residual-native-check numi_matter_shaders)
    target_link_libraries(numi-matter-fgmres-true-residual-native-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-fgmres-true-residual-native-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fgmres-true-residual-native-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.fgmres_true_residual_kernel_contract
            COMMAND numi-matter-fgmres-true-residual-native-check)
        set_tests_properties(matter.metal.fgmres_true_residual_kernel_contract
            PROPERTIES LABELS "matter;metal;fgmres;true-residual;synthetic"
            TIMEOUT 120 RUN_SERIAL TRUE)
    endif()
    add_executable(numi-matter-fem-kinematic-release-check tools/fem_kinematic_release_check.mm)
    target_link_libraries(numi-matter-fem-kinematic-release-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-fem-kinematic-release-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-fem-kinematic-release-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fem-kinematic-release-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.fem_kinematic_release COMMAND numi-matter-fem-kinematic-release-check)
        set_tests_properties(matter.metal.fem_kinematic_release PROPERTIES LABELS "matter;fem;release" TIMEOUT 120)
    endif()
    add_executable(numi-matter-fem-mutable-snapshot-check tools/fem_mutable_snapshot_check.mm)
    target_link_libraries(numi-matter-fem-mutable-snapshot-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-fem-mutable-snapshot-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-fem-mutable-snapshot-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fem-mutable-snapshot-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.fem_mutable_snapshot COMMAND numi-matter-fem-mutable-snapshot-check)
        set_tests_properties(matter.metal.fem_mutable_snapshot PROPERTIES LABELS "matter;fem;topology;snapshot" TIMEOUT 120)
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
    add_executable(numi-matter-cardboard-capsule-armijo-check tools/cardboard_capsule_armijo_check.mm)
    target_link_libraries(numi-matter-cardboard-capsule-armijo-check PRIVATE numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-capsule-armijo-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-capsule-armijo-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-capsule-armijo-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_capsule_armijo COMMAND numi-matter-cardboard-capsule-armijo-check)
        set_tests_properties(matter.metal.cardboard_capsule_armijo PROPERTIES LABELS "matter;cardboard;contact;armijo" TIMEOUT 180)
    endif()
    add_executable(numi-matter-cardboard-contact-ccd-native-check
        tools/cardboard_contact_ccd_native_check.mm)
    add_dependencies(numi-matter-cardboard-contact-ccd-native-check numi_matter_shaders)
    target_link_libraries(numi-matter-cardboard-contact-ccd-native-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-contact-ccd-native-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-contact-ccd-native-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-contact-ccd-native-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_contact_ccd_native
            COMMAND numi-matter-cardboard-contact-ccd-native-check)
        set_tests_properties(matter.metal.cardboard_contact_ccd_native PROPERTIES
            LABELS "matter;cardboard;contact;ccd;native-kernel" TIMEOUT 60)
    endif()
    # Standalone exact-input VTCCD budget replay. This target is deliberately
    # not registered as a CTest because it consumes retained native capture files.
    add_executable(numi-matter-cardboard-vtccd-budget-replay-check
        tools/cardboard_vtccd_iteration_budget_check.mm)
    add_dependencies(numi-matter-cardboard-vtccd-budget-replay-check numi_matter_shaders)
    target_link_libraries(numi-matter-cardboard-vtccd-budget-replay-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-vtccd-budget-replay-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-vtccd-budget-replay-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-vtccd-budget-replay-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    add_executable(numi-matter-cardboard-full-contact-scratch-budget-check
        tools/cardboard_full_contact_armijo_scratch_budget_check.cpp)
    target_include_directories(numi-matter-cardboard-full-contact-scratch-budget-check
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_options(numi-matter-cardboard-full-contact-scratch-budget-check
        PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.cardboard_full_contact_scratch_budget
            COMMAND numi-matter-cardboard-full-contact-scratch-budget-check)
        set_tests_properties(matter.compiler.cardboard_full_contact_scratch_budget
            PROPERTIES TIMEOUT 10 LABELS "matter;cardboard;contact;memory;cpu")
    endif()
    add_executable(numi-matter-cardboard-full-contact-armijo-check tools/cardboard_full_contact_armijo_check.mm)
    target_link_libraries(numi-matter-cardboard-full-contact-armijo-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-full-contact-armijo-check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-full-contact-armijo-check PRIVATE NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-full-contact-armijo-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.cardboard_full_contact_armijo COMMAND numi-matter-cardboard-full-contact-armijo-check
            "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter" --cpu-only)
        set_tests_properties(matter.compiler.cardboard_full_contact_armijo PROPERTIES TIMEOUT 60 LABELS "matter;cardboard;contact;armijo;synthetic")
        add_test(NAME matter.compiler.cardboard_compensated_full_contact
            COMMAND numi-matter-cardboard-full-contact-armijo-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter"
                --cpu-only --compensated-material-arithmetic)
        set_tests_properties(matter.compiler.cardboard_compensated_full_contact PROPERTIES
            TIMEOUT 60 LABELS "matter;cardboard;contact;arithmetic;cpu;synthetic")
    endif()
    # Direct native angular free-BOX contact/operator regression.  This is a
    # kernel-level check with no Matter FEM time steps; it is intentionally
    # separate from the full Armijo integration fixture.
    add_executable(numi-matter-cardboard-free-box-angular-native-check
        tools/cardboard_free_box_angular_native_check.mm)
    add_dependencies(numi-matter-cardboard-free-box-angular-native-check
        numi_matter_shaders)
    target_link_libraries(numi-matter-cardboard-free-box-angular-native-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-free-box-angular-native-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-free-box-angular-native-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-free-box-angular-native-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_free_box_angular_native
            COMMAND numi-matter-cardboard-free-box-angular-native-check)
        set_tests_properties(matter.metal.cardboard_free_box_angular_native PROPERTIES
            LABELS "matter;cardboard;contact;angular;native-kernel" TIMEOUT 180)
    endif()
    add_executable(numi-matter-cardboard-coupled-free-box-operator-check
        tools/cardboard_coupled_free_box_operator_check.mm)
    add_dependencies(numi-matter-cardboard-coupled-free-box-operator-check numi_matter_shaders)
    target_link_libraries(numi-matter-cardboard-coupled-free-box-operator-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-coupled-free-box-operator-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-coupled-free-box-operator-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-coupled-free-box-operator-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_coupled_free_box_operator
            COMMAND numi-matter-cardboard-coupled-free-box-operator-check)
        set_tests_properties(matter.metal.cardboard_coupled_free_box_operator PROPERTIES
            TIMEOUT 180 LABELS "matter;cardboard;contact;native-kernel;synthetic")
    endif()
    add_executable(numi-matter-cardboard-free-force-residual-check
        tools/cardboard_free_force_residual_check.cpp)
    target_include_directories(numi-matter-cardboard-free-force-residual-check
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" tools)
    target_compile_options(numi-matter-cardboard-free-force-residual-check
        PRIVATE -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.cardboard_free_force_residual
            COMMAND numi-matter-cardboard-free-force-residual-check)
        set_tests_properties(matter.compiler.cardboard_free_force_residual
            PROPERTIES LABELS "matter;cardboard;force-residual;cpu")
    endif()
    add_executable(numi-matter-cardboard-free-box-proxy-admission-check
        tools/cardboard_free_box_true_residual_proxy_admission_check.cpp)
    target_include_directories(numi-matter-cardboard-free-box-proxy-admission-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_options(numi-matter-cardboard-free-box-proxy-admission-check PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.cardboard_free_box_proxy_admission
            COMMAND numi-matter-cardboard-free-box-proxy-admission-check)
        set_tests_properties(matter.compiler.cardboard_free_box_proxy_admission PROPERTIES
            LABELS "matter;cardboard;contact;admission;cpu")
    endif()
    add_executable(numi-matter-cardboard-contact-node-incidence-sort-native-check
        tools/cardboard_contact_node_incidence_sort_native_check.mm)
    add_dependencies(numi-matter-cardboard-contact-node-incidence-sort-native-check
        numi_matter_shaders)
    target_link_libraries(numi-matter-cardboard-contact-node-incidence-sort-native-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-cardboard-contact-node-incidence-sort-native-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-contact-node-incidence-sort-native-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_contact_node_incidence_sort_native
            COMMAND numi-matter-cardboard-contact-node-incidence-sort-native-check)
        set_tests_properties(matter.metal.cardboard_contact_node_incidence_sort_native PROPERTIES
            TIMEOUT 180 LABELS "matter;cardboard;contact;native-kernel;synthetic")
    endif()
    add_executable(numi-matter-cardboard-graph-lu-search-native-check
        tools/cardboard_graph_lu_search_native_check.mm)
    add_dependencies(numi-matter-cardboard-graph-lu-search-native-check numi_matter_shaders)
    target_link_libraries(numi-matter-cardboard-graph-lu-search-native-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-graph-lu-search-native-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    add_executable(numi-matter-cardboard-probe tools/cardboard_probe.mm)
    target_link_libraries(numi-matter-cardboard-probe PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_options(numi-matter-cardboard-probe PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_compile_definitions(numi-matter-cardboard-probe PRIVATE
        NUMI_CARDBOARD_MATERIAL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/materials"
        NUMI_CARDBOARD_METALLIB="${NUMI_MATTER_METALLIB}")
    if(BUILD_TESTING)
        if(Python3_Interpreter_FOUND)
            add_test(NAME matter.cli.cardboard_argument_admission
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_cardboard_probe_argument_admission.py"
                    --probe $<TARGET_FILE:numi-matter-cardboard-probe>)
            set_tests_properties(matter.cli.cardboard_argument_admission PROPERTIES
                TIMEOUT 120 LABELS "matter;cardboard;cpu;cli")
        endif()
        add_test(NAME matter.compiler.cardboard_pin_fixture_preflight COMMAND numi-matter-cardboard-probe
            --pin-adhesion --preset literature2009 --compile-only
            --output "${CMAKE_CURRENT_BINARY_DIR}/cardboard-pin-fixture-cpu-preflight")
        set_tests_properties(matter.compiler.cardboard_pin_fixture_preflight PROPERTIES
            TIMEOUT 120 LABELS "matter;cardboard;pin;geometry;cpu")
        add_test(NAME matter.cli.cardboard_graph_option
            COMMAND numi-matter-cardboard-probe --help)
        set_tests_properties(matter.cli.cardboard_graph_option PROPERTIES
            PASS_REGULAR_EXPRESSION "regional-galerkin-schwarz"
            TIMEOUT 15 LABELS "matter;cardboard;cpu;cli")
    endif()
    add_executable(numi-matter-cardboard-box-native-driver tools/cardboard_probe.mm)
    target_link_libraries(numi-matter-cardboard-box-native-driver PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-box-native-driver PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_options(numi-matter-cardboard-box-native-driver PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
    target_compile_definitions(numi-matter-cardboard-box-native-driver PRIVATE
        NUMI_CARDBOARD_MATERIAL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/materials"
        NUMI_CARDBOARD_METALLIB="${NUMI_MATTER_METALLIB}"
        NUMI_CARDBOARD_BOX_NATIVE_DRIVER=1)
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
            ../src/metal/NumiHumanRestingBed.metalinc
            ../include/metalrobo/numi_human_resting_bed_gpu.h
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
    set(NUMI_CARDBOARD_PAPER_DEPFILE "${NUMI_MATTER_SHADER_DIR}/cardboard_paper_check.d")
    file(GLOB NUMI_CARDBOARD_PRODUCTION_METAL_DEPENDENCIES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/*.metalinc")
    add_custom_command(OUTPUT "${NUMI_CARDBOARD_PAPER_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -MMD -MF "${NUMI_CARDBOARD_PAPER_DEPFILE}" -MT "${NUMI_CARDBOARD_PAPER_METALLIB}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_paper_check.metal" -o "${NUMI_CARDBOARD_PAPER_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_PAPER_AIR}" -o "${NUMI_CARDBOARD_PAPER_METALLIB}"
        DEPENDS tools/cardboard_paper_check.metal src/matter_kernels.metal
            ${NUMI_CARDBOARD_PRODUCTION_METAL_DEPENDENCIES}
            include/numi/matter/paper_compensated_bytecode.h
            include/numi/matter/fem_compensated_deformation.h
            include/numi/matter/hill_scalar_initializer.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_geometry_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_translation_gpu.h"
            include/numi/matter/shared.h include/numi/matter/fiber_exp_linear.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        DEPFILE "${NUMI_CARDBOARD_PAPER_DEPFILE}"
        VERBATIM)
    add_custom_target(numi_cardboard_paper_shaders DEPENDS "${NUMI_CARDBOARD_PAPER_METALLIB}")
    # Analytical checks of the production contact helpers at physical board scales.
    set(NUMI_CARDBOARD_GEOMETRY_AIR "${NUMI_MATTER_SHADER_DIR}/cardboard_contact_geometry_check.air")
    set(NUMI_CARDBOARD_GEOMETRY_METALLIB "${NUMI_MATTER_SHADER_DIR}/cardboard_contact_geometry_check.metallib")
    set(NUMI_CARDBOARD_GEOMETRY_DEPFILE "${NUMI_MATTER_SHADER_DIR}/cardboard_contact_geometry_check.d")
    add_custom_command(OUTPUT "${NUMI_CARDBOARD_GEOMETRY_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -MMD -MF "${NUMI_CARDBOARD_GEOMETRY_DEPFILE}" -MT "${NUMI_CARDBOARD_GEOMETRY_METALLIB}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_contact_geometry_check.metal" -o "${NUMI_CARDBOARD_GEOMETRY_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_GEOMETRY_AIR}" -o "${NUMI_CARDBOARD_GEOMETRY_METALLIB}"
        DEPENDS tools/cardboard_contact_geometry_check.metal src/matter_kernels.metal
            ${NUMI_CARDBOARD_PRODUCTION_METAL_DEPENDENCIES}
            include/numi/matter/paper_compensated_bytecode.h
            include/numi/matter/fem_compensated_deformation.h
            include/numi/matter/hill_scalar_initializer.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_geometry_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_translation_gpu.h"
            include/numi/matter/shared.h include/numi/matter/fiber_exp_linear.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        DEPFILE "${NUMI_CARDBOARD_GEOMETRY_DEPFILE}"
        VERBATIM)
    add_custom_target(numi_cardboard_contact_geometry_shaders DEPENDS "${NUMI_CARDBOARD_GEOMETRY_METALLIB}")
    add_executable(numi-matter-cardboard-contact-geometry-check tools/cardboard_contact_geometry_check.mm)
    add_dependencies(numi-matter-cardboard-contact-geometry-check numi_cardboard_contact_geometry_shaders)
    target_link_libraries(numi-matter-cardboard-contact-geometry-check PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-contact-geometry-check PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-contact-geometry-check PRIVATE NUMI_CARDBOARD_GEOMETRY_METALLIB="${NUMI_CARDBOARD_GEOMETRY_METALLIB}")
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_contact_geometry COMMAND numi-matter-cardboard-contact-geometry-check)
        set_tests_properties(matter.metal.cardboard_contact_geometry PROPERTIES
            LABELS "matter;cardboard;contact;native-geometry" TIMEOUT 60 RUN_SERIAL TRUE)
    endif()
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
    # Separate native check for the source-fit 15-state defaults. The existing
    # paper GPU checker intentionally remains a 13-state ideal-Hill regression.
    add_executable(numi-matter-cardboard-bilinear-paper-gpu-check
        tools/cardboard_bilinear_paper_gpu_check.mm)
    add_dependencies(numi-matter-cardboard-bilinear-paper-gpu-check
        numi_cardboard_paper_shaders)
    target_link_libraries(numi-matter-cardboard-bilinear-paper-gpu-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-bilinear-paper-gpu-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-bilinear-paper-gpu-check PRIVATE
        NUMI_CARDBOARD_PAPER_METALLIB="${NUMI_CARDBOARD_PAPER_METALLIB}")
    # Exact N0/N63 capture replay for the opt-in 15-state projection policy.
    # This executable is intentionally not a CTest: it requires sealed external
    # capture/reference inputs and a physical Apple GPU when explicitly run.
    add_executable(numi-matter-cardboard-bilinear15-captured-point-check
        tools/cardboard_bilinear15_captured_point_check.mm)
    add_dependencies(numi-matter-cardboard-bilinear15-captured-point-check
        numi_cardboard_paper_shaders)
    target_link_libraries(numi-matter-cardboard-bilinear15-captured-point-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-bilinear15-captured-point-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-bilinear15-captured-point-check PRIVATE
        NUMI_CARDBOARD_PAPER_METALLIB="${NUMI_CARDBOARD_PAPER_METALLIB}"
        NUMI_CARDBOARD_BASE_SNAPSHOT_SHA256="98fe026ee2007982e48f89547a435a2957db3978fb3730e008fdbb99d08f5704")
    # Exact replay of the source-370 first rejected material candidate. This
    # diagnostic uses the production FEM deformation/material-frame helpers,
    # then evaluates the same captured material point with explicit 16/32
    # local projection budgets. It performs no physical transactions and is
    # deliberately not a CTest or a full-coupon qualification.
    # This source-bound replay uses the same metallib as the existing paper
    # material checks; the production kernel includes above provide the exact
    # deformation helpers that the older common-only library did not contain.
    add_executable(numi-matter-cardboard-source370-material-point-check
        tools/cardboard_source370_material_point_check.mm)
    add_dependencies(numi-matter-cardboard-source370-material-point-check
        numi_cardboard_paper_shaders)
    target_link_libraries(numi-matter-cardboard-source370-material-point-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-source370-material-point-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-source370-material-point-check PRIVATE
        NUMI_CARDBOARD_PAPER_METALLIB="${NUMI_CARDBOARD_PAPER_METALLIB}"
        NUMI_CARDBOARD_SOURCE107_MANIFEST_SHA256="09ae737a09623ff15fc4b1d13f8a96cb14e5d87e0c245cb7b1deb70bcf145d97")
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_bilinear_paper
            COMMAND numi-matter-cardboard-bilinear-paper-gpu-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_liner_hill_bilinear_fit.nmatter"
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_medium_hill_bilinear_fit.nmatter"
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/cardboard/source-data/haj_ali_2009_bilinear_tension_fit_nodes.csv")
        set_tests_properties(matter.metal.cardboard_bilinear_paper PROPERTIES
            LABELS "matter;cardboard;plasticity;native-material-point"
            TIMEOUT 300)
        add_test(NAME matter.compiler.cardboard_bilinear_liner_rest_fp32
            COMMAND numi-matter-cardboard-bilinear-paper-gpu-check --cpu-rest-diagnose
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_liner_hill_bilinear_fit.nmatter")
        add_test(NAME matter.compiler.cardboard_bilinear_medium_rest_fp32
            COMMAND numi-matter-cardboard-bilinear-paper-gpu-check --cpu-rest-diagnose
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_medium_hill_bilinear_fit.nmatter")
        set_tests_properties(matter.compiler.cardboard_bilinear_liner_rest_fp32
            matter.compiler.cardboard_bilinear_medium_rest_fp32 PROPERTIES
            LABELS "matter;cardboard;plasticity;fp32-domain"
            TIMEOUT 30)
    endif()
    # Bounded direct stress/tangent/projector diagnostics for a source-bound
    # FEM Newton capture. This shader is separate from the ordinary paper
    # checker and only runs when the external checker is invoked explicitly.
    set(NUMI_CARDBOARD_BILINEAR15_CAPTURE_AIR
        "${NUMI_MATTER_SHADER_DIR}/cardboard_bilinear15_capture_diagnostic.air")
    set(NUMI_CARDBOARD_BILINEAR15_CAPTURE_METALLIB
        "${NUMI_MATTER_SHADER_DIR}/cardboard_bilinear15_capture_diagnostic.metallib")
    add_custom_command(OUTPUT "${NUMI_CARDBOARD_BILINEAR15_CAPTURE_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_bilinear15_capture_diagnostic.metal"
            -o "${NUMI_CARDBOARD_BILINEAR15_CAPTURE_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_BILINEAR15_CAPTURE_AIR}"
            -o "${NUMI_CARDBOARD_BILINEAR15_CAPTURE_METALLIB}"
        DEPENDS tools/cardboard_bilinear15_capture_diagnostic.metal src/metal/common.metalinc
            include/numi/matter/paper_compensated_bytecode.h
            include/numi/matter/fem_compensated_deformation.h
            include/numi/matter/hill_scalar_initializer.h
            include/numi/matter/shared.h include/numi/matter/fiber_exp_linear.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_geometry_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_translation_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM)
    add_custom_target(numi_cardboard_bilinear15_capture_shaders
        DEPENDS "${NUMI_CARDBOARD_BILINEAR15_CAPTURE_METALLIB}")
    add_executable(numi-matter-cardboard-bilinear15-capture-diagnostic-check
        tools/cardboard_bilinear15_capture_diagnostic_check.mm)
    add_dependencies(numi-matter-cardboard-bilinear15-capture-diagnostic-check
        numi_cardboard_bilinear15_capture_shaders)
    target_link_libraries(numi-matter-cardboard-bilinear15-capture-diagnostic-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-bilinear15-capture-diagnostic-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-bilinear15-capture-diagnostic-check PRIVATE
        NUMI_CARDBOARD_PAPER_METALLIB="${NUMI_CARDBOARD_BILINEAR15_CAPTURE_METALLIB}"
        NUMI_CARDBOARD_BASE_SNAPSHOT_SHA256="769a5e34c1d919293cb263fd27fbd041a482d22016c2818eec41d44e47935752")
    # The C1 compression-asymmetry transfer candidate stays separate from the
    # ordinary 15-state source-fit defaults and the 13-state ideal regression.
    add_executable(numi-matter-cardboard-asymmetric-c1-paper-gpu-check
        tools/cardboard_asymmetric_c1_paper_gpu_check.mm)
    add_dependencies(numi-matter-cardboard-asymmetric-c1-paper-gpu-check
        numi_cardboard_paper_shaders)
    target_link_libraries(numi-matter-cardboard-asymmetric-c1-paper-gpu-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-asymmetric-c1-paper-gpu-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-asymmetric-c1-paper-gpu-check PRIVATE
        NUMI_CARDBOARD_PAPER_METALLIB="${NUMI_CARDBOARD_PAPER_METALLIB}")
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_asymmetric_c1_paper
            COMMAND numi-matter-cardboard-asymmetric-c1-paper-gpu-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_liner_hill_asymmetric_c1_transfer.nmatter"
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/hajali2009_medium_hill_asymmetric_c1_transfer.nmatter"
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/cardboard/source-data/haj_ali_2009_bilinear_tension_fit_nodes.csv"
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/cardboard/source-data/haj_ali_2009_asymmetric_c1_transfer_v1.csv")
        set_tests_properties(matter.metal.cardboard_asymmetric_c1_paper PROPERTIES
            LABELS "matter;cardboard;plasticity;native-material-point;c1-transfer"
            TIMEOUT 300)
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
    set(NUMI_CARDBOARD_EXPRESSION_CACHE_AIR
        "${NUMI_MATTER_SHADER_DIR}/cardboard_expression_cache_check.air")
    set(NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB
        "${NUMI_MATTER_SHADER_DIR}/cardboard_expression_cache_check.metallib")
    add_custom_command(OUTPUT "${NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_expression_cache_check.metal"
            -o "${NUMI_CARDBOARD_EXPRESSION_CACHE_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_EXPRESSION_CACHE_AIR}"
            -o "${NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB}"
        DEPENDS tools/cardboard_expression_cache_check.metal src/metal/common.metalinc
            include/numi/matter/paper_compensated_bytecode.h
            include/numi/matter/fem_compensated_deformation.h
            include/numi/matter/hill_scalar_initializer.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_geometry_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_translation_gpu.h"
            include/numi/matter/shared.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM)
    add_custom_target(numi_cardboard_expression_cache_shaders
        DEPENDS "${NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB}")
    add_executable(numi-matter-cardboard-expression-cache-gpu-check
        tools/cardboard_expression_cache_gpu_check.mm)
    add_dependencies(numi-matter-cardboard-expression-cache-gpu-check
        numi_cardboard_expression_cache_shaders)
    target_link_libraries(numi-matter-cardboard-expression-cache-gpu-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-expression-cache-gpu-check PRIVATE tools)
    target_compile_options(numi-matter-cardboard-expression-cache-gpu-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-expression-cache-gpu-check PRIVATE
        NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB="${NUMI_CARDBOARD_EXPRESSION_CACHE_METALLIB}")
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_expression_cache
            COMMAND numi-matter-cardboard-expression-cache-gpu-check)
        set_tests_properties(matter.metal.cardboard_expression_cache
            PROPERTIES LABELS "matter;metal;cardboard;abi;expression-cache" TIMEOUT 60)
    endif()
    add_executable(numi-matter-cardboard-bulk-damage-gpu-check
        tools/cardboard_bulk_damage_gpu_check.mm)
    add_dependencies(numi-matter-cardboard-bulk-damage-gpu-check numi_cardboard_delamination_shaders)
    target_link_libraries(numi-matter-cardboard-bulk-damage-gpu-check PRIVATE
        numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-bulk-damage-gpu-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-bulk-damage-gpu-check PRIVATE
        NUMI_CARDBOARD_BULK_METALLIB="${NUMI_CARDBOARD_DELAMINATION_METALLIB}")
    # Separately named native instrument for the retained Haj-Ali matrix plus
    # an explicitly synthetic damage overlay. Building this target compiles
    # Metal and Objective-C++ only; it is never registered as an auto-run test.
    set(NUMI_CARDBOARD_RETAINED_DAMAGE_AIR
        "${NUMI_MATTER_SHADER_DIR}/cardboard_retained_matrix_damage_check.air")
    set(NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB
        "${NUMI_MATTER_SHADER_DIR}/cardboard_retained_matrix_damage_check.metallib")
    add_custom_command(OUTPUT "${NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_MATTER_SHADER_DIR}"
        COMMAND xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/../include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_retained_matrix_damage_check.metal"
            -o "${NUMI_CARDBOARD_RETAINED_DAMAGE_AIR}"
        COMMAND xcrun -sdk macosx metallib "${NUMI_CARDBOARD_RETAINED_DAMAGE_AIR}"
            -o "${NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB}"
        DEPENDS tools/cardboard_retained_matrix_damage_check.metal
            src/metal/common.metalinc include/numi/matter/shared.h
            include/numi/matter/paper_compensated_bytecode.h
            include/numi/matter/fem_compensated_deformation.h
            include/numi/matter/hill_scalar_initializer.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_geometry_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/compensated_translation_gpu.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/engine_types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/metalrobo/rod_gpu_shared.h"
        VERBATIM)
    add_custom_target(numi_cardboard_retained_damage_shaders
        DEPENDS "${NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB}")
    add_executable(numi-matter-cardboard-retained-matrix-damage-metal-check
        tools/cardboard_retained_matrix_damage_gpu_check.mm)
    add_dependencies(numi-matter-cardboard-retained-matrix-damage-metal-check
        numi_cardboard_retained_damage_shaders)
    target_link_libraries(numi-matter-cardboard-retained-matrix-damage-metal-check
        PRIVATE numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_options(numi-matter-cardboard-retained-matrix-damage-metal-check
        PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    target_compile_definitions(numi-matter-cardboard-retained-matrix-damage-metal-check
        PRIVATE NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB="${NUMI_CARDBOARD_RETAINED_DAMAGE_METALLIB}")
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_delamination_coupon
            COMMAND numi-matter-cardboard-delamination-gpu-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/tryding2023_bleached_paperboard_modeI_coupon.nmatter")
        set_tests_properties(matter.metal.cardboard_delamination_coupon
            PROPERTIES LABELS "matter;cardboard;delamination;material" TIMEOUT 180)
    endif()

    add_executable(numi-matter-cardboard-damage-cohesive-transaction-check
        tools/cardboard_damage_cohesive_transaction_check.mm)
    target_link_libraries(numi-matter-cardboard-damage-cohesive-transaction-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-damage-cohesive-transaction-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-damage-cohesive-transaction-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-damage-cohesive-transaction-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.cardboard_damage_cohesive_transaction
            COMMAND numi-matter-cardboard-damage-cohesive-transaction-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_damage_cohesive_transaction_parameters.json"
                --cpu-only)
        add_test(NAME matter.metal.cardboard_damage_cohesive_transaction
            COMMAND numi-matter-cardboard-damage-cohesive-transaction-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/cardboard_damage_cohesive_transaction_parameters.json")
        set_tests_properties(matter.compiler.cardboard_damage_cohesive_transaction
            PROPERTIES TIMEOUT 60 LABELS "matter;cardboard;damage;cohesive;cpu;synthetic")
        set_tests_properties(matter.metal.cardboard_damage_cohesive_transaction
            PROPERTIES TIMEOUT 300 LABELS "matter;cardboard;damage;cohesive;metal;synthetic")
    endif()
    add_executable(numi-matter-cardboard-cohesive-face-check
        tools/cardboard_cohesive_face_check.mm)
    add_executable(numi-matter-cardboard-mixed-cohesive-face-check
        tools/cardboard_mixed_cohesive_face_check.mm)
    target_link_libraries(numi-matter-cardboard-mixed-cohesive-face-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-mixed-cohesive-face-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-mixed-cohesive-face-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-mixed-cohesive-face-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.cardboard_mixed_cohesive_face
            COMMAND numi-matter-cardboard-mixed-cohesive-face-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter" --cpu-only)
        add_test(NAME matter.metal.cardboard_mixed_cohesive_face
            COMMAND numi-matter-cardboard-mixed-cohesive-face-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter")
        set_tests_properties(matter.compiler.cardboard_mixed_cohesive_face
            matter.metal.cardboard_mixed_cohesive_face PROPERTIES
            TIMEOUT 180 LABELS "matter;cardboard;cohesive;synthetic")
    endif()
    target_link_libraries(numi-matter-cardboard-cohesive-face-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-cohesive-face-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-cohesive-face-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-cohesive-face-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.cardboard_cohesive_face
            COMMAND numi-matter-cardboard-cohesive-face-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter"
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/tryding2023_bleached_paperboard_modeI_coupon.nmatter"
                --cpu-only)
        add_test(NAME matter.metal.cardboard_cohesive_face
            COMMAND numi-matter-cardboard-cohesive-face-check
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/nagasawa2013_medium_orthotropic.nmatter"
                "${CMAKE_CURRENT_SOURCE_DIR}/materials/tryding2023_bleached_paperboard_modeI_coupon.nmatter")
        set_tests_properties(matter.compiler.cardboard_cohesive_face
            matter.metal.cardboard_cohesive_face
            PROPERTIES LABELS "matter;cardboard;cohesive;material" TIMEOUT 900)
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
            include/numi/matter/paper_compensated_bytecode.h
            include/numi/matter/fem_compensated_deformation.h
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
    add_executable(numi-matter-fem-regional-galerkin-native-check
        tools/fem_regional_galerkin_native_check.mm)
    target_link_libraries(numi-matter-fem-regional-galerkin-native-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-fem-regional-galerkin-native-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-fem-regional-galerkin-native-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-fem-regional-galerkin-native-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)

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

    add_executable(numi-matter-robertsson-fem-coupling-check
        tools/robertsson_fem_coupling_check.mm)
    target_link_libraries(numi-matter-robertsson-fem-coupling-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-robertsson-fem-coupling-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include" tools)
    target_compile_definitions(numi-matter-robertsson-fem-coupling-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-robertsson-fem-coupling-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.compiler.cardboard_robertsson_fem_coupling
            COMMAND numi-matter-robertsson-fem-coupling-check --cpu-only)
        set_tests_properties(matter.compiler.cardboard_robertsson_fem_coupling
            PROPERTIES LABELS "matter;compiler;cardboard;robertsson;fem"
                TIMEOUT 120)
        add_test(NAME matter.metal.cardboard_robertsson_fem_coupling
            COMMAND numi-matter-robertsson-fem-coupling-check
                "${CMAKE_CURRENT_BINARY_DIR}/robertsson_fem_path.csv")
        set_tests_properties(matter.metal.cardboard_robertsson_fem_coupling
            PROPERTIES LABELS "matter;metal;cardboard;robertsson;fem;synthetic"
                TIMEOUT 1800)
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


if(APPLE)
    add_executable(numi-matter-cardboard-regional-rap-tail-preserve-native-check
        tools/cardboard_regional_rap_tail_preserve_native_check.mm)
    target_link_libraries(numi-matter-cardboard-regional-rap-tail-preserve-native-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_include_directories(numi-matter-cardboard-regional-rap-tail-preserve-native-check PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    target_compile_definitions(numi-matter-cardboard-regional-rap-tail-preserve-native-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-regional-rap-tail-preserve-native-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_regional_rap_tail_preserve
            COMMAND numi-matter-cardboard-regional-rap-tail-preserve-native-check)
        set_tests_properties(matter.metal.cardboard_regional_rap_tail_preserve
            PROPERTIES RUN_SERIAL TRUE LABELS "matter;metal;cardboard;synthetic" TIMEOUT 120)
    endif()
endif()

if(APPLE)
    add_executable(numi-matter-cardboard-regional-recycle-masked-basis-native-check
        tools/cardboard_regional_recycle_masked_basis_native_check.mm)
    target_link_libraries(numi-matter-cardboard-regional-recycle-masked-basis-native-check PRIVATE
        numi_matter_runtime "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
    target_compile_definitions(numi-matter-cardboard-regional-recycle-masked-basis-native-check PRIVATE
        NUMI_MATTER_METALLIB="${NUMI_MATTER_METALLIB}")
    target_compile_options(numi-matter-cardboard-regional-recycle-masked-basis-native-check PRIVATE
        -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
    if(BUILD_TESTING)
        add_test(NAME matter.metal.cardboard_regional_recycle_masked_basis
            COMMAND numi-matter-cardboard-regional-recycle-masked-basis-native-check)
        set_tests_properties(matter.metal.cardboard_regional_recycle_masked_basis
            PROPERTIES LABELS "matter;metal;cardboard;synthetic" TIMEOUT 700)
    endif()
endif()

# Insert adjacent to the source-370 point target in matter/CMakeLists.txt.
add_executable(numi-matter-cardboard-source392-tet858-point-check
    tools/cardboard_source392_tet_point_check.mm)
add_dependencies(numi-matter-cardboard-source392-tet858-point-check
    numi_cardboard_paper_shaders)
target_link_libraries(numi-matter-cardboard-source392-tet858-point-check PRIVATE
    numi_matter_compiler "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
target_compile_options(numi-matter-cardboard-source392-tet858-point-check PRIVATE
    -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
target_compile_definitions(numi-matter-cardboard-source392-tet858-point-check PRIVATE
    NUMI_CARDBOARD_PAPER_METALLIB="${NUMI_CARDBOARD_PAPER_METALLIB}"
    NUMI_CARDBOARD_SOURCE118_BUILD_SHA256="da94e6051e767ea4bfba58705ccaf321e5dd97e5e9b40b32715f056676980c9f")
