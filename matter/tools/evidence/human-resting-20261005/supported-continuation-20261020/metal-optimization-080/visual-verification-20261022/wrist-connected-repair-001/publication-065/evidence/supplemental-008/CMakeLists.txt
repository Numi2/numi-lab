cmake_minimum_required(VERSION 3.28)
project(NumiMatterPerTetMassMultiobject LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()
add_subdirectory("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-040-shared-columns/attempt-007-owner-per-child-mass/compiler-integration-main79/matter" matter)
add_executable(numi-matter-fem-tet-mass-multiobject-check
    source/fem_tetrahedron_mass_multiobject_check.cpp)
target_link_libraries(numi-matter-fem-tet-mass-multiobject-check PRIVATE
    numi_matter_compiler)
target_compile_options(numi-matter-fem-tet-mass-multiobject-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
target_compile_definitions(numi-matter-fem-tet-mass-multiobject-check PRIVATE
    NUMI_TEST_MATERIAL_PATH="/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-040-shared-columns/attempt-007-owner-per-child-mass/compiler-integration-main79/matter/materials/damageable_silicone.nmatter")
add_test(NAME explicit_fem_tetrahedron_mass_multiobject
    COMMAND numi-matter-fem-tet-mass-multiobject-check)
set_tests_properties(explicit_fem_tetrahedron_mass_multiobject PROPERTIES
    TIMEOUT 30 LABELS "matter;compiler;fem;mass;cpu;synthetic")
