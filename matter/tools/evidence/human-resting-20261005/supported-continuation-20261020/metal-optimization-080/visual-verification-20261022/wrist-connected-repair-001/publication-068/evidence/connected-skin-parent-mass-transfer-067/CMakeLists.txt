cmake_minimum_required(VERSION 3.28)
project(ConnectedSkinParentMassTransfer067 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()
add_subdirectory("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-040-shared-columns/attempt-007-owner-per-child-mass/compiler-integration-main79/matter" matter)
add_executable(parent-reference-mass-transfer-check tests/ParentReferenceMassTransferCheck.cpp)
target_include_directories(parent-reference-mass-transfer-check BEFORE PRIVATE
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-skin-parent-mass-transfer-067/apps"
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-040-shared-columns/attempt-007-owner-per-child-mass/apps"
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-040-shared-columns/attempt-007-owner-per-child-mass/compiler-integration-main79/matter/include"
    "/Users/n/numi-human-aortic-contact-opt-20261019/include")
target_compile_options(parent-reference-mass-transfer-check PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
add_test(NAME parent_reference_mass_transfer COMMAND parent-reference-mass-transfer-check)
add_executable(unpaired-owner-parent-mass-integration tests/NumiHumanUnpairedSkinFatVolumeCheck.cpp)
target_include_directories(unpaired-owner-parent-mass-integration BEFORE PRIVATE
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-skin-parent-mass-transfer-067/apps"
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-040-shared-columns/attempt-007-owner-per-child-mass/apps"
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-040-shared-columns/attempt-007-owner-per-child-mass/compiler-integration-main79/matter/include"
    "/Users/n/numi-human-aortic-contact-opt-20261019/include")
target_link_libraries(unpaired-owner-parent-mass-integration PRIVATE numi_matter_compiler)
target_compile_options(unpaired-owner-parent-mass-integration PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
target_compile_definitions(unpaired-owner-parent-mass-integration PRIVATE
    NUMI_TEST_MATERIAL_PATH="/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-040-shared-columns/attempt-007-owner-per-child-mass/compiler-integration-main79/matter/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter")
add_test(NAME owner_parent_mass_compiles_to_fem_inertia COMMAND unpaired-owner-parent-mass-integration)
set_tests_properties(parent_reference_mass_transfer owner_parent_mass_compiles_to_fem_inertia PROPERTIES TIMEOUT 60 LABELS "matter;human;mass;cpu;synthetic")
