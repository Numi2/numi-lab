cmake_minimum_required(VERSION 3.28)
project(NumiMatterPerTetMassCheck LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(numi_matter_compiler STATIC
    src/matter.cpp
    src/language_impl.cpp
    src/constitutive_impl.cpp
    src/compiler_impl.cpp
    src/vascular_impl.cpp
    src/fingerprint_impl.cpp
    src/validation_impl.cpp
    src/package_impl.cpp
    src/surgical_tissue.cpp)
target_include_directories(numi_matter_compiler PUBLIC include PRIVATE ../include)
target_compile_options(numi_matter_compiler PRIVATE -Wall -Wextra -Wpedantic)
add_executable(numi-matter-explicit-fem-tet-mass-check
    tools/fem_tetrahedron_mass_compiler_check.cpp)
target_link_libraries(numi-matter-explicit-fem-tet-mass-check PRIVATE numi_matter_compiler)
target_compile_options(numi-matter-explicit-fem-tet-mass-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
enable_testing()
add_test(NAME explicit_fem_tetrahedron_mass
    COMMAND numi-matter-explicit-fem-tet-mass-check)

add_executable(numi-legacy-package-bytes-current
    tools/legacy_package_bytes_fixture.cpp)
target_link_libraries(numi-legacy-package-bytes-current PRIVATE numi_matter_compiler)
add_library(numi_package_v19_baseline OBJECT
    ../baseline-v19/package_impl_v19.cpp)
target_include_directories(numi_package_v19_baseline PRIVATE include ../include)
target_compile_options(numi_package_v19_baseline PRIVATE -Wall -Wextra -Wpedantic)
add_executable(numi-legacy-package-bytes-v19
    tools/legacy_package_bytes_fixture.cpp
    $<TARGET_OBJECTS:numi_package_v19_baseline>)
target_link_libraries(numi-legacy-package-bytes-v19 PRIVATE numi_matter_compiler)

add_executable(numi-package-v19-reader-check
    tools/legacy_package_reader.cpp
    $<TARGET_OBJECTS:numi_package_v19_baseline>)
target_link_libraries(numi-package-v19-reader-check PRIVATE numi_matter_compiler)

add_executable(numi-human-unpaired-owner-compiler-integration-check
    ../../tests/NumiHumanUnpairedSkinFatVolumeCheck.cpp)
target_link_libraries(numi-human-unpaired-owner-compiler-integration-check
    PRIVATE numi_matter_compiler)
target_include_directories(numi-human-unpaired-owner-compiler-integration-check BEFORE PRIVATE
    include
    ../../apps
    ../include
    /Users/numi/numi-human-aortic-contact-opt-20261019/include)
target_compile_options(numi-human-unpaired-owner-compiler-integration-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
target_compile_definitions(numi-human-unpaired-owner-compiler-integration-check PRIVATE
    NUMI_TEST_MATERIAL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter")
add_test(NAME owner_emitted_mass_compiles_to_fem_inertia
    COMMAND numi-human-unpaired-owner-compiler-integration-check)
