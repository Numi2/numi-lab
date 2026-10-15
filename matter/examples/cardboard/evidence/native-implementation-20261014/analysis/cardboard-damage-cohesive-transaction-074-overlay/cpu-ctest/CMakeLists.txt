cmake_minimum_required(VERSION 3.28)
project(CardboardDamageCohesiveTransactionReceipt LANGUAGES CXX OBJCXX)
include(CTest)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 23)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

set(NUMI_SOURCE_ROOT "" CACHE PATH "Frozen source root")
set(NUMI_BUILD_ROOT "" CACHE PATH "Frozen Matter build root")
if(NOT NUMI_SOURCE_ROOT OR NOT NUMI_BUILD_ROOT)
    message(FATAL_ERROR "Set NUMI_SOURCE_ROOT and NUMI_BUILD_ROOT")
endif()

add_library(numi_matter_compiler STATIC IMPORTED GLOBAL)
set_target_properties(numi_matter_compiler PROPERTIES
    IMPORTED_LOCATION "${NUMI_BUILD_ROOT}/matter/libnumi_matter_compiler.a")
add_library(numi_matter_runtime STATIC IMPORTED GLOBAL)
set_target_properties(numi_matter_runtime PROPERTIES
    IMPORTED_LOCATION "${NUMI_BUILD_ROOT}/matter/libnumi_matter_runtime.a")

set(_overlay_root "${CMAKE_CURRENT_LIST_DIR}/../overlay/matter/tools")
set(_materials "${NUMI_SOURCE_ROOT}/matter/materials")
set(_parameters "${_overlay_root}/cardboard_damage_cohesive_transaction_parameters.json")
set(_metallib "${NUMI_BUILD_ROOT}/matter/shaders/NumiMatter.metallib")
add_executable(numi-matter-cardboard-damage-cohesive-transaction-check
    "${_overlay_root}/cardboard_damage_cohesive_transaction_check.mm")
target_include_directories(numi-matter-cardboard-damage-cohesive-transaction-check PRIVATE
    "${NUMI_SOURCE_ROOT}/matter/include"
    "${NUMI_SOURCE_ROOT}/include"
    "${NUMI_SOURCE_ROOT}/matter/tools")
target_compile_definitions(numi-matter-cardboard-damage-cohesive-transaction-check PRIVATE
    NUMI_MATTER_METALLIB="${_metallib}")
target_compile_options(numi-matter-cardboard-damage-cohesive-transaction-check PRIVATE
    -fobjc-arc -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(numi-matter-cardboard-damage-cohesive-transaction-check PRIVATE
    numi_matter_runtime numi_matter_compiler "-framework Foundation" "-framework Metal")

if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_damage_cohesive_transaction
        COMMAND numi-matter-cardboard-damage-cohesive-transaction-check
            "${_materials}" "${_parameters}" --cpu-only)
    set_tests_properties(matter.compiler.cardboard_damage_cohesive_transaction PROPERTIES
        TIMEOUT 60 LABELS "matter;cardboard;damage;cohesive;cpu;synthetic")
endif()
