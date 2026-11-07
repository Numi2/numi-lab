cmake_minimum_required(VERSION 3.28)
project(NumiCCDCompleteSteps LANGUAGES CXX OBJCXX)
get_filename_component(NUMI_WORLD_ROOT "../../.." ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(LibXml2 REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/owner_shader_recipe.py"
    "${NUMI_WORLD_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/owner-shaders.cmake"
    COMMAND_ERROR_IS_FATAL ANY)
include("${CMAKE_CURRENT_BINARY_DIR}/owner-shaders.cmake")
# Static archives pull the actual production owners used by MetalWorld.
file(GLOB _core "${NUMI_WORLD_ROOT}/src/core/*.cpp")
add_library(numi_ccd_world_owner STATIC ${_core}
    "${NUMI_WORLD_ROOT}/src/metal/MetalWorld.mm"
    "${NUMI_WORLD_ROOT}/src/metal/MetalArticulatedOperator.mm")
target_include_directories(numi_ccd_world_owner PUBLIC "${NUMI_WORLD_ROOT}/include" "${NUMI_WORLD_ROOT}/matter/include")
target_compile_options(numi_ccd_world_owner PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>" -fno-fast-math -ffp-contract=off)
foreach(_framework Foundation Metal QuartzCore Accelerate CoreGraphics CoreImage ImageIO MetalKit ModelIO)
 find_library(_f_${_framework} ${_framework} REQUIRED)
 target_link_libraries(numi_ccd_world_owner PUBLIC "${_f_${_framework}}")
endforeach()
target_link_libraries(numi_ccd_world_owner PUBLIC LibXml2::LibXml2 SQLite::SQLite3)
add_executable(metalrobo_ccd_world_step_probe "${NUMI_WORLD_ROOT}/apps/ccd_world_step_probe.cpp")
target_link_libraries(metalrobo_ccd_world_step_probe PRIVATE numi_ccd_world_owner)
target_link_options(metalrobo_ccd_world_step_probe PRIVATE -Wl,-dead_strip)
add_dependencies(metalrobo_ccd_world_step_probe metalrobo_shaders)

include(CTest)
if(BUILD_TESTING)
 add_test(NAME metalrobo.ccd.contact_frame_diagnostic
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_contact_frame_diagnostic.py")
 add_test(NAME metalrobo.ccd.sliding_reference
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_sliding_contact_reference.py")
 add_test(NAME metalrobo.ccd.interval_reference
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_interval_monotonic_reference.py")
 add_test(NAME metalrobo.ccd.acceptance COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/regression.py" "$<TARGET_FILE:metalrobo_ccd_world_step_probe>"
  "${METALROBO_METALLIB}" "${CMAKE_CURRENT_BINARY_DIR}/acceptance-evidence")
 set_tests_properties(metalrobo.ccd.acceptance PROPERTIES TIMEOUT 180)
endif()
