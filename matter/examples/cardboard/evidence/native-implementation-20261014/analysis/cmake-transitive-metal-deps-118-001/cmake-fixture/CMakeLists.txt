cmake_minimum_required(VERSION 3.28)
project(metal_depfile_probe NONE)
set(AIR ${CMAKE_CURRENT_BINARY_DIR}/probe.air)
set(LIB ${CMAKE_CURRENT_BINARY_DIR}/probe.metallib)
set(DEP ${CMAKE_CURRENT_BINARY_DIR}/probe.d)
add_custom_command(OUTPUT ${LIB}
 COMMAND xcrun -sdk macosx metal -std=metal4.0 -MMD -MF ${DEP} -MT ${LIB}
   -c ${CMAKE_CURRENT_SOURCE_DIR}/main.metal -o ${AIR}
 COMMAND xcrun -sdk macosx metallib ${AIR} -o ${LIB}
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main.metal
 DEPFILE ${DEP} VERBATIM)
add_custom_target(probe ALL DEPENDS ${LIB})
