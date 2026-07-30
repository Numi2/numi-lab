cmake_minimum_required(VERSION 3.27)

project(metalrobo_mlx_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DEFINED ENV{METALROBO_PYTHON_EXECUTABLE})
    set(
        Python_EXECUTABLE
        "$ENV{METALROBO_PYTHON_EXECUTABLE}"
        CACHE FILEPATH
        "Python used to invoke the MLX extension build"
        FORCE
    )
endif()
find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module
    REQUIRED
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT
)
find_package(MLX CONFIG REQUIRED)

get_filename_component(
    METALROBO_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    ABSOLUTE
)
set(
    METALROBO_BUILD_DIR
    "${METALROBO_ROOT}/build"
    CACHE PATH
    "MetalRobo CMake build directory"
)
find_library(
    METALROBO_LIBRARY
    NAMES metalrobo
    PATHS "${METALROBO_BUILD_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)
set(
    METALROBO_METALLIB
    "${METALROBO_BUILD_DIR}/shaders/MetalRobo.metallib"
)
if(NOT EXISTS "${METALROBO_METALLIB}")
    message(
        FATAL_ERROR
        "Build MetalRobo first; missing ${METALROBO_METALLIB}"
    )
endif()

add_library(
    metalrobo_mlx
    STATIC
    metalrobo_mlx.cpp
)
target_include_directories(
    metalrobo_mlx
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}"
        "${METALROBO_ROOT}/include"
)
target_link_libraries(
    metalrobo_mlx
    PUBLIC mlx "${METALROBO_LIBRARY}"
)
target_compile_definitions(
    metalrobo_mlx
    PRIVATE
        METALROBO_DEFAULT_METALLIB="${METALROBO_METALLIB}"
)

if(MLX_BUILD_METAL)
    mlx_build_metallib(
        TARGET metalrobo_mlx_metallib
        TITLE MetalRoboMLX
        SOURCES
            "${CMAKE_CURRENT_LIST_DIR}/metalrobo_mlx.metal"
        INCLUDE_DIRS
            "${METALROBO_ROOT}/include"
            ${MLX_INCLUDE_DIRS}
        DEPS
            "${METALROBO_ROOT}/include/metalrobo/engine_types.h"
            "${METALROBO_ROOT}/include/metalrobo/generalized_constraint_shared.h"
            "${METALROBO_ROOT}/include/metalrobo/gpu_types.h"
            "${METALROBO_ROOT}/include/metalrobo/tactile_types.h"
        OUTPUT_DIRECTORY
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )
    add_dependencies(
        metalrobo_mlx
        metalrobo_mlx_metallib
    )
endif()

nanobind_add_module(
    _mlx_ext
    NB_STATIC
    STABLE_ABI
    LTO
    NOMINSIZE
    NB_DOMAIN mlx
    bindings.cpp
)
target_link_libraries(_mlx_ext PRIVATE metalrobo_mlx)
target_link_options(
    _mlx_ext
    PRIVATE
        "-Wl,-rpath,${METALROBO_BUILD_DIR}/lib"
        "-Wl,-rpath,@loader_path"
)
