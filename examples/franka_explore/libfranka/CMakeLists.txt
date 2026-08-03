cmake_minimum_required(VERSION 3.16)

project(NumiFrankaLibfrankaExplore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Franka CONFIG REQUIRED)

add_executable(
    numi_franka_libfranka_explore
    ../libfranka_main.cpp
)
target_include_directories(
    numi_franka_libfranka_explore
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(
    numi_franka_libfranka_explore
    PRIVATE
        Franka::Franka
)
target_compile_options(
    numi_franka_libfranka_explore
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
)
