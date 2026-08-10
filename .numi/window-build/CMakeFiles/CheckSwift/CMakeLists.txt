cmake_minimum_required(VERSION 4.0.1)
set(CMAKE_MODULE_PATH "")
project(CheckSwift Swift)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
  "set(CMAKE_Swift_COMPILER \"${CMAKE_Swift_COMPILER}\")\n"
  
  )