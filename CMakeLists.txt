cmake_minimum_required(VERSION 3.16)
project(insaloSAT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(insaloSAT
    "src/cpp/subsystems/Controller.cpp"
    "src/cpp/subsystems/Stepper.cpp"
)

target_include_directories(insaloSAT PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include "
)
