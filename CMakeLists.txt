cmake_minimum_required(VERSION 3.12)
project(pla-compressor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
add_compile_options(-O3)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options(-march=native)
endif()

add_library(pla-compressor INTERFACE)
target_include_directories(pla-compressor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/sdsl-lite/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/sux
)
target_link_libraries(pla-compressor INTERFACE OpenMP::OpenMP_CXX)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE pla-compressor)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE pla-compressor)