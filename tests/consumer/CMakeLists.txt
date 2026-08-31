# Independent InferX consumer (m0.md section 6.7). A separate CMake project:
# no add_subdirectory, no source-tree paths, no build-tree targets. It finds
# InferX only through CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 3.28)
project(inferx_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(InferX CONFIG REQUIRED)

add_executable(inferx_consumer main.cc)
target_link_libraries(inferx_consumer PRIVATE inferx::base)
