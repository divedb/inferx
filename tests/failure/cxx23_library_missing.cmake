# Fixture: C++23 library facility missing. Configures
# cleanly but the compile must fail with the named missing facility; the
# sentinel's static asserts name std::expected and the toolchain guidance.
cmake_minimum_required(VERSION 3.28)
project(cxx23_library_missing_fixture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_executable(cxx23_probe "${INFERX_ROOT}/tests/failure/cxx23_library_missing_probe.cc")
target_include_directories(cxx23_probe PRIVATE "${INFERX_ROOT}/tests/compile")
