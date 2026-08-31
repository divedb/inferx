# Fixture: CUDA requested but the toolkit is absent (m0.md 13.5 row 6). The
# bogus compiler path forces detection failure even on machines that have a
# toolkit; configuration must fail instead of silently disabling CUDA.
cmake_minimum_required(VERSION 3.28)
project(cuda_toolkit_missing_fixture LANGUAGES CXX)

set(INFERX_ENABLE_CUDA ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER "/nonexistent/bin/nvcc")
include("${INFERX_ROOT}/cmake/InferXCuda.cmake")
