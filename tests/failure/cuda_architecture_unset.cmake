# Fixture: CUDA architecture unset/outside the accepted list. Validated before
# toolkit detection so the failure is actionable on any
# machine; must print the accepted architecture list.
cmake_minimum_required(VERSION 3.28)
project(cuda_architecture_unset_fixture LANGUAGES CXX)

set(INFERX_ENABLE_CUDA ON)
set(CMAKE_CUDA_ARCHITECTURES "")
include("${INFERX_ROOT}/cmake/InferXCuda.cmake")
