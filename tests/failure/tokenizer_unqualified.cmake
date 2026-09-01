cmake_minimum_required(VERSION 3.28)
project(tokenizer_unqualified_fixture LANGUAGES C CXX)

set(BUILD_TESTING OFF)
set(INFERX_BUILD_BENCHMARKS OFF)
set(INFERX_ENABLE_TOKENIZATION ON)
set(INFERX_DEPENDENCY_PROVIDER submodule)
set(INFERX_THIRD_PARTY_DIR "${INFERX_ROOT}/third_party")
include("${INFERX_ROOT}/cmake/InferXDependencies.cmake")
