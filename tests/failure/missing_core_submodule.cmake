# Fixture: a required core submodule is missing. The
# dependency module must fail with the exact bootstrap remediation.
cmake_minimum_required(VERSION 3.28)
project(missing_core_submodule_fixture LANGUAGES CXX)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/empty_third_party")
set(INFERX_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/empty_third_party")
set(INFERX_DEPENDENCY_PROVIDER "submodule")
set(BUILD_TESTING ON)
include("${INFERX_ROOT}/cmake/InferXDependencies.cmake")
