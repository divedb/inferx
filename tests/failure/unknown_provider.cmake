# Fixture: unknown dependency provider. Must fail listing
# the valid values rather than guessing.
cmake_minimum_required(VERSION 3.28)
project(unknown_provider_fixture LANGUAGES CXX)

set(INFERX_DEPENDENCY_PROVIDER "vendored")
include("${INFERX_ROOT}/cmake/InferXDependencies.cmake")
