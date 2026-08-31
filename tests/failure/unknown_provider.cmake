# Fixture: unknown dependency provider (m0.md 13.5 row 2). Must fail listing
# the valid values rather than guessing.
cmake_minimum_required(VERSION 3.28)
project(unknown_provider_fixture LANGUAGES CXX)

set(INFERX_DEPENDENCY_PROVIDER "vendored")
include("${INFERX_ROOT}/cmake/InferXDependencies.cmake")
