cmake_minimum_required(VERSION 3.28)

project(InferX VERSION 0.0.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Language policy: C++23, no extensions, required (never silently falls back).
# CUDA is enabled only by the CUDA module; this file never touches it.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Project policy: options, warnings, sanitizers, dependencies, install rules.
# Order matters: options before include(CTest) so BUILD_TESTING's default is
# ours (top-level ON, subproject OFF).
# ---------------------------------------------------------------------------
include(cmake/InferXProjectOptions.cmake)
include(cmake/InferXSanitizers.cmake)
include(GNUInstallDirs)
include(CTest)
include(cmake/InferXDependencies.cmake)
include(cmake/InferXInstall.cmake)

# ---------------------------------------------------------------------------
# First-party subdirectories only. Optional dependency directories are never
# inspected unless their feature is enabled (see InferXDependencies.cmake).
# ---------------------------------------------------------------------------
add_subdirectory(src/base)
add_subdirectory(src/config)
add_subdirectory(src/api)
add_subdirectory(src/lifecycle)
add_subdirectory(src/scheduler)
add_subdirectory(src/engine)
add_subdirectory(src/tensor)
add_subdirectory(src/runtime)
add_subdirectory(src/artifacts)
add_subdirectory(src/model)
add_subdirectory(src/tokenization)
add_subdirectory(src/input)
add_subdirectory(src/ops)
add_subdirectory(src/model_adapters)
add_subdirectory(simulator)

if(INFERX_BUILD_TOOLS)
  add_subdirectory(apps/inferx_info)
  add_subdirectory(apps/inferx_model_inspect)
  add_subdirectory(apps/inferx_sim)
  add_subdirectory(apps/inferx_fsm_schema)
endif()

if(INFERX_ENABLE_CUDA)
  include(cmake/InferXCuda.cmake)
endif()

# Unified kernels layer (ADR 0031): the neutral dispatch target builds on
# every preset; CUDA categories register only under INFERX_ENABLE_CUDA and
# after InferXCuda.cmake has set the CUDA language variables.
add_subdirectory(kernels)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(INFERX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Format targets: 'format' mutates tracked InferX
# sources; 'format-check' never mutates and is what CI runs.
# ---------------------------------------------------------------------------
add_custom_target(format
  COMMAND ${CMAKE_COMMAND} -E env INFERX_CLANG_FORMAT=${INFERX_CLANG_FORMAT_ENV}
          ${CMAKE_SOURCE_DIR}/tools/check_format.sh --fix
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  USES_TERMINAL
  COMMENT "InferX format: rewrite tracked sources (clang-format major 18)")

add_custom_target(format-check
  COMMAND ${CMAKE_SOURCE_DIR}/tools/check_format.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  USES_TERMINAL
  COMMENT "InferX format-check: verify formatting without mutating")
