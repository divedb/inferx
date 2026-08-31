# Optional, strict CUDA enablement (m0.md section 10.1; ADR 0005). Included by
# the top-level CMakeLists only when INFERX_ENABLE_CUDA=ON. Order matters:
#
#   1. explicit architecture list is required BEFORE toolkit detection, so an
#      unset/unsupported list fails with actionable guidance on any machine;
#   2. check_language(CUDA) — absent toolkit is fatal (never silently off);
#   3. enable_language + find_package(CUDAToolkit REQUIRED);
#   4. language level: C++23 when the qualified NVCC supports it, otherwise
#      the CUDA subset is isolated at C++20 (ADR 0005 section 4);
#   5. host-compiler/toolkit pairing validated where known-incompatible.

# Accepted architecture list for the owned GPU lane (m0.md section 6.6: never
# 'native' in presets/CI). Developers override via CMakeUserPresets.json.
set(INFERX_CUDA_ACCEPTED_ARCHITECTURES "89" CACHE STRING
    "GPU compute capabilities InferX accepts (SM list, e.g. 89;90).")

if(NOT CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR
    "INFERX_ENABLE_CUDA=ON requires an explicit CMAKE_CUDA_ARCHITECTURES from "
    "the accepted list '${INFERX_CUDA_ACCEPTED_ARCHITECTURES}' ('native' is "
    "forbidden in release/CI). Example: -DCMAKE_CUDA_ARCHITECTURES=89. See "
    "docs/supported-platforms.md.")
endif()

separate_arguments(_INFERX_REQUESTED_ARCHS NATIVE_COMMAND "${CMAKE_CUDA_ARCHITECTURES}")
separate_arguments(_INFERX_ACCEPTED_ARCHS NATIVE_COMMAND "${INFERX_CUDA_ACCEPTED_ARCHITECTURES}")
foreach(_arch IN ITEMS ${_INFERX_REQUESTED_ARCHS})
  if(NOT _arch IN_LIST _INFERX_ACCEPTED_ARCHS)
    message(FATAL_ERROR
      "CUDA architecture '${_arch}' is outside the accepted list "
      "'${INFERX_CUDA_ACCEPTED_ARCHITECTURES}'. Update the support matrix via "
      "ADR 0005 or choose an accepted value; see docs/supported-platforms.md.")
  endif()
endforeach()

include(CheckLanguage)
# If a CUDA compiler was explicitly requested, validate it ourselves first so
# the failure carries InferX guidance instead of a bare enable_language error.
if(DEFINED CMAKE_CUDA_COMPILER)
  find_program(_INFERX_CUDA_COMPILER_CHECK NAMES "${CMAKE_CUDA_COMPILER}")
  if(NOT _INFERX_CUDA_COMPILER_CHECK)
    message(FATAL_ERROR
      "INFERX_ENABLE_CUDA=ON but the requested CUDA compiler "
      "'${CMAKE_CUDA_COMPILER}' does not exist. Install the accepted CUDA "
      "toolkit (ADR 0005; see docker/cuda-dev.Dockerfile and "
      "docs/supported-platforms.md) and ensure nvcc is on PATH or "
      "CMAKE_CUDA_COMPILER points at it. CUDA is never silently disabled.")
  endif()
  unset(_INFERX_CUDA_COMPILER_CHECK CACHE)
endif()
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "INFERX_ENABLE_CUDA=ON but no usable CUDA compiler/toolkit was found. "
    "Install the accepted toolkit (ADR 0005; see docker/cuda-dev.Dockerfile "
    "and docs/supported-platforms.md) and ensure nvcc is on PATH or "
    "CMAKE_CUDA_COMPILER points at it. CUDA is never silently disabled.")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# NVCC language level (ADR 0005): C++23 where both the toolkit and CMake can
# express it, else the CUDA subset is isolated at C++20. The host project
# stays C++23 regardless. (CMake gained the cuda_std_23 feature after 3.28,
# so a 3.28 floor isolates at C++20 even on newer NVCC.)
list(FIND CMAKE_CUDA_COMPILE_FEATURES cuda_std_23 _INFERX_CUDA23_FEATURE_INDEX)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.4
    AND _INFERX_CUDA23_FEATURE_INDEX GREATER -1)
  set(INFERX_CUDA_CXX_STANDARD 23)
else()
  set(INFERX_CUDA_CXX_STANDARD 20)
  message(STATUS
    "InferX CUDA: isolating .cu translation units at C++20 (NVCC "
    "${CMAKE_CUDA_COMPILER_VERSION}, CMake ${CMAKE_VERSION}); ADR 0005.")
endif()
unset(_INFERX_CUDA23_FEATURE_INDEX)
set(CMAKE_CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Host-compiler pairing: no speculative rejection — local evidence shows the
# CUDA 12.0.140 + GCC 13 pairing compiles and runs this smoke (recorded in
# docs/supported-platforms.md as an experimental lane). NVCC's own unsupported-
# host error is authoritative if a future combination regresses.
