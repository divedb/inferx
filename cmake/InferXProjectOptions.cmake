# InferX project options: compiler floor, cache options, target policy libraries,
# and the single helper that applies InferX policy to an owned target.
#
# Includes: called once from the top-level CMakeLists.txt (and from dependency
# fixtures that intentionally reuse the checks).

# ---------------------------------------------------------------------------
# Compiler floor (ADR 0005). Fail at configure time, before project sources
# compile, with the minimum accepted version.
# ---------------------------------------------------------------------------
set(INFERX_MIN_GCC_VERSION 13)
set(INFERX_MIN_CLANG_VERSION 18)

if(NOT DEFINED INFERX_SKIP_COMPILER_FLOOR_CHECK)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
      AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS INFERX_MIN_GCC_VERSION)
    message(FATAL_ERROR
      "InferX requires GCC ${INFERX_MIN_GCC_VERSION} or newer; found GCC "
      "${CMAKE_CXX_COMPILER_VERSION}. See docs/supported-platforms.md and "
      "docker/cpu-dev.Dockerfile for the accepted toolchain.")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
      AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS INFERX_MIN_CLANG_VERSION)
    message(FATAL_ERROR
      "InferX requires Clang ${INFERX_MIN_CLANG_VERSION} or newer; found Clang "
      "${CMAKE_CXX_COMPILER_VERSION}. See docs/supported-platforms.md and "
      "docker/cpu-dev.Dockerfile for the accepted toolchain.")
  elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR
      "Compiler '${CMAKE_CXX_COMPILER_ID}' is not on the supported platform "
      "matrix (GCC >= ${INFERX_MIN_GCC_VERSION} or Clang >= "
      "${INFERX_MIN_CLANG_VERSION}). See docs/supported-platforms.md.")
  endif()
endif()

# ---------------------------------------------------------------------------
# Cache options (defaults depend on whether InferX is the top-level project).
# Every option must remain visible and documented in `cmake -LH` output.
# ---------------------------------------------------------------------------
set(_INFERX_TOP_LEVEL_DEFAULT ON)
if(NOT PROJECT_IS_TOP_LEVEL)
  set(_INFERX_TOP_LEVEL_DEFAULT OFF)
endif()

# BUILD_TESTING is CTest's switch and the single source of truth for test
# targets; default it before include(CTest) runs in the top-level file.
if(NOT DEFINED BUILD_TESTING)
  set(BUILD_TESTING ${_INFERX_TOP_LEVEL_DEFAULT} CACHE BOOL
      "Build InferX tests (CTest standard switch).")
endif()

option(INFERX_BUILD_BENCHMARKS "Build InferX benchmarks (Google Benchmark smoke in M0)."
       ${_INFERX_TOP_LEVEL_DEFAULT})
option(INFERX_BUILD_TOOLS "Build InferX developer tools such as inferx-info."
       ${_INFERX_TOP_LEVEL_DEFAULT})
option(INFERX_ENABLE_CUDA "Enable the optional CUDA platform (explicit opt-in; a missing toolkit is fatal when ON)."
       OFF)
option(INFERX_BUILD_GPU_TESTS "Build M2 GPU tests (requires INFERX_ENABLE_CUDA)."
       ${_INFERX_TOP_LEVEL_DEFAULT})
option(INFERX_BUILD_COMPUTE_SANITIZER_TESTS
       "Register M2 Compute Sanitizer tests (requires CUDA and compute-sanitizer)." OFF)
option(INFERX_CUDA_ENABLE_LINEINFO
       "Compile InferX CUDA kernels with device line information." ON)
if(INFERX_BUILD_GPU_TESTS AND NOT INFERX_ENABLE_CUDA)
  # GPU tests default with top-level builds but remain dormant in CPU builds.
  set(INFERX_BUILD_GPU_TESTS OFF CACHE BOOL
      "Build M2 GPU tests (requires INFERX_ENABLE_CUDA)." FORCE)
endif()
if(INFERX_BUILD_COMPUTE_SANITIZER_TESTS AND NOT INFERX_ENABLE_CUDA)
  message(FATAL_ERROR
    "INFERX_BUILD_COMPUTE_SANITIZER_TESTS=ON requires INFERX_ENABLE_CUDA=ON")
endif()
option(INFERX_WARNINGS_AS_ERRORS "Treat warnings as errors on InferX-owned targets only."
       ${_INFERX_TOP_LEVEL_DEFAULT})
option(INFERX_ENABLE_CLANG_TIDY "Run clang-tidy on InferX-owned targets (analysis preset)."
       OFF)
option(INFERX_ENABLE_IWYU "Run include-what-you-use on InferX-owned targets (analysis preset)."
       OFF)
option(INFERX_ENABLE_ASAN "Enable AddressSanitizer (CPU-only in M0; mutually exclusive with TSan)."
       OFF)
option(INFERX_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer (combined with ASan in the standard preset)."
       OFF)
option(INFERX_ENABLE_TSAN "Enable ThreadSanitizer (CPU-only and mutually exclusive with ASan)."
       OFF)
set(INFERX_DEPENDENCY_PROVIDER "submodule" CACHE STRING
    "How InferX obtains source dependencies: 'submodule' (pinned gitlinks) or 'system' (find_package).")
set_property(CACHE INFERX_DEPENDENCY_PROVIDER PROPERTY
             STRINGS submodule system)

# Optional absolute paths for the analysis tools; empty means autodetect.
set(INFERX_CLANG_TIDY_BIN "" CACHE STRING
    "clang-tidy binary to use with INFERX_ENABLE_CLANG_TIDY (default: autodetect, major version 18).")
set(INFERX_IWYU_BIN "" CACHE STRING
    "include-what-you-use binary to use with INFERX_ENABLE_IWYU (default: autodetect).")

# ---------------------------------------------------------------------------
# Target-scoped policy libraries. Nothing here is global; every InferX-owned
# target attaches them through inferx_apply_project_options().
# ---------------------------------------------------------------------------
add_library(inferx_project_options INTERFACE)
add_library(inferx::project_options ALIAS inferx_project_options)

add_library(inferx_project_warnings INTERFACE)
add_library(inferx::project_warnings ALIAS inferx_project_warnings)

set(_INFERX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wcast-qual
    -Wformat=2
    -Wundef
    -Wnon-virtual-dtor
    -Woverloaded-virtual)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # CXX-only: host warning flags must not be forwarded through NVCC (CUDA
  # warning flags are selected separately; m0.md section 6.5).
  target_compile_options(inferx_project_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${_INFERX_WARNING_FLAGS}>)
  if(INFERX_WARNINGS_AS_ERRORS)
    target_compile_options(inferx_project_warnings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()
# Deliberately no CUDA forwarding here: NVCC warning flags are selected in
# platform/cuda/smoke because blindly forwarding host flags is fragile.

# ---------------------------------------------------------------------------
# inferx_apply_project_options(<target>)
#
# Applies InferX policy to a library, executable, test, benchmark, or CUDA host
# target. Attaching warnings and sanitizers to final executables as well keeps
# sanitizer runtime link flags from being lost when an instrumented static
# library is consumed.
# ---------------------------------------------------------------------------
function(inferx_apply_project_options target)
  # BUILD_INTERFACE keeps build-only policy libraries out of the installed
  # export of a static library (m0.md section 6.4).
  target_link_libraries(${target} PRIVATE
    $<BUILD_INTERFACE:inferx_project_options>
    $<BUILD_INTERFACE:inferx_project_warnings>)
  if(TARGET inferx_sanitizers)
    target_link_libraries(${target} PRIVATE $<BUILD_INTERFACE:inferx_sanitizers>)
  endif()

  if(INFERX_ENABLE_CLANG_TIDY)
    if(NOT DEFINED INFERX_CLANG_TIDY_COMMAND)
      message(FATAL_ERROR
        "Internal error: INFERX_ENABLE_CLANG_TIDY is ON but the tidy command "
        "was not resolved; InferXSanitizers/analysis setup should have failed "
        "earlier if the tool is absent.")
    endif()
    set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${INFERX_CLANG_TIDY_COMMAND}")
  endif()
  if(INFERX_ENABLE_IWYU)
    if(NOT DEFINED INFERX_IWYU_COMMAND)
      message(FATAL_ERROR
        "Internal error: INFERX_ENABLE_IWYU is ON but the IWYU command was not "
        "resolved; analysis setup should have failed earlier if the tool is absent.")
    endif()
    set_target_properties(${target} PROPERTIES
      CXX_INCLUDE_WHAT_YOU_USE "${INFERX_IWYU_COMMAND}")
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Analysis tool resolution (fail clearly when requested and absent).
# ---------------------------------------------------------------------------
if(INFERX_ENABLE_CLANG_TIDY)
  if(INFERX_CLANG_TIDY_BIN)
    find_program(INFERX_CLANG_TIDY_FOUND NAMES ${INFERX_CLANG_TIDY_BIN})
  else()
    find_program(INFERX_CLANG_TIDY_FOUND
                 NAMES clang-tidy clang-tidy-18 clang-tidy-19)
  endif()
  if(NOT INFERX_CLANG_TIDY_FOUND)
    set(_INFERX_EXPECTED_TIDY
        "expected binary: ${INFERX_CLANG_TIDY_BIN}; accepted major version: 18")
    message(FATAL_ERROR
      "INFERX_ENABLE_CLANG_TIDY is ON but clang-tidy was not found "
      "(${_INFERX_EXPECTED_TIDY}). Install clang-tidy-18 or point "
      "INFERX_CLANG_TIDY_BIN at a compatible binary; see "
      "docs/supported-platforms.md.")
  endif()
  set(INFERX_CLANG_TIDY_COMMAND "${INFERX_CLANG_TIDY_FOUND}")
endif()

if(INFERX_ENABLE_IWYU)
  if(INFERX_IWYU_BIN)
    find_program(INFERX_IWYU_FOUND NAMES ${INFERX_IWYU_BIN})
  else()
    find_program(INFERX_IWYU_FOUND NAMES include-what-you-use iwyu)
  endif()
  if(NOT INFERX_IWYU_FOUND)
    message(FATAL_ERROR
      "INFERX_ENABLE_IWYU is ON but include-what-you-use was not found. "
      "Install include-what-you-use (qualified version: 0.21) or point "
      "INFERX_IWYU_BIN at a binary; see docs/supported-platforms.md.")
  endif()
  set(INFERX_IWYU_COMMAND
      "${INFERX_IWYU_FOUND};-Xiwyu;--no_fwd_decls")
endif()
