# InferX source/system dependency acquisition (ADR 0003).
#
# CMake never downloads or runs Git. In 'submodule' provider mode each required
# dependency must already be initialized (tools/deps/bootstrap.py); missing
# ones fail configuration with the exact remediation. In 'system' mode,
# dependencies come from find_package with enforced minimums (packagers).
#
# INFERX_THIRD_PARTY_DIR exists so expected-failure fixtures can point the
# checks at an empty directory; production builds always use ../third_party.
set(INFERX_THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party" CACHE PATH
    "Directory holding the pinned source dependencies (fixtures may override).")

# ---------------------------------------------------------------------------
# Provider validation: fail with the valid values, never with a guess.
# ---------------------------------------------------------------------------
set(_INFERX_PROVIDERS submodule system)
if(NOT INFERX_DEPENDENCY_PROVIDER IN_LIST _INFERX_PROVIDERS)
  message(FATAL_ERROR
    "INFERX_DEPENDENCY_PROVIDER is '${INFERX_DEPENDENCY_PROVIDER}' but valid "
    "values are: ${_INFERX_PROVIDERS}. See docs/dependency-policy.md.")
endif()

# ---------------------------------------------------------------------------
# Scoped dependency option handling.
#
# inferx_dependency_scope_push()/..._pop() save the listed variables and
# restore them, so options set for one dependency can never leak into a
# sibling dependency or an embedding parent project. No CACHE FORCE is used
# anywhere; dependencies read normal variables set here.
# ---------------------------------------------------------------------------
set(_INFERX_DEP_SCOPE_DEPTH 0)

macro(inferx_dependency_scope_push)
  math(EXPR _INFERX_DEP_SCOPE_DEPTH "${_INFERX_DEP_SCOPE_DEPTH} + 1")
  foreach(_var IN ITEMS ${ARGN})
    if(DEFINED ${_var})
      set("_scope_${_INFERX_DEP_SCOPE_DEPTH}_${_var}" "${${_var}}")
    else()
      set("_scope_${_INFERX_DEP_SCOPE_DEPTH}_${_var}_UNSET" TRUE)
    endif()
  endforeach()
endmacro()

macro(inferx_dependency_scope_pop)
  foreach(_var IN ITEMS ${ARGN})
    if(DEFINED "_scope_${_INFERX_DEP_SCOPE_DEPTH}_${_var}")
      set(${_var} "${_scope_${_INFERX_DEP_SCOPE_DEPTH}_${_var}}")
      unset("_scope_${_INFERX_DEP_SCOPE_DEPTH}_${_var}")
    else()
      unset(${_var})
      unset("_scope_${_INFERX_DEP_SCOPE_DEPTH}_${_var}_UNSET")
    endif()
  endforeach()
  math(EXPR _INFERX_DEP_SCOPE_DEPTH "${_INFERX_DEP_SCOPE_DEPTH} - 1")
endmacro()

# Report an uninitialized required dependency with the exact remediation.
function(inferx_fail_missing_dependency name relative_path profile)
  message(FATAL_ERROR
    "Required source dependency '${name}' is not initialized under "
    "${INFERX_THIRD_PARTY_DIR} (expected ${INFERX_THIRD_PARTY_DIR}/${relative_path}). "
    "Run: python3 tools/deps/bootstrap.py --profile ${profile} "
    "(see docs/dependency-policy.md). CMake does not download dependencies.")
endfunction()

# ---------------------------------------------------------------------------
# Core profile dependencies.
#
# Abseil, simdjson, and BLAKE3 are production requirements for the M3 artifact
# reader. GoogleTest remains test-only and Google Benchmark benchmark-only.
# ---------------------------------------------------------------------------
set(_INFERX_CORE_PROFILE core)

if(_INFERX_CORE_PROFILE)
  # --- Abseil ---------------------------------------------------------------
  if(TRUE)
    if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
      if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/abseil-cpp/CMakeLists.txt")
        inferx_fail_missing_dependency(abseil-cpp abseil-cpp ${_INFERX_CORE_PROFILE})
      endif()
      inferx_dependency_scope_push(ABSL_PROPAGATE_CXX_STD ABSL_ENABLE_INSTALL
                                   ABSL_RUN_TESTS ABSL_BUILD_TESTING_HELPERS)
      set(ABSL_PROPAGATE_CXX_STD ON)
      # Dependency tests stay off for ordinary builds; upstream's install rules
      # are enabled only inside the dedicated installed-dependency fixture.
      set(ABSL_ENABLE_INSTALL OFF)
      set(ABSL_RUN_TESTS OFF)
      set(ABSL_BUILD_TESTING_HELPERS OFF)
      add_subdirectory("${INFERX_THIRD_PARTY_DIR}/abseil-cpp"
                       "${CMAKE_BINARY_DIR}/third_party/abseil-cpp"
                       SYSTEM EXCLUDE_FROM_ALL)
      inferx_dependency_scope_pop(ABSL_PROPAGATE_CXX_STD ABSL_ENABLE_INSTALL
                                  ABSL_RUN_TESTS ABSL_BUILD_TESTING_HELPERS)
    else()
      # System provider: minimum enforced only when the package reports a
      # version (upstream live-at-head packages do not).
      find_package(absl CONFIG REQUIRED)
      if(absl_VERSION AND absl_VERSION VERSION_LESS 20240722.0)
        message(FATAL_ERROR
          "System absl ${absl_VERSION} is older than the accepted minimum "
          "20240722.0 (the submodule pin is the reference; see "
          "docs/dependencies/abseil-cpp.md).")
      endif()
    endif()
  endif()

  # --- simdjson -------------------------------------------------------------
  if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
    if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/simdjson/CMakeLists.txt")
      inferx_fail_missing_dependency(simdjson simdjson ${_INFERX_CORE_PROFILE})
    endif()
    inferx_dependency_scope_push(BUILD_SHARED_LIBS SIMDJSON_INSTALL
                                 SIMDJSON_ENABLE_THREADS
                                 SIMDJSON_DISABLE_DEPRECATED_API
                                 SIMDJSON_DEVELOPER_MODE)
    set(BUILD_SHARED_LIBS OFF)
    set(SIMDJSON_INSTALL OFF)
    set(SIMDJSON_ENABLE_THREADS OFF)
    set(SIMDJSON_DISABLE_DEPRECATED_API ON)
    set(SIMDJSON_DEVELOPER_MODE OFF)
    add_subdirectory("${INFERX_THIRD_PARTY_DIR}/simdjson"
                     "${CMAKE_BINARY_DIR}/third_party/simdjson"
                     SYSTEM EXCLUDE_FROM_ALL)
    inferx_dependency_scope_pop(BUILD_SHARED_LIBS SIMDJSON_INSTALL
                                SIMDJSON_ENABLE_THREADS
                                SIMDJSON_DISABLE_DEPRECATED_API
                                SIMDJSON_DEVELOPER_MODE)
  else()
    find_package(simdjson 4.6.5 CONFIG REQUIRED)
  endif()

  # --- BLAKE3 portable C implementation ------------------------------------
  if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
    if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/blake3/c/blake3.c")
      inferx_fail_missing_dependency(blake3 blake3 ${_INFERX_CORE_PROFILE})
    endif()
    add_library(inferx_blake3 STATIC
      "${INFERX_THIRD_PARTY_DIR}/blake3/c/blake3.c"
      "${INFERX_THIRD_PARTY_DIR}/blake3/c/blake3_dispatch.c"
      "${INFERX_THIRD_PARTY_DIR}/blake3/c/blake3_portable.c")
    target_include_directories(inferx_blake3 SYSTEM PUBLIC
      "${INFERX_THIRD_PARTY_DIR}/blake3/c")
    target_compile_definitions(inferx_blake3 PRIVATE
      BLAKE3_NO_SSE2 BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_NO_AVX512)
    set_target_properties(inferx_blake3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(blake3::blake3 ALIAS inferx_blake3)
  else()
    find_package(blake3 1.8.7 CONFIG REQUIRED)
  endif()

  if(INFERX_ENABLE_TOKENIZATION)
    message(FATAL_ERROR
      "INFERX_ENABLE_TOKENIZATION is ON, but divedb/tokenizer f109b7a is "
      "candidate-only and fails the M3.0 qualification gate: local-only "
      "build, error-returning construction, and streaming decode ABI are "
      "required. Keep this option OFF until ADR 0024 records an approved pin.")
  endif()

  # --- GoogleTest -----------------------------------------------------------
  if(BUILD_TESTING)
    if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
      if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/googletest/CMakeLists.txt")
        inferx_fail_missing_dependency(googletest googletest ${_INFERX_CORE_PROFILE})
      endif()
      inferx_dependency_scope_push(INSTALL_GTEST BUILD_GMOCK
                                   googletest_disable_pthreads)
      set(INSTALL_GTEST OFF)
      # M0 declares GoogleMock part of the test dependency surface.
      set(BUILD_GMOCK ON)
      add_subdirectory("${INFERX_THIRD_PARTY_DIR}/googletest"
                       "${CMAKE_BINARY_DIR}/third_party/googletest"
                       SYSTEM EXCLUDE_FROM_ALL)
      inferx_dependency_scope_pop(INSTALL_GTEST BUILD_GMOCK
                                  googletest_disable_pthreads)
    else()
      find_package(GTest CONFIG REQUIRED)
      if(GTest_VERSION AND GTest_VERSION VERSION_LESS 1.14.0)
        message(FATAL_ERROR
          "System GTest ${GTest_VERSION} is older than the accepted minimum "
          "1.14.0 (submodule pin: v1.18.0).")
      endif()
    endif()
  endif()

  # --- Google Benchmark -----------------------------------------------------
  if(INFERX_BUILD_BENCHMARKS)
    if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
      if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/benchmark/CMakeLists.txt")
        inferx_fail_missing_dependency(benchmark benchmark ${_INFERX_CORE_PROFILE})
      endif()
      inferx_dependency_scope_push(BENCHMARK_ENABLE_TESTING
                                   BENCHMARK_ENABLE_GTEST_TESTS
                                   BENCHMARK_ENABLE_INSTALL
                                   BENCHMARK_ENABLE_DOXYGEN
                                   BENCHMARK_INSTALL_DOCS)
      set(BENCHMARK_ENABLE_TESTING OFF)
      set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
      set(BENCHMARK_ENABLE_INSTALL OFF)
      set(BENCHMARK_ENABLE_DOXYGEN OFF)
      set(BENCHMARK_INSTALL_DOCS OFF)
      add_subdirectory("${INFERX_THIRD_PARTY_DIR}/benchmark"
                       "${CMAKE_BINARY_DIR}/third_party/benchmark"
                       SYSTEM EXCLUDE_FROM_ALL)
      inferx_dependency_scope_pop(BENCHMARK_ENABLE_TESTING
                                  BENCHMARK_ENABLE_GTEST_TESTS
                                  BENCHMARK_ENABLE_INSTALL
                                  BENCHMARK_ENABLE_DOXYGEN
                                  BENCHMARK_INSTALL_DOCS)
    else()
      find_package(benchmark CONFIG REQUIRED)
      if(benchmark_VERSION AND benchmark_VERSION VERSION_LESS 1.8.0)
        message(FATAL_ERROR
          "System benchmark ${benchmark_VERSION} is older than the accepted "
          "minimum 1.8.0 (submodule pin: v1.9.5).")
      endif()
    endif()
  endif()
endif()
