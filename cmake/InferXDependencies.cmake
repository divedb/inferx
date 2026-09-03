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
# Abseil, simdjson, BLAKE3, and (when enabled) curl are production requirements
# for artifact/model resolution. GoogleTest remains test-only and Google
# Benchmark benchmark-only.
# ---------------------------------------------------------------------------
# InferX exposes Abseil through inferx::base public headers (ADR 0008), so the
# core profile is an unconditional prerequisite; GoogleTest/Benchmark remain
# gated by their options below.
set(_INFERX_CORE_PROFILE core)

if(_INFERX_CORE_PROFILE)
  # --- Abseil (unconditional: public dependency of inferx::base) ------------
  if(TRUE)
    if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
      if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/abseil-cpp/CMakeLists.txt")
        inferx_fail_missing_dependency(abseil-cpp abseil-cpp ${_INFERX_CORE_PROFILE})
      endif()
      inferx_dependency_scope_push(ABSL_PROPAGATE_CXX_STD ABSL_ENABLE_INSTALL
                                   ABSL_RUN_TESTS ABSL_BUILD_TESTING_HELPERS)
      set(ABSL_PROPAGATE_CXX_STD ON)
      # Dependency tests stay off; upstream's install rules stay off for the
      # ordinary build (install mode renames Abseil targets to unprefixed
      # names, e.g. a `check` library that collides with our aggregate
      # target). The installed package instead resolves Abseil by name via
      # find_dependency(absl CONFIG) against the standalone-installed pinned
      # Abseil in the same prefix when installed together.
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

  # --- simdjson (unconditional: config and artifact parsing are core) --------
  if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
    if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/simdjson/CMakeLists.txt")
      inferx_fail_missing_dependency(simdjson simdjson ${_INFERX_CORE_PROFILE})
    endif()
    inferx_dependency_scope_push(BUILD_SHARED_LIBS SIMDJSON_INSTALL
                                 SIMDJSON_ENABLE_THREADS
                                 SIMDJSON_DISABLE_DEPRECATED_API
                                 SIMDJSON_DEVELOPER_MODE)
    set(BUILD_SHARED_LIBS OFF)
    set(SIMDJSON_INSTALL ON)
    set(SIMDJSON_ENABLE_THREADS OFF)
    set(SIMDJSON_DISABLE_DEPRECATED_API ON)
    set(SIMDJSON_DEVELOPER_MODE OFF)
    # Keep simdjson's install rules in the top-level install graph. Installed
    # static inferx::config/model/artifacts targets retain a link-only
    # simdjson::simdjson dependency that InferXConfig resolves from this same
    # prefix.
    add_subdirectory("${INFERX_THIRD_PARTY_DIR}/simdjson"
                     "${CMAKE_BINARY_DIR}/third_party/simdjson"
                     SYSTEM)
    inferx_dependency_scope_pop(BUILD_SHARED_LIBS SIMDJSON_INSTALL
                                SIMDJSON_ENABLE_THREADS
                                SIMDJSON_DISABLE_DEPRECATED_API
                                SIMDJSON_DEVELOPER_MODE)
  else()
    find_package(simdjson CONFIG REQUIRED)
    if(simdjson_VERSION AND simdjson_VERSION VERSION_LESS 3.0.0)
      message(FATAL_ERROR
        "System simdjson ${simdjson_VERSION} is older than the accepted "
        "minimum 3.0.0 (submodule pin: v4.6.5); see docs/dependencies/simdjson.md")
    endif()
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
      $<BUILD_INTERFACE:${INFERX_THIRD_PARTY_DIR}/blake3/c>)
    target_compile_definitions(inferx_blake3 PRIVATE
      BLAKE3_NO_SSE2 BLAKE3_NO_SSE41 BLAKE3_NO_AVX2 BLAKE3_NO_AVX512)
    set_target_properties(inferx_blake3 PROPERTIES
      EXPORT_NAME blake3_internal
      POSITION_INDEPENDENT_CODE ON)
    add_library(blake3::blake3 ALIAS inferx_blake3)
  else()
    find_package(blake3 1.8.7 CONFIG REQUIRED)
  endif()

  if(INFERX_ENABLE_TOKENIZATION)
    # The qualified tokenizer backend (ADR 0024): an owned local-only
    # adaptation of divedb/tokenizer's C++ over an owned error-returning Rust
    # FFI shim on the Hugging Face `tokenizers` engine. nlohmann/json and
    # minja (chat-template closure) are pinned submodules; the crate closure
    # is vendored under third_party/tokenizer/rust/vendor and the cargo build
    # is offline.
    find_program(INFERX_CARGO_EXECUTABLE NAMES cargo)
    if(NOT INFERX_CARGO_EXECUTABLE)
      message(FATAL_ERROR
        "INFERX_ENABLE_TOKENIZATION is ON, but cargo was not found. The "
        "qualified tokenizer backend builds the Hugging Face tokenizers "
        "engine from the vendored crate closure "
        "(third_party/tokenizer/rust). Install Rust (rustup) so `cargo` is "
        "on PATH, or configure with -DINFERX_ENABLE_TOKENIZATION=OFF to "
        "build the artifact/model layers without the tokenizer.")
    endif()
    if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/nlohmann-json/single_include")
      inferx_fail_missing_dependency(nlohmann-json nlohmann-json core)
    endif()
    if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/minja/include/minja")
      inferx_fail_missing_dependency(minja minja core)
    endif()
    if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/tokenizer/CMakeLists.txt")
      inferx_fail_missing_dependency(tokenizer tokenizer core)
    endif()
    find_package(Threads REQUIRED)
    add_subdirectory("${INFERX_THIRD_PARTY_DIR}/tokenizer"
                     "${CMAKE_BINARY_DIR}/third_party/tokenizer")
  endif()

  # --- libcurl (native Hugging Face model resolution) ----------------------
  if(INFERX_ENABLE_HF_HUB)
    if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
      if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/curl/CMakeLists.txt")
        inferx_fail_missing_dependency(curl curl core)
      endif()
      inferx_dependency_scope_push(
        BUILD_CURL_EXE BUILD_EXAMPLES BUILD_LIBCURL_DOCS BUILD_MISC_DOCS
        BUILD_SHARED_LIBS BUILD_STATIC_LIBS BUILD_TESTING ENABLE_CURL_MANUAL
        CURL_BROTLI CURL_BUILD_EVERYTHING CURL_ENABLE_EXPORT_TARGET CURL_USE_LIBPSL
        CURL_ZLIB CURL_ZSTD CURL_USE_LIBSSH2 CURL_USE_OPENSSL HTTP_ONLY PICKY_COMPILER
        CMAKE_INSTALL_BINDIR)
      set(BUILD_CURL_EXE OFF)
      set(BUILD_EXAMPLES OFF)
      set(BUILD_LIBCURL_DOCS OFF)
      set(BUILD_MISC_DOCS OFF)
      set(BUILD_SHARED_LIBS OFF)
      set(BUILD_STATIC_LIBS ON)
      set(BUILD_TESTING OFF)
      set(ENABLE_CURL_MANUAL OFF)
      set(CURL_BUILD_EVERYTHING OFF)
      set(CURL_BROTLI OFF)
      set(CURL_ZLIB OFF)
      set(CURL_ZSTD OFF)
      set(CURL_ENABLE_EXPORT_TARGET ON)
      set(CURL_USE_LIBPSL OFF)
      set(CURL_USE_LIBSSH2 OFF)
      set(CURL_USE_OPENSSL ON)
      set(HTTP_ONLY ON)
      set(PICKY_COMPILER OFF)
      # curl installs a curl-config helper even when BUILD_CURL_EXE is OFF.
      # Keep dependency tooling out of InferX's user-facing binary directory.
      set(CMAKE_INSTALL_BINDIR "${CMAKE_INSTALL_LIBEXECDIR}/inferx/internal")
      find_package(OpenSSL 3.0 REQUIRED)
      add_subdirectory("${INFERX_THIRD_PARTY_DIR}/curl"
                       "${CMAKE_BINARY_DIR}/third_party/curl"
                       SYSTEM)
      inferx_dependency_scope_pop(
        BUILD_CURL_EXE BUILD_EXAMPLES BUILD_LIBCURL_DOCS BUILD_MISC_DOCS
        BUILD_SHARED_LIBS BUILD_STATIC_LIBS BUILD_TESTING ENABLE_CURL_MANUAL
        CURL_BROTLI CURL_BUILD_EVERYTHING CURL_ENABLE_EXPORT_TARGET CURL_USE_LIBPSL
        CURL_ZLIB CURL_ZSTD CURL_USE_LIBSSH2 CURL_USE_OPENSSL HTTP_ONLY PICKY_COMPILER
        CMAKE_INSTALL_BINDIR)
    else()
      find_package(CURL 8.21 REQUIRED)
    endif()
  endif()

  # --- CLI11 (the sole product command-line parser) -------------------------
  if(INFERX_BUILD_CLI)
    if(INFERX_DEPENDENCY_PROVIDER STREQUAL "submodule")
      if(NOT EXISTS "${INFERX_THIRD_PARTY_DIR}/CLI11/CMakeLists.txt")
        inferx_fail_missing_dependency(CLI11 CLI11 ${_INFERX_CORE_PROFILE})
      endif()
      inferx_dependency_scope_push(
        CLI11_BUILD_DOCS CLI11_BUILD_EXAMPLES CLI11_BUILD_EXAMPLES_JSON
        CLI11_BUILD_TESTS CLI11_INSTALL CLI11_PRECOMPILED CLI11_SANITIZERS
        CLI11_SINGLE_FILE CLI11_SINGLE_FILE_TESTS CLI11_WARNINGS_AS_ERRORS)
      set(CLI11_BUILD_DOCS OFF)
      set(CLI11_BUILD_EXAMPLES OFF)
      set(CLI11_BUILD_EXAMPLES_JSON OFF)
      set(CLI11_BUILD_TESTS OFF)
      set(CLI11_INSTALL OFF)
      set(CLI11_PRECOMPILED OFF)
      set(CLI11_SANITIZERS OFF)
      set(CLI11_SINGLE_FILE OFF)
      set(CLI11_SINGLE_FILE_TESTS OFF)
      set(CLI11_WARNINGS_AS_ERRORS OFF)
      add_subdirectory("${INFERX_THIRD_PARTY_DIR}/CLI11"
                       "${CMAKE_BINARY_DIR}/third_party/CLI11"
                       SYSTEM EXCLUDE_FROM_ALL)
      inferx_dependency_scope_pop(
        CLI11_BUILD_DOCS CLI11_BUILD_EXAMPLES CLI11_BUILD_EXAMPLES_JSON
        CLI11_BUILD_TESTS CLI11_INSTALL CLI11_PRECOMPILED CLI11_SANITIZERS
        CLI11_SINGLE_FILE CLI11_SINGLE_FILE_TESTS CLI11_WARNINGS_AS_ERRORS)
    else()
      find_package(CLI11 2.6.2 CONFIG REQUIRED)
    endif()
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
      # GoogleMock is part of the test dependency surface.
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
