# InferX test tree (m0.md sections 7, 13). CTest labels: unit, compile,
# dependency, integration, failure, and the M2 category labels. Every test
# gets a bounded timeout and runs from any working directory through absolute
# paths.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ---------------------------------------------------------------------------
# unit: base/version test. The public header is the first include so
# self-containment is exercised in the test itself as well as by
# tools/check_headers.py.
# ---------------------------------------------------------------------------
add_executable(inferx_base_test
  unit/base/version_test.cc
  unit/base/version_second_tu.cc)
inferx_apply_project_options(inferx_base_test)
target_link_libraries(inferx_base_test PRIVATE inferx::base GTest::gtest_main)
target_compile_definitions(inferx_base_test PRIVATE
  INFERX_EXPECTED_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  INFERX_EXPECTED_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  INFERX_EXPECTED_VERSION_PATCH=${PROJECT_VERSION_PATCH})
include(GoogleTest)
gtest_discover_tests(inferx_base_test
  PROPERTIES LABELS "unit" TIMEOUT 60)

# ---------------------------------------------------------------------------
# m1-unit: M1 base contract tests (labels include the general `m1` label).
# ---------------------------------------------------------------------------
add_executable(inferx_m1_value_test
  unit/base/m1_id_test.cc
  unit/base/m1_value_test.cc)
inferx_apply_project_options(inferx_m1_value_test)
target_link_libraries(inferx_m1_value_test PRIVATE
  inferx::base GTest::gtest_main absl::strings absl::status)
gtest_discover_tests(inferx_m1_value_test
  PROPERTIES LABELS "m1-unit" TIMEOUT 60)

add_executable(inferx_m1_status_test unit/base/m1_status_test.cc)
inferx_apply_project_options(inferx_m1_status_test)
target_link_libraries(inferx_m1_status_test PRIVATE
  inferx::base GTest::gtest_main absl::strings absl::status)
gtest_discover_tests(inferx_m1_status_test
  PROPERTIES LABELS "m1-unit" TIMEOUT 60)

add_executable(inferx_m1_fsm_test unit/engine/m1_fsm_test.cc)
inferx_apply_project_options(inferx_m1_fsm_test)
target_link_libraries(inferx_m1_fsm_test PRIVATE
  inferx::engine GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_m1_fsm_test
  PROPERTIES LABELS "m1-unit" TIMEOUT 60)

add_executable(inferx_m1_config_test unit/config/m1_config_test.cc)
inferx_apply_project_options(inferx_m1_config_test)
target_link_libraries(inferx_m1_config_test PRIVATE
  inferx::config GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_m1_config_test
  PROPERTIES LABELS "m1-unit" TIMEOUT 60)

# m1-channel: bounded channel semantics (threaded; TSan lane selects these).
add_executable(inferx_m1_channel_test unit/base/m1_channel_test.cc)
inferx_apply_project_options(inferx_m1_channel_test)
target_link_libraries(inferx_m1_channel_test PRIVATE
  inferx::base GTest::gtest_main absl::strings)
gtest_discover_tests(inferx_m1_channel_test
  PROPERTIES LABELS "m1-channel" TIMEOUT 120)

# ---------------------------------------------------------------------------
# m2-unit: checked tensor values, views, ownership, and CPU reference copy.
# ---------------------------------------------------------------------------
add_executable(inferx_m2_tensor_test unit/tensor/m2_tensor_test.cc)
inferx_apply_project_options(inferx_m2_tensor_test)
target_link_libraries(inferx_m2_tensor_test PRIVATE
  inferx::tensor GTest::gtest_main absl::status absl::statusor)
gtest_discover_tests(inferx_m2_tensor_test
  PROPERTIES LABELS "m2-unit" TIMEOUT 120)

add_executable(inferx_m2_runtime_test
  unit/runtime/m2_runtime_test.cc
  fakes/fake_allocator.cc
  fakes/fake_fence.cc)
inferx_apply_project_options(inferx_m2_runtime_test)
target_include_directories(inferx_m2_runtime_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(inferx_m2_runtime_test PRIVATE
  inferx::runtime GTest::gtest_main absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_m2_runtime_test
  PROPERTIES LABELS "m2-unit" TIMEOUT 120)

add_executable(inferx_m2_config_test unit/config/m2_cuda_config_test.cc)
inferx_apply_project_options(inferx_m2_config_test)
target_link_libraries(inferx_m2_config_test PRIVATE
  inferx::config GTest::gtest_main absl::status absl::statusor)
gtest_discover_tests(inferx_m2_config_test
  PROPERTIES LABELS "m2-unit" TIMEOUT 120)

add_executable(inferx_m2_runtime_stress_test
  stress/runtime/m2_runtime_stress_test.cc
  fakes/fake_fence.cc)
inferx_apply_project_options(inferx_m2_runtime_stress_test)
target_include_directories(inferx_m2_runtime_stress_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(inferx_m2_runtime_stress_test PRIVATE
  inferx::runtime GTest::gtest_main absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_m2_runtime_stress_test
  PROPERTIES LABELS "m2-stress" TIMEOUT 180)

if(INFERX_ENABLE_CUDA AND INFERX_BUILD_GPU_TESTS)
  add_executable(inferx_m2_cuda_unit_test
    unit/cuda/m2_cuda_unit_test.cc
    fakes/fake_allocator.cc)
  inferx_apply_project_options(inferx_m2_cuda_unit_test)
  target_include_directories(inferx_m2_cuda_unit_test PRIVATE
    ${CMAKE_SOURCE_DIR})
  target_link_libraries(inferx_m2_cuda_unit_test PRIVATE
    inferx_cuda GTest::gtest_main CUDA::cudart
    absl::status absl::statusor absl::strings)
  target_compile_definitions(inferx_m2_cuda_unit_test PRIVATE
    INFERX_CUDA_DEBUG_POINTERS=1)
  gtest_discover_tests(inferx_m2_cuda_unit_test
    PROPERTIES LABELS "m2-unit" TIMEOUT 120)

  add_executable(inferx_cuda_integration_test
    integration/cuda/cuda_test_pipeline.cc
    integration/cuda/m2_cuda_integration_test.cc)
  inferx_apply_project_options(inferx_cuda_integration_test)
  target_include_directories(inferx_cuda_integration_test PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/kernels/cuda/test)
  target_link_libraries(inferx_cuda_integration_test PRIVATE
    inferx_cuda inferx_cuda_test_kernels GTest::gtest_main CUDA::cudart
    absl::status absl::statusor absl::strings)
  gtest_discover_tests(inferx_cuda_integration_test
    PROPERTIES LABELS "m2-integration" TIMEOUT 600)

  add_executable(inferx_cuda_failure_child
    failure/cuda/m2_cuda_failure_child.cc)
  inferx_apply_project_options(inferx_cuda_failure_child)
  target_link_libraries(inferx_cuda_failure_child PRIVATE
    inferx_cuda CUDA::cudart absl::status absl::statusor)
  target_include_directories(inferx_cuda_failure_child PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels/cuda/test)
  target_link_libraries(inferx_cuda_failure_child PRIVATE
    inferx_cuda_test_kernels)
  add_test(NAME m2_cuda_illegal_address_child
    COMMAND inferx_cuda_failure_child --illegal-address)
  set_tests_properties(m2_cuda_illegal_address_child PROPERTIES
    LABELS "m2-failure" TIMEOUT 60)
  add_test(NAME m2_cuda_assert_child
    COMMAND inferx_cuda_failure_child --assert)
  set_tests_properties(m2_cuda_assert_child PROPERTIES
    LABELS "m2-failure" TIMEOUT 60)

  add_executable(inferx_tensor_cuda_correctness
    correctness/tensor_cuda/m2_tensor_cuda_correctness.cc)
  inferx_apply_project_options(inferx_tensor_cuda_correctness)
  target_include_directories(inferx_tensor_cuda_correctness PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels/cuda/test)
  target_link_libraries(inferx_tensor_cuda_correctness PRIVATE
    inferx_cuda inferx_cuda_test_kernels GTest::gtest_main CUDA::cudart
    absl::status absl::statusor)
  gtest_discover_tests(inferx_tensor_cuda_correctness
    PROPERTIES LABELS "m2-correctness" TIMEOUT 900)
endif()

# ---------------------------------------------------------------------------
# compile: C++23 feature sentinel. Exercises library facilities, not macros.
# ---------------------------------------------------------------------------
add_executable(inferx_cxx23_smoke compile/cxx23_smoke.cc)
inferx_apply_project_options(inferx_cxx23_smoke)
add_test(NAME cxx23_sentinel COMMAND inferx_cxx23_smoke)
set_tests_properties(cxx23_sentinel PROPERTIES LABELS "compile" TIMEOUT 60)

# ---------------------------------------------------------------------------
# dependency: Abseil qualification (ADR 0002 pin; kept inside the test target
# so inferx::base does not expose Abseil before M1).
# ---------------------------------------------------------------------------
add_executable(inferx_absl_qualification_test dependency/absl_qualification_test.cc)
inferx_apply_project_options(inferx_absl_qualification_test)
target_link_libraries(inferx_absl_qualification_test PRIVATE
  GTest::gtest_main absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_absl_qualification_test
  PROPERTIES LABELS "dependency" TIMEOUT 60)

# ---------------------------------------------------------------------------
# integration: CLI behavior and install/consumer flows. The consumer projects
# under tests/consumer* are separate CMake projects driven by Python drivers
# so they cannot see the source tree or build tree by accident.
# ---------------------------------------------------------------------------
add_test(NAME cli
  COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/integration/cli_test.py
          --binary $<TARGET_FILE:inferx-info>
          --expected-version ${PROJECT_VERSION})
set_tests_properties(cli PROPERTIES LABELS "integration" TIMEOUT 120)

if(PROJECT_IS_TOP_LEVEL AND NOT (INFERX_ENABLE_ASAN OR INFERX_ENABLE_UBSAN OR INFERX_ENABLE_TSAN))
  # Only the top-level build may install; a subproject embedder uses its own
  # install rules. Sanitizer lanes skip these: an instrumented static library
  # cannot be linked by a non-instrumented consumer, and install/consumer
  # coverage is owned by the cpu-release lane (m0.md section 11.1).
  add_test(NAME install_consumer
    COMMAND Python3::Interpreter
            ${CMAKE_CURRENT_SOURCE_DIR}/integration/install_consumer.py
            --source-dir ${CMAKE_SOURCE_DIR}
            --build-dir ${CMAKE_BINARY_DIR}
            --consumer-dir ${CMAKE_CURRENT_SOURCE_DIR}/consumer
            --output-root ${CMAKE_SOURCE_DIR}/out/consumer
            --cxx-compiler ${CMAKE_CXX_COMPILER}
            --expected-version ${PROJECT_VERSION})
  set_tests_properties(install_consumer PROPERTIES LABELS "integration" TIMEOUT 600)

  add_test(NAME install_absl_consumer
    COMMAND Python3::Interpreter
            ${CMAKE_CURRENT_SOURCE_DIR}/integration/install_consumer.py
            --source-dir ${CMAKE_SOURCE_DIR}
            --build-dir ${CMAKE_BINARY_DIR}
            --consumer-dir ${CMAKE_CURRENT_SOURCE_DIR}/consumer_absl
            --output-root ${CMAKE_SOURCE_DIR}/out/consumer
            --cxx-compiler ${CMAKE_CXX_COMPILER}
            --expected-version ${PROJECT_VERSION})
  set_tests_properties(install_absl_consumer PROPERTIES LABELS "integration" TIMEOUT 900)
endif()

# ---------------------------------------------------------------------------
# failure: expected configure/compile failures (m0.md section 13.5). A test
# passes only when the command fails AND every expected message fragment is
# present, so an unrelated failure does not satisfy it.
# ---------------------------------------------------------------------------
function(inferx_add_failure_test name)
  set(multi EXPECT)
  cmake_parse_arguments(FXT "" "" "${multi}" ${ARGN})
  set(_INFERX_EXPECT_ARGS "")
  foreach(_fragment IN ITEMS ${FXT_EXPECT})
    list(APPEND _INFERX_EXPECT_ARGS "--expect" "${_fragment}")
  endforeach()
  add_test(NAME failure_${name}
    COMMAND Python3::Interpreter
            ${CMAKE_CURRENT_SOURCE_DIR}/failure/run_expected_failure.py
            --source-root ${CMAKE_SOURCE_DIR}
            --work-dir ${CMAKE_CURRENT_BINARY_DIR}/failure/${name}
            ${_INFERX_EXPECT_ARGS}
            ${FXT_UNPARSED_ARGUMENTS})
  set_tests_properties(failure_${name} PROPERTIES LABELS "failure" TIMEOUT 300)
endfunction()

inferx_add_failure_test(missing_core_submodule
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/missing_core_submodule.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "abseil-cpp" "bootstrap.py --profile core")

inferx_add_failure_test(unknown_dependency_provider
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/unknown_provider.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "INFERX_DEPENDENCY_PROVIDER" "submodule" "system")

inferx_add_failure_test(cmake_too_old
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/cmake_too_old.cmake
  EXPECT "or higher is required")

inferx_add_failure_test(compiler_too_old
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/compiler_too_old.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "GCC" "13")

inferx_add_failure_test(cxx23_library_missing
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/cxx23_library_missing.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  --build
  EXPECT "std::unreachable" "supported standard library")

inferx_add_failure_test(asan_and_tsan
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/asan_and_tsan.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "mutually exclusive")

inferx_add_failure_test(cuda_toolkit_missing
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/cuda_toolkit_missing.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "CUDA" "toolkit")

inferx_add_failure_test(cuda_architecture_unset
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/cuda_architecture_unset.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "CMAKE_CUDA_ARCHITECTURES" "89")

inferx_add_failure_test(analysis_tool_absent
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/analysis_tool_absent.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "clang-tidy" "18")

inferx_add_failure_test(consumer_missing_dependency
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/consumer_missing_dependency.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  --arg CONSUMER_DIR=${CMAKE_CURRENT_SOURCE_DIR}/consumer_absl
  EXPECT "absl")

inferx_add_failure_test(manifest_gitlink_drift
  --script ${CMAKE_CURRENT_SOURCE_DIR}/failure/manifest_drift.py
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "revision drift" "third_party/abseil-cpp" "gitlink")
# The drift fixture mutates the mirrored abseil-cpp revision in a temporary
# manifest copy; the real gitlink must win (see failure/manifest_drift.py).

add_test(NAME m2_cuda_header_boundary
  COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/check_cuda_boundaries.py
          --source-root ${CMAKE_SOURCE_DIR})
set_tests_properties(m2_cuda_header_boundary PROPERTIES
  LABELS "m2-unit" TIMEOUT 60)

# ---------------------------------------------------------------------------
# Aggregate convenience targets (m0.md section 9.4). They never configure
# other build trees; sanitizer and CUDA lanes stay separate invocations.
# ---------------------------------------------------------------------------
add_custom_target(check-fast
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target format-check
  COMMAND ${CMAKE_CTEST_COMMAND} -L "unit|compile" --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM
  USES_TERMINAL
  COMMENT "InferX check-fast: format-check + unit/compile tests")

add_custom_target(check-analysis
  COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/check_cuda_boundaries.py
          --source-root ${CMAKE_SOURCE_DIR}
  COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/check_headers.py
          --source-root ${CMAKE_SOURCE_DIR} --build-dir ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_SOURCE_DIR}/tools/run_clang_tidy.sh ${CMAKE_BINARY_DIR}
  COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/run_iwyu.py
          --build-dir ${CMAKE_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM
  USES_TERMINAL
  COMMENT "InferX check-analysis: header probes + clang-tidy + IWYU")

add_custom_target(check
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-fast
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-analysis
  COMMAND ${CMAKE_CTEST_COMMAND} -L "integration|failure|dependency"
          --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM
  USES_TERMINAL
  COMMENT "InferX check: all CPU quality gates in this build tree")
