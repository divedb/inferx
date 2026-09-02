# InferX test tree (project quality policy). CTest labels: unit, compile,
# dependency, integration, failure, and the platform category labels. Every test
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

if(INFERX_BUILD_CLI)
  add_executable(inferx_cli_test unit/cli/app_test.cc)
  inferx_apply_project_options(inferx_cli_test)
  target_link_libraries(inferx_cli_test PRIVATE inferx_cli GTest::gtest_main)
  gtest_discover_tests(inferx_cli_test
    PROPERTIES LABELS "unit;integration" TIMEOUT 60)
endif()

# model artifact/model contract: rooted path safety, official digest vectors,
# safetensors layout checks, mapping accounting, and complete Llama coverage.
add_executable(inferx_artifacts_test
  unit/artifacts/artifacts_test.cc)
inferx_apply_project_options(inferx_artifacts_test)
target_link_libraries(inferx_artifacts_test PRIVATE
  inferx::artifacts inferx::model GTest::gtest_main)
gtest_discover_tests(inferx_artifacts_test
  PROPERTIES TIMEOUT 60)

if(INFERX_ENABLE_TOKENIZATION)
  add_executable(inferx_tokenization_test
    unit/tokenization/tokenizer_smoke_test.cc)
  inferx_apply_project_options(inferx_tokenization_test)
  target_link_libraries(inferx_tokenization_test PRIVATE
    inferx::tokenization GTest::gtest_main)
  target_compile_definitions(inferx_tokenization_test PRIVATE
    TOKENIZER_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model/tokenizer_reference/qwen2.5/tokenizer.json"
    TOKENIZER_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model/tokenizer_reference/qwen2.5/tokenizer_config.json")
  gtest_discover_tests(inferx_tokenization_test
    PROPERTIES TIMEOUT 120)

  add_executable(inferx_tokenizer_differential_test
    correctness/tokenization/tokenizer_differential_test.cc)
  inferx_apply_project_options(inferx_tokenizer_differential_test)
  target_link_libraries(inferx_tokenizer_differential_test PRIVATE
    inferx::tokenization GTest::gtest_main simdjson::simdjson)
  target_compile_definitions(inferx_tokenizer_differential_test PRIVATE
    TOKENIZER_REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model/tokenizer_reference")
  gtest_discover_tests(inferx_tokenizer_differential_test
    PROPERTIES TIMEOUT 600 LABELS "model;model-correctness")

  add_executable(inferx_tokenizer_subprocess_test
    failure/tokenization/tokenizer_subprocess_test.cc)
  inferx_apply_project_options(inferx_tokenizer_subprocess_test)
  target_link_libraries(inferx_tokenizer_subprocess_test PRIVATE
    inferx::tokenization GTest::gtest_main)
  target_compile_definitions(inferx_tokenizer_subprocess_test PRIVATE
    TOKENIZER_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model/tokenizer_reference/qwen2.5/tokenizer.json")
  gtest_discover_tests(inferx_tokenizer_subprocess_test
    PROPERTIES TIMEOUT 300 LABELS "model;model-failure")

  add_executable(inferx_tokenizer_pool_stress_test
    stress/tokenization/tokenizer_pool_stress_test.cc)
  inferx_apply_project_options(inferx_tokenizer_pool_stress_test)
  target_link_libraries(inferx_tokenizer_pool_stress_test PRIVATE
    inferx::tokenization GTest::gtest_main)
  target_compile_definitions(inferx_tokenizer_pool_stress_test PRIVATE
    TOKENIZER_JSON_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model/tokenizer_reference/qwen2.5/tokenizer.json")
  gtest_discover_tests(inferx_tokenizer_pool_stress_test
    PROPERTIES TIMEOUT 300 LABELS "model;model-stress")
endif()
# GoogleTest discovers individual cases after build. Assign the labels at
# CTest time so the semicolon-separated label list remains one property value.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/artifact_test_labels.cmake")

if(INFERX_BUILD_OPERATOR_REFERENCE_TESTS)
  add_executable(inferx_ops_test unit/ops/ops_test.cc)
  inferx_apply_project_options(inferx_ops_test)
  target_link_libraries(inferx_ops_test PRIVATE
    inferx::ops inferx::tensor GTest::gtest_main absl::status absl::statusor)
  gtest_discover_tests(inferx_ops_test PROPERTIES TIMEOUT 120)

  add_executable(inferx_artifact_adapter_test
    unit/ops/artifact_adapter_test.cc)
  inferx_apply_project_options(inferx_artifact_adapter_test)
  target_include_directories(inferx_artifact_adapter_test PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(inferx_artifact_adapter_test PRIVATE
    inferx_artifact_tensor_adapter GTest::gtest_main absl::status absl::statusor)
  gtest_discover_tests(inferx_artifact_adapter_test PROPERTIES TIMEOUT 120)
endif()

if(INFERX_BUILD_OPERATOR_REFERENCE_TESTS)
  add_executable(inferx_kernels_dispatch_test unit/kernels/kernel_dispatch_test.cc)
  inferx_apply_project_options(inferx_kernels_dispatch_test)
  target_compile_definitions(inferx_kernels_dispatch_test PRIVATE
    INFERX_KERNELS_SOURCE_DIR="${CMAKE_SOURCE_DIR}/kernels")
  target_link_libraries(inferx_kernels_dispatch_test PRIVATE
    inferx_kernels GTest::gtest_main)
  gtest_discover_tests(inferx_kernels_dispatch_test PROPERTIES TIMEOUT 120)
endif()

if(INFERX_ENABLE_CUDA AND INFERX_BUILD_OPERATOR_GPU_TESTS)
  add_executable(inferx_kernels_cuda_test unit/cuda/kernels_cuda_test.cc)
  inferx_apply_project_options(inferx_kernels_cuda_test)
  target_include_directories(inferx_kernels_cuda_test PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels/cuda/dispatch ${CMAKE_SOURCE_DIR}/kernels/cuda/include)
  target_link_libraries(inferx_kernels_cuda_test PRIVATE
    inferx_kernels inferx_kernels_cuda inferx::ops inferx::tensor
    GTest::gtest_main CUDA::cudart absl::status absl::statusor absl::strings
    absl::synchronization)
  gtest_discover_tests(inferx_kernels_cuda_test PROPERTIES TIMEOUT 120)
endif()

if(INFERX_ENABLE_CUDA AND INFERX_BUILD_OPERATOR_GPU_TESTS)
  add_executable(inferx_kernels_ops_oracle_test unit/cuda/kernels_ops_oracle_test.cc)
  inferx_apply_project_options(inferx_kernels_ops_oracle_test)
  target_include_directories(inferx_kernels_ops_oracle_test PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels/cuda/dispatch ${CMAKE_SOURCE_DIR}/kernels/cuda/include/inferx/kernels/cuda
    ${CMAKE_SOURCE_DIR}/kernels/cuda/include
    ${CMAKE_SOURCE_DIR}/kernels/cuda/include)
  target_link_libraries(inferx_kernels_ops_oracle_test PRIVATE
    inferx_kernels inferx_kernels_cuda inferx::ops inferx::tensor
    GTest::gtest_main CUDA::cudart absl::status absl::statusor absl::strings
    absl::synchronization)
  gtest_discover_tests(inferx_kernels_ops_oracle_test PROPERTIES TIMEOUT 120)

endif()

if(INFERX_BUILD_OPERATOR_REFERENCE_TESTS OR
   (INFERX_ENABLE_CUDA AND INFERX_BUILD_OPERATOR_GPU_TESTS))
  set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/operator_test_labels.cmake")
endif()

# ---------------------------------------------------------------------------
# core-unit: core base contract tests (labels include the general `core` label).
# ---------------------------------------------------------------------------
add_executable(inferx_value_test
  unit/base/id_test.cc
  unit/base/value_test.cc)
inferx_apply_project_options(inferx_value_test)
target_link_libraries(inferx_value_test PRIVATE
  inferx::base GTest::gtest_main absl::strings absl::status)
gtest_discover_tests(inferx_value_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_status_test unit/base/status_test.cc)
inferx_apply_project_options(inferx_status_test)
target_link_libraries(inferx_status_test PRIVATE
  inferx::base GTest::gtest_main absl::strings absl::status)
gtest_discover_tests(inferx_status_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_fsm_test unit/engine/fsm_test.cc)
inferx_apply_project_options(inferx_fsm_test)
target_link_libraries(inferx_fsm_test PRIVATE
  inferx::engine GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_fsm_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_config_test unit/config/config_test.cc)
inferx_apply_project_options(inferx_config_test)
target_link_libraries(inferx_config_test PRIVATE
  inferx::config GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_config_test
  PROPERTIES TIMEOUT 60)

# core-channel: bounded channel semantics (threaded; TSan lane selects these).
add_executable(inferx_channel_test unit/base/channel_test.cc)
inferx_apply_project_options(inferx_channel_test)
target_link_libraries(inferx_channel_test PRIVATE
  inferx::base GTest::gtest_main absl::strings)
gtest_discover_tests(inferx_channel_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_scheduler_test unit/scheduler/scheduler_test.cc)
inferx_apply_project_options(inferx_scheduler_test)
target_link_libraries(inferx_scheduler_test PRIVATE
  inferx::scheduler GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_scheduler_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_simulator_components_test
  unit/simulator/simulator_components_test.cc)
inferx_apply_project_options(inferx_simulator_components_test)
target_link_libraries(inferx_simulator_components_test PRIVATE
  inferx::simulator GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_simulator_components_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_simulator_integration_test
  integration/simulator/simulator_integration_test.cc)
inferx_apply_project_options(inferx_simulator_integration_test)
target_link_libraries(inferx_simulator_integration_test PRIVATE
  inferx::simulator GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_simulator_integration_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_replay_correctness_test
  correctness/replay/replay_correctness_test.cc)
inferx_apply_project_options(inferx_replay_correctness_test)
target_link_libraries(inferx_replay_correctness_test PRIVATE
  inferx::simulator GTest::gtest_main absl::status absl::strings)
target_compile_definitions(inferx_replay_correctness_test PRIVATE
  INFERX_REPLAY_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/correctness/replay/data")
gtest_discover_tests(inferx_replay_correctness_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_simulator_failure_test
  failure/simulator/simulator_failure_test.cc)
inferx_apply_project_options(inferx_simulator_failure_test)
target_link_libraries(inferx_simulator_failure_test PRIVATE
  inferx::simulator GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_simulator_failure_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_simulator_stress
  stress/simulator/simulator_stress.cc)
inferx_apply_project_options(inferx_simulator_stress)
target_link_libraries(inferx_simulator_stress PRIVATE
  inferx::simulator absl::status absl::strings)
set_target_properties(inferx_simulator_stress PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME simulator_stress_50000
  COMMAND $<TARGET_FILE:inferx_simulator_stress>
          --seed=5562318636401906515
          --operations=50000
          --failure_output=${CMAKE_SOURCE_DIR}/out/failures/simulator)
set_tests_properties(simulator_stress_50000 PROPERTIES
  LABELS "core;core-stress" TIMEOUT 300)

if(TARGET inferx)
  add_test(NAME simulator_cli
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/integration/simulator_cli_test.py
            --binary $<TARGET_FILE:inferx>
            --config ${CMAKE_CURRENT_SOURCE_DIR}/integration/simulator/data/basic_config.json
            --workload ${CMAKE_CURRENT_SOURCE_DIR}/integration/simulator/data/basic_workload.json)
  set_tests_properties(simulator_cli PROPERTIES
    LABELS "core;core-integration" TIMEOUT 120)
endif()

if(TARGET inferx)
  add_test(NAME fsm_docs_current
    COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/docs/render_request_fsm.py
            --binary $<TARGET_FILE:inferx>
            --output ${CMAKE_SOURCE_DIR}/docs/request-lifecycle.md
            --check)
  set_tests_properties(fsm_docs_current PROPERTIES
    LABELS "core;core-correctness" TIMEOUT 60)
endif()

# GoogleTest discovers individual cases after build. This CTest-time include
# runs after those discovery includes and assigns both the general core label
# and the required subsystem category without lossy list escaping.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/core_test_labels.cmake")

# ---------------------------------------------------------------------------
# platform-unit: checked tensor values, views, ownership, and CPU reference copy.
# ---------------------------------------------------------------------------
add_executable(inferx_tensor_test unit/tensor/tensor_test.cc)
inferx_apply_project_options(inferx_tensor_test)
target_link_libraries(inferx_tensor_test PRIVATE
  inferx::tensor GTest::gtest_main absl::status absl::statusor)
gtest_discover_tests(inferx_tensor_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_runtime_test
  unit/runtime/runtime_test.cc
  fakes/fake_allocator.cc
  fakes/fake_fence.cc)
inferx_apply_project_options(inferx_runtime_test)
target_include_directories(inferx_runtime_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(inferx_runtime_test PRIVATE
  inferx::runtime GTest::gtest_main absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_runtime_test
  PROPERTIES TIMEOUT 120)

# M5: semantic layer, runtime contracts, and the CPU reference end-to-end.
add_executable(inferx_m5_semantics_test unit/model/m5_semantics_test.cc)
inferx_apply_project_options(inferx_m5_semantics_test)
target_link_libraries(inferx_m5_semantics_test PRIVATE
  inferx::model inferx::ops inferx::tensor GTest::gtest_main
  absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_m5_semantics_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m5_runtime_test unit/runtime/m5_runtime_test.cc)
inferx_apply_project_options(inferx_m5_runtime_test)
target_link_libraries(inferx_m5_runtime_test PRIVATE
  inferx::execution_runtime GTest::gtest_main
  absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_m5_runtime_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m5_cpu_backend_test integration/m5/m5_cpu_backend_test.cc)
inferx_apply_project_options(inferx_m5_cpu_backend_test)
target_compile_definitions(inferx_m5_cpu_backend_test PRIVATE
  M5_MODEL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model")
target_link_libraries(inferx_m5_cpu_backend_test PRIVATE
  inferx::execution_runtime inferx::input GTest::gtest_main
  absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_m5_cpu_backend_test
  PROPERTIES TIMEOUT 300)

set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/m5_test_labels.cmake")
if(DEFINED inferx_execution_config_test_TESTS)
  set_tests_properties(${inferx_execution_config_test_TESTS}
    PROPERTIES LABELS "m5;m5-unit;config" TIMEOUT 120)
endif()

add_executable(inferx_cuda_config_test unit/config/cuda_config_test.cc)

add_executable(inferx_execution_config_test unit/config/execution_config_test.cc)
inferx_apply_project_options(inferx_execution_config_test)
target_link_libraries(inferx_execution_config_test PRIVATE
  inferx::config GTest::gtest_main absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_execution_config_test
  PROPERTIES TIMEOUT 120)
inferx_apply_project_options(inferx_cuda_config_test)
target_link_libraries(inferx_cuda_config_test PRIVATE
  inferx::config GTest::gtest_main absl::status absl::statusor)
gtest_discover_tests(inferx_cuda_config_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_runtime_stress_test
  stress/runtime/runtime_stress_test.cc
  fakes/fake_fence.cc)
inferx_apply_project_options(inferx_runtime_stress_test)
target_include_directories(inferx_runtime_stress_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(inferx_runtime_stress_test PRIVATE
  inferx::runtime GTest::gtest_main absl::status absl::statusor absl::strings)
if(INFERX_ENABLE_TSAN)
  # TSan provides strong global new/delete interceptors. The same warmed pool
  # cycles still run in this lane, while allocation-count evidence is owned by
  # the ordinary and ASan/UBSan lanes.
  target_compile_definitions(inferx_runtime_stress_test PRIVATE
    INFERX_TSAN_BUILD=1)
endif()
gtest_discover_tests(inferx_runtime_stress_test
  PROPERTIES TIMEOUT 180)

# GoogleTest discovers individual cases after build. Assign both the general
# platform label and the required category at CTest time; passing a semicolon list
# through gtest_discover_tests loses the second label.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime_test_labels.cmake")

# ---------------------------------------------------------------------------
# compile: C++23 feature sentinel. Exercises library facilities, not macros.
# ---------------------------------------------------------------------------
add_executable(inferx_cxx23_smoke compile/cxx23_smoke.cc)
inferx_apply_project_options(inferx_cxx23_smoke)
add_test(NAME cxx23_sentinel COMMAND inferx_cxx23_smoke)
set_tests_properties(cxx23_sentinel PROPERTIES LABELS "compile" TIMEOUT 60)

# ---------------------------------------------------------------------------
# dependency: Abseil qualification (ADR 0002 pin; kept inside the test target
# so inferx::base does not expose Abseil before core).
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
if(INFERX_BUILD_CLI)
  add_test(NAME cli
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/integration/cli_test.py
            --binary $<TARGET_FILE:inferx>
            --expected-version ${PROJECT_VERSION})
  set_tests_properties(cli PROPERTIES LABELS "integration" TIMEOUT 120)
  set(_INFERX_MODEL_INSPECT_TEST_OPTIONS "")
  if(NOT INFERX_ENABLE_HF_HUB)
    list(APPEND _INFERX_MODEL_INSPECT_TEST_OPTIONS --expect-downloads-disabled)
  endif()
  add_test(NAME model_inspect_cli
    COMMAND Python3::Interpreter
            ${CMAKE_CURRENT_SOURCE_DIR}/integration/model_inspect_cli_test.py
            --binary $<TARGET_FILE:inferx>
            --fixture ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model/tiny_llama
            ${_INFERX_MODEL_INSPECT_TEST_OPTIONS})
  set_tests_properties(model_inspect_cli PROPERTIES LABELS "integration;model" TIMEOUT 120)

  if(INFERX_BUILD_VLLM_DIFFERENTIAL_TESTS)
    add_test(NAME real_model_vllm_differential
      COMMAND Python3::Interpreter
              ${CMAKE_CURRENT_SOURCE_DIR}/integration/real_model_vllm_test.py
              --inferx-binary $<TARGET_FILE:inferx>
              --download-dir ${CMAKE_BINARY_DIR}/real-model-cache)
    set_tests_properties(real_model_vllm_differential PROPERTIES
      LABELS "integration;model;external"
      RUN_SERIAL TRUE
      TIMEOUT 900)
  endif()
endif()

if(PROJECT_IS_TOP_LEVEL AND NOT (INFERX_ENABLE_ASAN OR INFERX_ENABLE_UBSAN OR INFERX_ENABLE_TSAN))
  # Only the top-level build may install; a subproject embedder uses its own
  # install rules. Sanitizer lanes skip these: an instrumented static library
  # cannot be linked by a non-instrumented consumer, and install/consumer
  # coverage is owned by the cpu-release lane (project quality policy).
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
# failure: expected configure/compile failures (project quality policy). A test
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

# The tokenizer_unqualified fixture existed to prove the rejected divedb
# candidate could not be enabled accidentally (ADR 0024, pre-acceptance).
# The gate is closed: the owned qualified backend configures and builds, and
# the model tokenization suites cover it; the obsolete expected-failure is gone.

inferx_add_failure_test(manifest_gitlink_drift
  --script ${CMAKE_CURRENT_SOURCE_DIR}/failure/manifest_drift.py
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "revision drift" "third_party/abseil-cpp" "gitlink")
# The drift fixture mutates the mirrored abseil-cpp revision in a temporary
# manifest copy; the real gitlink must win (see failure/manifest_drift.py).

add_test(NAME cuda_header_boundary
  COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/check_cuda_boundaries.py
          --source-root ${CMAKE_SOURCE_DIR})
set_tests_properties(cuda_header_boundary PROPERTIES
  LABELS "platform;platform-unit" TIMEOUT 60)

# ---------------------------------------------------------------------------
# Aggregate convenience targets (project quality policy). They never configure
# other build trees; sanitizer and CUDA lanes stay separate invocations.
# ---------------------------------------------------------------------------
add_custom_target(check-fast
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target format-check
  COMMAND ${CMAKE_CTEST_COMMAND} -L "unit|compile" --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM
  USES_TERMINAL
  COMMENT "InferX check-fast: format-check + unit/compile tests")
add_dependencies(check-fast
  inferx_base_test
  inferx_artifacts_test
  inferx_cxx23_smoke)

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
