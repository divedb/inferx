# InferX test tree (project quality policy). CTest labels: unit and the
# platform category labels. Every test gets a bounded timeout and runs from
# any working directory through absolute paths.

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

add_executable(inferx_log_test unit/base/log_test.cc)
inferx_apply_project_options(inferx_log_test)
target_link_libraries(inferx_log_test PRIVATE inferx::base GTest::gtest_main)
gtest_discover_tests(inferx_log_test
  PROPERTIES TIMEOUT 60)

if(INFERX_BUILD_CLI)
  add_executable(inferx_cli_test unit/cli/app_test.cc)
  inferx_apply_project_options(inferx_cli_test)
  target_link_libraries(inferx_cli_test PRIVATE inferx_cli GTest::gtest_main)
  gtest_discover_tests(inferx_cli_test
    PROPERTIES LABELS "unit" TIMEOUT 60)
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

# GoogleTest discovers individual cases after build. Assign both the general
# platform label and the required category at CTest time; passing a semicolon list
# through gtest_discover_tests loses the second label.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime_test_labels.cmake")

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
  COMMAND ${CMAKE_CTEST_COMMAND} -L "unit" --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM
  USES_TERMINAL
  COMMENT "InferX check-fast: format-check + unit tests")
add_dependencies(check-fast
  inferx_base_test
  inferx_artifacts_test)

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
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM
  USES_TERMINAL
  COMMENT "InferX check: all CPU quality gates in this build tree")
