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

# M3 artifact/model contract: rooted path safety, official digest vectors,
# safetensors layout checks, mapping accounting, and complete Llama coverage.
add_executable(inferx_m3_artifacts_test
  unit/artifacts/m3_artifacts_test.cc)
inferx_apply_project_options(inferx_m3_artifacts_test)
target_link_libraries(inferx_m3_artifacts_test PRIVATE
  inferx::artifacts inferx::model GTest::gtest_main)
gtest_discover_tests(inferx_m3_artifacts_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_m3_model_resolver_test
  unit/artifacts/m3_model_resolver_test.cc)
inferx_apply_project_options(inferx_m3_model_resolver_test)
target_include_directories(inferx_m3_model_resolver_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(inferx_m3_model_resolver_test PRIVATE
  inferx::artifacts GTest::gtest_main absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_m3_model_resolver_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_m3_model_package_integration_test
  integration/model_package/m3_model_package_integration_test.cc)
inferx_apply_project_options(inferx_m3_model_package_integration_test)
target_compile_definitions(inferx_m3_model_package_integration_test PRIVATE
  INFERX_M3_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/m3")
target_link_libraries(inferx_m3_model_package_integration_test PRIVATE
  inferx::artifacts inferx::model GTest::gtest_main)
gtest_discover_tests(inferx_m3_model_package_integration_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m3_safetensors_corpus_test
  correctness/safetensors/m3_safetensors_corpus_test.cc)
inferx_apply_project_options(inferx_m3_safetensors_corpus_test)
target_compile_definitions(inferx_m3_safetensors_corpus_test PRIVATE
  INFERX_M3_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/m3")
target_link_libraries(inferx_m3_safetensors_corpus_test PRIVATE
  inferx::artifacts GTest::gtest_main)
gtest_discover_tests(inferx_m3_safetensors_corpus_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m3_artifact_stress_test
  stress/artifacts/m3_artifact_stress_test.cc)
inferx_apply_project_options(inferx_m3_artifact_stress_test)
target_compile_definitions(inferx_m3_artifact_stress_test PRIVATE
  INFERX_M3_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/m3")
target_link_libraries(inferx_m3_artifact_stress_test PRIVATE
  inferx::artifacts inferx::model GTest::gtest_main)
gtest_discover_tests(inferx_m3_artifact_stress_test
  PROPERTIES TIMEOUT 300)

add_custom_target(m3-fixture-check
  COMMAND Python3::Interpreter
          ${CMAKE_SOURCE_DIR}/tools/fixtures/generate_m3_artifact_fixtures.py
          --check
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  VERBATIM
  COMMENT "InferX M3: verify committed artifact fixtures")
# GoogleTest discovers individual cases after build. Assign the labels at
# CTest time so the semicolon-separated label list remains one property value.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/m3_test_labels.cmake")

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
  PROPERTIES TIMEOUT 60)

add_executable(inferx_m1_status_test unit/base/m1_status_test.cc)
inferx_apply_project_options(inferx_m1_status_test)
target_link_libraries(inferx_m1_status_test PRIVATE
  inferx::base GTest::gtest_main absl::strings absl::status)
gtest_discover_tests(inferx_m1_status_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_m1_fsm_test unit/engine/m1_fsm_test.cc)
inferx_apply_project_options(inferx_m1_fsm_test)
target_link_libraries(inferx_m1_fsm_test PRIVATE
  inferx::engine GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_m1_fsm_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_m1_config_test unit/config/m1_config_test.cc)
inferx_apply_project_options(inferx_m1_config_test)
target_link_libraries(inferx_m1_config_test PRIVATE
  inferx::config GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_m1_config_test
  PROPERTIES TIMEOUT 60)

# m1-channel: bounded channel semantics (threaded; TSan lane selects these).
add_executable(inferx_m1_channel_test unit/base/m1_channel_test.cc)
inferx_apply_project_options(inferx_m1_channel_test)
target_link_libraries(inferx_m1_channel_test PRIVATE
  inferx::base GTest::gtest_main absl::strings)
gtest_discover_tests(inferx_m1_channel_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m1_scheduler_test unit/scheduler/m1_scheduler_test.cc)
inferx_apply_project_options(inferx_m1_scheduler_test)
target_link_libraries(inferx_m1_scheduler_test PRIVATE
  inferx::scheduler GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_m1_scheduler_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_m1_simulator_components_test
  unit/simulator/m1_simulator_components_test.cc)
inferx_apply_project_options(inferx_m1_simulator_components_test)
target_link_libraries(inferx_m1_simulator_components_test PRIVATE
  inferx::simulator GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_m1_simulator_components_test
  PROPERTIES TIMEOUT 60)

add_executable(inferx_m1_simulator_integration_test
  integration/simulator/m1_simulator_integration_test.cc)
inferx_apply_project_options(inferx_m1_simulator_integration_test)
target_link_libraries(inferx_m1_simulator_integration_test PRIVATE
  inferx::simulator GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_m1_simulator_integration_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m1_replay_correctness_test
  correctness/replay/m1_replay_correctness_test.cc)
inferx_apply_project_options(inferx_m1_replay_correctness_test)
target_link_libraries(inferx_m1_replay_correctness_test PRIVATE
  inferx::simulator GTest::gtest_main absl::status absl::strings)
target_compile_definitions(inferx_m1_replay_correctness_test PRIVATE
  INFERX_M1_REPLAY_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/correctness/replay/data")
gtest_discover_tests(inferx_m1_replay_correctness_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m1_simulator_failure_test
  failure/simulator/m1_simulator_failure_test.cc)
inferx_apply_project_options(inferx_m1_simulator_failure_test)
target_link_libraries(inferx_m1_simulator_failure_test PRIVATE
  inferx::simulator GTest::gtest_main absl::status absl::strings)
gtest_discover_tests(inferx_m1_simulator_failure_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_simulator_stress
  stress/simulator/m1_simulator_stress.cc)
inferx_apply_project_options(inferx_simulator_stress)
target_link_libraries(inferx_simulator_stress PRIVATE
  inferx::simulator absl::status absl::strings)
set_target_properties(inferx_simulator_stress PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME m1_simulator_stress_50000
  COMMAND $<TARGET_FILE:inferx_simulator_stress>
          --seed=0x4d31535452455353
          --operations=50000
          --failure_output=${CMAKE_SOURCE_DIR}/out/failures/m1)
set_tests_properties(m1_simulator_stress_50000 PROPERTIES
  LABELS "m1;m1-stress" TIMEOUT 300)

if(TARGET inferx-sim)
  add_test(NAME m1_sim_cli
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/integration/m1_sim_cli_test.py
            --binary $<TARGET_FILE:inferx-sim>
            --config ${CMAKE_CURRENT_SOURCE_DIR}/integration/simulator/data/basic_config.json
            --workload ${CMAKE_CURRENT_SOURCE_DIR}/integration/simulator/data/basic_workload.json)
  set_tests_properties(m1_sim_cli PROPERTIES
    LABELS "m1;m1-integration" TIMEOUT 120)
endif()

if(TARGET inferx_fsm_schema)
  add_test(NAME m1_fsm_docs_current
    COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/docs/render_request_fsm.py
            --binary $<TARGET_FILE:inferx_fsm_schema>
            --output ${CMAKE_SOURCE_DIR}/docs/request-lifecycle.md
            --check)
  set_tests_properties(m1_fsm_docs_current PROPERTIES
    LABELS "m1;m1-correctness" TIMEOUT 60)
endif()

# GoogleTest discovers individual cases after build. This CTest-time include
# runs after those discovery includes and assigns both the general M1 label
# and the required roadmap category without lossy list escaping.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/m1_test_labels.cmake")

# ---------------------------------------------------------------------------
# m2-unit: checked tensor values, views, ownership, and CPU reference copy.
# ---------------------------------------------------------------------------
add_executable(inferx_m2_tensor_test unit/tensor/m2_tensor_test.cc)
inferx_apply_project_options(inferx_m2_tensor_test)
target_link_libraries(inferx_m2_tensor_test PRIVATE
  inferx::tensor GTest::gtest_main absl::status absl::statusor)
gtest_discover_tests(inferx_m2_tensor_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m2_runtime_test
  unit/runtime/m2_runtime_test.cc
  fakes/fake_allocator.cc
  fakes/fake_fence.cc)
inferx_apply_project_options(inferx_m2_runtime_test)
target_include_directories(inferx_m2_runtime_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(inferx_m2_runtime_test PRIVATE
  inferx::runtime GTest::gtest_main absl::status absl::statusor absl::strings)
gtest_discover_tests(inferx_m2_runtime_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m2_config_test unit/config/m2_cuda_config_test.cc)
inferx_apply_project_options(inferx_m2_config_test)
target_link_libraries(inferx_m2_config_test PRIVATE
  inferx::config GTest::gtest_main absl::status absl::statusor)
gtest_discover_tests(inferx_m2_config_test
  PROPERTIES TIMEOUT 120)

add_executable(inferx_m2_runtime_stress_test
  stress/runtime/m2_runtime_stress_test.cc
  fakes/fake_fence.cc)
inferx_apply_project_options(inferx_m2_runtime_stress_test)
target_include_directories(inferx_m2_runtime_stress_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(inferx_m2_runtime_stress_test PRIVATE
  inferx::runtime GTest::gtest_main absl::status absl::statusor absl::strings)
if(INFERX_ENABLE_TSAN)
  # TSan provides strong global new/delete interceptors. The same warmed pool
  # cycles still run in this lane, while allocation-count evidence is owned by
  # the ordinary and ASan/UBSan lanes.
  target_compile_definitions(inferx_m2_runtime_stress_test PRIVATE
    INFERX_M2_TSAN_BUILD=1)
endif()
gtest_discover_tests(inferx_m2_runtime_stress_test
  PROPERTIES TIMEOUT 180)

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
    PROPERTIES TIMEOUT 120)

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
    PROPERTIES TIMEOUT 600)

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
    LABELS "m2;m2-failure" TIMEOUT 60)
  add_test(NAME m2_cuda_assert_child
    COMMAND inferx_cuda_failure_child --assert)
  set_tests_properties(m2_cuda_assert_child PROPERTIES
    LABELS "m2;m2-failure" TIMEOUT 60)
  add_test(NAME m2_cuda_timeout_classification_child
    COMMAND inferx_cuda_failure_child --classify-timeout)
  set_tests_properties(m2_cuda_timeout_classification_child PROPERTIES
    LABELS "m2;m2-failure" TIMEOUT 60)
  add_test(NAME m2_cuda_device_lost_classification_child
    COMMAND inferx_cuda_failure_child --classify-device-lost)
  set_tests_properties(m2_cuda_device_lost_classification_child PROPERTIES
    LABELS "m2;m2-failure" TIMEOUT 60)

  add_executable(inferx_tensor_cuda_correctness
    correctness/tensor_cuda/m2_tensor_cuda_correctness.cc)
  inferx_apply_project_options(inferx_tensor_cuda_correctness)
  target_include_directories(inferx_tensor_cuda_correctness PRIVATE
    ${CMAKE_SOURCE_DIR}/kernels/cuda/test)
  target_link_libraries(inferx_tensor_cuda_correctness PRIVATE
    inferx_cuda inferx_cuda_test_kernels GTest::gtest_main CUDA::cudart
    absl::status absl::statusor)
  gtest_discover_tests(inferx_tensor_cuda_correctness
    PROPERTIES TIMEOUT 900)
endif()

# GoogleTest discovers individual cases after build. Assign both the general
# M2 label and the required category at CTest time; passing a semicolon list
# through gtest_discover_tests loses the second label.
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/m2_test_labels.cmake")

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
if(INFERX_BUILD_TOOLS)
  add_test(NAME cli
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/integration/cli_test.py
            --binary $<TARGET_FILE:inferx-info>
            --expected-version ${PROJECT_VERSION})
  set_tests_properties(cli PROPERTIES LABELS "integration" TIMEOUT 120)
  set(_INFERX_MODEL_INSPECT_TEST_OPTIONS "")
  if(NOT INFERX_ENABLE_HF_HUB)
    list(APPEND _INFERX_MODEL_INSPECT_TEST_OPTIONS --expect-downloads-disabled)
  endif()
  add_test(NAME model_inspect_cli
    COMMAND Python3::Interpreter
            ${CMAKE_CURRENT_SOURCE_DIR}/integration/model_inspect_cli_test.py
            --binary $<TARGET_FILE:inferx-model-inspect>
            --fixture ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/m3/tiny_llama
            ${_INFERX_MODEL_INSPECT_TEST_OPTIONS})
  set_tests_properties(model_inspect_cli PROPERTIES LABELS "integration;m3" TIMEOUT 120)
endif()

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

inferx_add_failure_test(tokenizer_unqualified
  --fixture ${CMAKE_CURRENT_SOURCE_DIR}/failure/tokenizer_unqualified.cmake
  --arg INFERX_ROOT=${CMAKE_SOURCE_DIR}
  EXPECT "INFERX_ENABLE_TOKENIZATION" "was removed" "streaming decode state")

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
  LABELS "m2;m2-unit" TIMEOUT 60)

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
add_dependencies(check-fast
  inferx_base_test
  inferx_cxx23_smoke
  inferx_m1_config_test
  inferx_m1_fsm_test
  inferx_m1_scheduler_test
  inferx_m1_simulator_components_test
  inferx_m1_status_test
  inferx_m1_value_test
  inferx_m2_config_test
  inferx_m2_runtime_test
  inferx_m2_tensor_test
  inferx_m3_artifacts_test
  inferx_m3_model_resolver_test)
if(INFERX_ENABLE_CUDA)
  add_dependencies(check-fast inferx_m2_cuda_unit_test)
endif()

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
