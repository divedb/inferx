# CTest-time label assignment for GoogleTest-discovered tests.
#
# gtest_discover_tests drops every label after the first semicolon of a
# LABELS property, so multi-label categories (a general label plus a
# category) are applied here instead, after discovery has defined the
# *_TESTS lists. tests/CMakeLists.txt registers this file once via
# TEST_INCLUDE_FILES. tools/ci/check_test_labels.py and the CI `-L` filters
# select on these categories.

# Required core categories (tools/ci/check_test_labels.py).
set_tests_properties(
  ${inferx_value_test_TESTS}
  ${inferx_status_test_TESTS}
  ${inferx_fsm_test_TESTS}
  ${inferx_config_test_TESTS}
  ${inferx_scheduler_test_TESTS}
  ${inferx_log_test_TESTS}
  PROPERTIES LABELS "core;core-unit")

set_tests_properties(${inferx_channel_test_TESTS}
  PROPERTIES LABELS "core;core-channel")

if(DEFINED inferx_artifacts_test_TESTS)
  set_tests_properties(${inferx_artifacts_test_TESTS}
    PROPERTIES LABELS "unit;model" TIMEOUT 60)
endif()

if(DEFINED inferx_m5_semantics_test_TESTS)
  set_tests_properties(${inferx_m5_semantics_test_TESTS}
    PROPERTIES LABELS "m5;m5-unit;model;model-unit" TIMEOUT 120)
endif()
if(DEFINED inferx_m5_runtime_test_TESTS)
  set_tests_properties(${inferx_m5_runtime_test_TESTS}
    PROPERTIES LABELS "m5;m5-unit;runtime" TIMEOUT 120)
endif()
if(DEFINED inferx_execution_config_test_TESTS)
  set_tests_properties(${inferx_execution_config_test_TESTS}
    PROPERTIES LABELS "m5;m5-unit;config" TIMEOUT 120)
endif()

set_tests_properties(
  ${inferx_tensor_test_TESTS}
  ${inferx_runtime_test_TESTS}
  ${inferx_cuda_config_test_TESTS}
  PROPERTIES LABELS "platform;platform-unit")

if(DEFINED inferx_ops_test_TESTS)
  set_tests_properties(${inferx_ops_test_TESTS}
    PROPERTIES LABELS "operators;operators-unit" TIMEOUT 120)
endif()
if(DEFINED inferx_artifact_adapter_test_TESTS)
  set_tests_properties(${inferx_artifact_adapter_test_TESTS}
    PROPERTIES LABELS "operators;operators-unit" TIMEOUT 120)
endif()
if(DEFINED inferx_kernels_cuda_test_TESTS)
  set_tests_properties(${inferx_kernels_cuda_test_TESTS}
    PROPERTIES LABELS "operators;operators-gpu" TIMEOUT 120)
endif()
if(DEFINED inferx_kernels_ops_oracle_test_TESTS)
  set_tests_properties(${inferx_kernels_ops_oracle_test_TESTS}
    PROPERTIES LABELS "operators;operators-gpu" TIMEOUT 120)
endif()
