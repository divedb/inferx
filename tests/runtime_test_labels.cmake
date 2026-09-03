set_tests_properties(
  ${inferx_tensor_test_TESTS}
  ${inferx_runtime_test_TESTS}
  ${inferx_cuda_config_test_TESTS}
  PROPERTIES LABELS "platform;platform-unit")

if(DEFINED inferx_cuda_unit_test_TESTS)
  set_tests_properties(${inferx_cuda_unit_test_TESTS}
    PROPERTIES LABELS "platform;platform-unit")
endif()
if(DEFINED inferx_cuda_integration_test_TESTS)
  set_tests_properties(${inferx_cuda_integration_test_TESTS}
    PROPERTIES LABELS "platform;platform-integration")
endif()
if(DEFINED inferx_tensor_cuda_correctness_TESTS)
  set_tests_properties(${inferx_tensor_cuda_correctness_TESTS}
    PROPERTIES LABELS "platform;platform-correctness")
endif()
