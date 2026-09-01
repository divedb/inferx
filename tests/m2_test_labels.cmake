set_tests_properties(
  ${inferx_m2_tensor_test_TESTS}
  ${inferx_m2_runtime_test_TESTS}
  ${inferx_m2_config_test_TESTS}
  PROPERTIES LABELS "m2;m2-unit")

set_tests_properties(${inferx_m2_runtime_stress_test_TESTS}
  PROPERTIES LABELS "m2;m2-stress")

if(DEFINED inferx_m2_cuda_unit_test_TESTS)
  set_tests_properties(${inferx_m2_cuda_unit_test_TESTS}
    PROPERTIES LABELS "m2;m2-unit")
endif()
if(DEFINED inferx_cuda_integration_test_TESTS)
  set_tests_properties(${inferx_cuda_integration_test_TESTS}
    PROPERTIES LABELS "m2;m2-integration")
endif()
if(DEFINED inferx_tensor_cuda_correctness_TESTS)
  set_tests_properties(${inferx_tensor_cuda_correctness_TESTS}
    PROPERTIES LABELS "m2;m2-correctness")
endif()
