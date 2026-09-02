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
