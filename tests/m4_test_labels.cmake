if(DEFINED inferx_m4_ops_test_TESTS)
  set_tests_properties(${inferx_m4_ops_test_TESTS}
    PROPERTIES LABELS "m4;m4-unit" TIMEOUT 120)
endif()
if(DEFINED inferx_m4_artifact_adapter_test_TESTS)
  set_tests_properties(${inferx_m4_artifact_adapter_test_TESTS}
    PROPERTIES LABELS "m4;m4-unit" TIMEOUT 120)
endif()
if(DEFINED inferx_m4_cuda_ops_test_TESTS)
  set_tests_properties(${inferx_m4_cuda_ops_test_TESTS}
    PROPERTIES LABELS "m4;m4-gpu" TIMEOUT 120)
endif()
