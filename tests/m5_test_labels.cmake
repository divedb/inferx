if(DEFINED inferx_m5_semantics_test_TESTS)
  set_tests_properties(${inferx_m5_semantics_test_TESTS}
    PROPERTIES LABELS "m5;m5-unit;model;model-unit" TIMEOUT 120)
endif()
if(DEFINED inferx_m5_runtime_test_TESTS)
  set_tests_properties(${inferx_m5_runtime_test_TESTS}
    PROPERTIES LABELS "m5;m5-unit;runtime" TIMEOUT 120)
endif()