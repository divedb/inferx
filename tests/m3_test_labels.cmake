if(DEFINED inferx_m3_artifacts_test_TESTS)
  set_tests_properties(${inferx_m3_artifacts_test_TESTS}
    PROPERTIES LABELS "unit;m3" TIMEOUT 60)
endif()
