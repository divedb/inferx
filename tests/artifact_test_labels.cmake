if(DEFINED inferx_artifacts_test_TESTS)
  set_tests_properties(${inferx_artifacts_test_TESTS}
    PROPERTIES LABELS "unit;model" TIMEOUT 60)
endif()
