if(DEFINED inferx_m3_artifacts_test_TESTS)
  set_tests_properties(${inferx_m3_artifacts_test_TESTS}
    PROPERTIES LABELS "unit;m3" TIMEOUT 60)
endif()

if(DEFINED inferx_m3_model_resolver_test_TESTS)
  set_tests_properties(${inferx_m3_model_resolver_test_TESTS}
    PROPERTIES LABELS "unit;m3" TIMEOUT 60)
endif()

if(DEFINED inferx_m3_model_package_integration_test_TESTS)
  set_tests_properties(${inferx_m3_model_package_integration_test_TESTS}
    PROPERTIES LABELS "m3;m3-integration" TIMEOUT 120)
endif()

if(DEFINED inferx_m3_safetensors_corpus_test_TESTS)
  set_tests_properties(${inferx_m3_safetensors_corpus_test_TESTS}
    PROPERTIES LABELS "m3;m3-correctness" TIMEOUT 120)
endif()

if(DEFINED inferx_m3_artifact_stress_test_TESTS)
  set_tests_properties(${inferx_m3_artifact_stress_test_TESTS}
    PROPERTIES LABELS "m3;m3-stress" TIMEOUT 300)
endif()
