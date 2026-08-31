set_tests_properties(
  ${inferx_m1_value_test_TESTS}
  ${inferx_m1_status_test_TESTS}
  ${inferx_m1_fsm_test_TESTS}
  ${inferx_m1_config_test_TESTS}
  ${inferx_m1_scheduler_test_TESTS}
  ${inferx_m1_simulator_components_test_TESTS}
  PROPERTIES LABELS "m1;m1-unit")

set_tests_properties(${inferx_m1_channel_test_TESTS}
  PROPERTIES LABELS "m1;m1-channel")
set_tests_properties(${inferx_m1_simulator_integration_test_TESTS}
  PROPERTIES LABELS "m1;m1-integration")
set_tests_properties(${inferx_m1_replay_correctness_test_TESTS}
  PROPERTIES LABELS "m1;m1-correctness")
set_tests_properties(${inferx_m1_simulator_failure_test_TESTS}
  PROPERTIES LABELS "m1;m1-failure")
