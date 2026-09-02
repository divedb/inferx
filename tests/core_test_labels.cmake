set_tests_properties(
  ${inferx_value_test_TESTS}
  ${inferx_status_test_TESTS}
  ${inferx_fsm_test_TESTS}
  ${inferx_config_test_TESTS}
  ${inferx_scheduler_test_TESTS}
  ${inferx_simulator_components_test_TESTS}
  PROPERTIES LABELS "core;core-unit")

set_tests_properties(${inferx_channel_test_TESTS}
  PROPERTIES LABELS "core;core-channel")
set_tests_properties(${inferx_simulator_integration_test_TESTS}
  PROPERTIES LABELS "core;core-integration")
set_tests_properties(${inferx_replay_correctness_test_TESTS}
  PROPERTIES LABELS "core;core-correctness")
set_tests_properties(${inferx_simulator_failure_test_TESTS}
  PROPERTIES LABELS "core;core-failure")
