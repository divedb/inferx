set_tests_properties(
  ${inferx_value_test_TESTS}
  ${inferx_status_test_TESTS}
  ${inferx_fsm_test_TESTS}
  ${inferx_config_test_TESTS}
  ${inferx_scheduler_test_TESTS}
  ${inferx_log_test_TESTS}
  PROPERTIES LABELS "core;core-unit")

set_tests_properties(${inferx_channel_test_TESTS}
  PROPERTIES LABELS "core;core-channel")
