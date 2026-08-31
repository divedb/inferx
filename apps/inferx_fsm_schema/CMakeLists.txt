# Build-only metadata emitter used by lifecycle documentation/tests.
add_executable(inferx_fsm_schema main.cc)
inferx_apply_project_options(inferx_fsm_schema)
target_link_libraries(inferx_fsm_schema PRIVATE
  inferx::engine absl::status absl::strings)
set_target_properties(inferx_fsm_schema PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
