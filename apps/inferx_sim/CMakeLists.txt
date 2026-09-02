# inferx-sim: deterministic simulator/config/replay developer CLI.
add_executable(inferx-sim main.cc)
inferx_apply_project_options(inferx-sim)
target_link_libraries(inferx-sim PRIVATE
  inferx::simulator
  absl::flags
  absl::flags_parse
  absl::log_initialize
  absl::status
  absl::strings)
set_target_properties(inferx-sim PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

install(TARGETS inferx-sim
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
