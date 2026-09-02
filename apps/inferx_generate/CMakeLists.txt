add_executable(inferx-generate main.cc)
target_link_libraries(inferx-generate PRIVATE
  inferx::artifacts inferx::execution_runtime inferx::input
  absl::status absl::statusor)
inferx_apply_project_options(inferx-generate)
