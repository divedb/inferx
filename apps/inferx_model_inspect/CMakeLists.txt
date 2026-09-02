add_executable(inferx-model-inspect main.cc)
target_link_libraries(inferx-model-inspect PRIVATE
  inferx::artifacts inferx::model inferx::input
  absl::status absl::statusor)
inferx_apply_project_options(inferx-model-inspect)
