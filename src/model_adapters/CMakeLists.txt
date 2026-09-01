add_library(inferx_artifact_tensor_adapter STATIC
  artifact_tensor_adapter.cc)
inferx_apply_project_options(inferx_artifact_tensor_adapter)
target_include_directories(inferx_artifact_tensor_adapter PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(inferx_artifact_tensor_adapter
  PUBLIC inferx::artifacts inferx::model inferx::tensor inferx::ops
  PRIVATE absl::status absl::statusor)
