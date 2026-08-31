add_library(inferx_artifacts STATIC
  artifact_file.cc
  artifact_limits.cc
  artifact_manifest.cc
  artifact_tensor.cc
  digest.cc
  mapped_region.cc
  model_fingerprint.cc
  model_locator_linux.cc
  safe_relative_path.cc
  safetensors_index.cc
  safetensors_reader.cc)
add_library(inferx::artifacts ALIAS inferx_artifacts)

set_target_properties(inferx_artifacts PROPERTIES
  EXPORT_NAME artifacts
  POSITION_INDEPENDENT_CODE ON)
target_compile_features(inferx_artifacts PUBLIC cxx_std_23)
target_include_directories(inferx_artifacts
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(inferx_artifacts
  PUBLIC inferx::base absl::status absl::statusor
  PRIVATE absl::strings simdjson::simdjson blake3::blake3)
if(INFERX_M3_ENABLE_OPENAT2)
  target_compile_definitions(inferx_artifacts PRIVATE INFERX_M3_USE_OPENAT2=1)
else()
  target_compile_definitions(inferx_artifacts PRIVATE INFERX_M3_USE_OPENAT2=0)
endif()
inferx_apply_project_options(inferx_artifacts)
