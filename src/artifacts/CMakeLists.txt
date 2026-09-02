add_library(inferx_artifacts STATIC
  artifact_file.cc
  artifact_limits.cc
  artifact_manifest.cc
  artifact_tensor.cc
  digest.cc
  hf_cache.cc
  hf_http_client.cc
  mapped_region.cc
  model_fingerprint.cc
  model_resolver.cc
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
  PUBLIC inferx::base
         $<BUILD_INTERFACE:absl::status>
         $<BUILD_INTERFACE:absl::statusor>
  PRIVATE $<BUILD_INTERFACE:absl::strings>
          simdjson::simdjson
          blake3::blake3)
if(INFERX_ENABLE_HF_HUB)
  target_sources(inferx_artifacts PRIVATE curl_http_client.cc)
  target_link_libraries(inferx_artifacts PRIVATE CURL::libcurl)
else()
  target_sources(inferx_artifacts PRIVATE hf_http_client_disabled.cc)
endif()
if(INFERX_ENABLE_OPENAT2)
  target_compile_definitions(inferx_artifacts PRIVATE INFERX_USE_OPENAT2=1)
else()
  target_compile_definitions(inferx_artifacts PRIVATE INFERX_USE_OPENAT2=0)
endif()
inferx_apply_project_options(inferx_artifacts)

# The bundled portable BLAKE3 archive is an implementation-only installed
# dependency. System-provider builds resolve blake3::blake3 from the package
# recorded in InferXConfig.cmake instead.
if(TARGET inferx_blake3)
  install(TARGETS inferx_blake3
          EXPORT InferXTargets
          ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

install(TARGETS inferx_artifacts
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/artifacts"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx"
        FILES_MATCHING PATTERN "*.h")
