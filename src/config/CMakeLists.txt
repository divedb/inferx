# inferx_config: strict configuration pipeline. simdjson
# is a PRIVATE adapter dependency; no simdjson type appears in public headers.
add_library(inferx_config STATIC
  config_loader.cc
  engine_config.cc
  internal/simdjson_reader.cc)
add_library(inferx::config ALIAS inferx_config)
set_target_properties(inferx_config PROPERTIES EXPORT_NAME config)

inferx_apply_project_options(inferx_config)
target_link_libraries(inferx_config
  PUBLIC inferx::base
  PRIVATE $<BUILD_INTERFACE:simdjson::simdjson>
          $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::strings>)

target_include_directories(inferx_config
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Same private-dependency posture as Abseil in inferx::base: the installed
# static library carries no simdjson usage requirements (adapter-internal).
install(TARGETS inferx_config
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/config"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx"
        FILES_MATCHING PATTERN "*.h")
