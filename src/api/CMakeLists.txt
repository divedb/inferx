add_library(inferx_api STATIC
  generate_request.cc
  response_event.cc)
add_library(inferx::api ALIAS inferx_api)
set_target_properties(inferx_api PROPERTIES EXPORT_NAME api)

inferx_apply_project_options(inferx_api)
target_link_libraries(inferx_api
  PUBLIC inferx::base
  PRIVATE $<BUILD_INTERFACE:absl::status> $<BUILD_INTERFACE:absl::strings>)

target_include_directories(inferx_api
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS inferx_api
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/api"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx"
        FILES_MATCHING PATTERN "*.h")
