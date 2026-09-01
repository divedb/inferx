# inferx_lifecycle: the shared closed state vocabulary (m1.md section 5).
add_library(inferx_lifecycle STATIC request_state.cc)
add_library(inferx::lifecycle ALIAS inferx_lifecycle)
set_target_properties(inferx_lifecycle PROPERTIES EXPORT_NAME lifecycle)

inferx_apply_project_options(inferx_lifecycle)
target_link_libraries(inferx_lifecycle
  PUBLIC inferx::base
  PRIVATE $<BUILD_INTERFACE:absl::status> $<BUILD_INTERFACE:absl::strings>)

target_include_directories(inferx_lifecycle
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS inferx_lifecycle
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/lifecycle"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx"
        FILES_MATCHING PATTERN "*.h")
