# Hardware-neutral M2 accounting, pool, workspace, and fence contracts.
add_library(inferx_runtime STATIC
  buffer_pool.cc
  completion_fence.cc
  memory_tracker.cc
  workspace.cc)
add_library(inferx::runtime ALIAS inferx_runtime)
set_target_properties(inferx_runtime PROPERTIES EXPORT_NAME runtime)

inferx_apply_project_options(inferx_runtime)
target_link_libraries(inferx_runtime
  PUBLIC inferx::base inferx::tensor
  PRIVATE $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::statusor>
          $<BUILD_INTERFACE:absl::strings>)
target_include_directories(inferx_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS inferx_runtime
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/runtime"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx")
