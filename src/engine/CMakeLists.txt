# inferx_engine: lifecycle ownership layer (m1.md sections 10-11). The
# LifecycleCoordinator/ResponseSink boundary lands with the simulator stage
# (M1.6) which fixes its concrete shape; the controller/registry/FSM here
# are the authoritative decision and ownership machinery.
add_library(inferx_engine STATIC
  lifecycle_coordinator.cc
  request_event.cc
  request_state_machine.cc
  request_controller.cc
  request_registry.cc)
add_library(inferx::engine ALIAS inferx_engine)
set_target_properties(inferx_engine PROPERTIES EXPORT_NAME engine)

inferx_apply_project_options(inferx_engine)
target_link_libraries(inferx_engine
  PUBLIC inferx::base inferx::api inferx::lifecycle inferx::scheduler
  PRIVATE $<BUILD_INTERFACE:absl::status> $<BUILD_INTERFACE:absl::strings>
        $<BUILD_INTERFACE:absl::flat_hash_map> $<BUILD_INTERFACE:absl::btree>)

target_include_directories(inferx_engine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS inferx_engine
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/engine"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx"
        FILES_MATCHING PATTERN "*.h")
