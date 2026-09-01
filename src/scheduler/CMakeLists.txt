# inferx_scheduler: immutable scheduling views, capacity transactions, FCFS
# policy, and leased plan buffers (m1.md section 12; ADR 0010).
add_library(inferx_scheduler STATIC
  admission_controller.cc
  batch_planner.cc
  fcfs_policy.cc
  resource_accountant.cc
  step_plan.cc
  step_plan_validator.cc
  work_kind.cc)
add_library(inferx::scheduler ALIAS inferx_scheduler)
set_target_properties(inferx_scheduler PROPERTIES EXPORT_NAME scheduler)

inferx_apply_project_options(inferx_scheduler)
target_link_libraries(inferx_scheduler
  PUBLIC inferx::base inferx::api inferx::lifecycle
  PRIVATE $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::strings>)

target_include_directories(inferx_scheduler
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS inferx_scheduler
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/scheduler"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx"
        FILES_MATCHING PATTERN "*.h")
