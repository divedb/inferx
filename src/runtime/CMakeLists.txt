# Hardware-neutral accounting, pool, workspace, and fence contracts.
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

# Hardware-neutral M5 execution contracts plus the CPU reference backend
# (ADR 0035). CUDA production realization lives under kernels/cuda/execution.
add_library(inferx_execution_runtime STATIC
  contiguous_kv_cache.cc
  cpu_execution_backend.cc
  model_memory_plan.cc
  single_request_runner.cc)
add_library(inferx::execution_runtime ALIAS inferx_execution_runtime)
set_target_properties(inferx_execution_runtime PROPERTIES EXPORT_NAME execution_runtime)
inferx_apply_project_options(inferx_execution_runtime)
target_link_libraries(inferx_execution_runtime
  PUBLIC inferx::base inferx::tensor inferx::runtime inferx::api inferx::engine
         inferx::scheduler inferx::model inferx::ops inferx::input
  PRIVATE $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::statusor>
          $<BUILD_INTERFACE:absl::strings>)
target_include_directories(inferx_execution_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Build-only until input/tokenization join an install/export set (ADR 0024
# packaging follow-up); the public headers install with the runtime tree.
