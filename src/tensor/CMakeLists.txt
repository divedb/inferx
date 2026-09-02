# CPU-only checked tensor and buffer contracts.
add_library(inferx_tensor STATIC
  buffer.cc
  cpu_allocator.cc
  device.cc
  dtype.cc
  shape.cc
  strides.cc
  tensor_view.cc)
add_library(inferx::tensor ALIAS inferx_tensor)
set_target_properties(inferx_tensor PROPERTIES EXPORT_NAME tensor)

inferx_apply_project_options(inferx_tensor)
target_link_libraries(inferx_tensor
  PUBLIC inferx::base
  PRIVATE $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::statusor>
          $<BUILD_INTERFACE:absl::strings>)
target_include_directories(inferx_tensor
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS inferx_tensor
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/tensor"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx")
