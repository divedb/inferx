add_library(inferx_ops STATIC
  attention_reference.cc
  cpu_op_executor.cc
  cpu_reference_executor.cc
  embedding_reference.cc
  gemm_reference.cc
  kernel_key.cc
  kernel_registry.cc
  logits_reference.cc
  rms_norm_reference.cc
  rope_reference.cc
  swiglu_reference.cc
  validation_internal.cc
  warmup.cc)
add_library(inferx::ops ALIAS inferx_ops)

set_target_properties(inferx_ops PROPERTIES
  EXPORT_NAME ops
  POSITION_INDEPENDENT_CODE ON)
target_compile_features(inferx_ops PUBLIC cxx_std_23)
target_include_directories(inferx_ops
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(inferx_ops
  PUBLIC inferx::base inferx::tensor
  PRIVATE $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::statusor>
          $<BUILD_INTERFACE:absl::strings>)
inferx_apply_project_options(inferx_ops)

install(TARGETS inferx_ops
        EXPORT InferXTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/inferx/ops"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/inferx")
