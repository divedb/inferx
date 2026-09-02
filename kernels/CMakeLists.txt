# Unified kernels layer (ADR 0031).
#
# The neutral dispatch target builds on every preset (strictly CUDA-free; the
# CPU policy test enforces this). The CUDA backend and its operator categories
# are wired only under INFERX_ENABLE_CUDA, next to the platform kernel targets
# that remain untouched during the architecture transition.
add_library(inferx_kernels STATIC
  src/kernel_dispatch.cc
  src/reference/reference_elementwise.cc
  src/reference/reference_model_fused.cc
  src/reference/reference_select.cc)
add_library(inferx::kernels ALIAS inferx_kernels)

set_target_properties(inferx_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(inferx_kernels PUBLIC cxx_std_23)
target_include_directories(inferx_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(inferx_kernels
  PUBLIC inferx::ops inferx::tensor
  PRIVATE $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::statusor>
          $<BUILD_INTERFACE:absl::strings>)
inferx_apply_project_options(inferx_kernels)
# Build-only during the transition (no install/export surface yet); see ADR 0031.

if(INFERX_ENABLE_CUDA)
  foreach(category IN ITEMS activation attention attn_res conv embedding gemm
                            hyperconnection layernorm mhc moe quantization
                            sampling transform)
    add_subdirectory(cuda/ops/${category})
  endforeach()
  add_subdirectory(cuda/dispatch)
endif()
