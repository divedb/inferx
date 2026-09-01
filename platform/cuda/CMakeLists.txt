add_library(inferx_cuda STATIC
  src/cuda_allocator.cc
  src/cuda_api.cc
  src/cuda_copy.cc
  src/cuda_device.cc
  src/cuda_device_context.cc
  src/cuda_error.cc
  src/cuda_event_pool.cc
  src/cuda_metadata_ring.cc
  src/cuda_stream.cc)
inferx_apply_project_options(inferx_cuda)
target_include_directories(inferx_cuda
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(inferx_cuda
  PUBLIC inferx::base inferx::tensor inferx::runtime
  PRIVATE CUDA::cudart absl::status absl::statusor absl::strings)
set_target_properties(inferx_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
if(INFERX_BUILD_GPU_TESTS)
  target_compile_definitions(inferx_cuda PRIVATE INFERX_CUDA_DEBUG_POINTERS=1)
endif()

add_library(inferx_cuda_ops STATIC
  src/ops/cublaslt_gemm.cc
  src/ops/cuda_attention.cc
  src/ops/cuda_kernel_registry.cc
  src/ops/flashinfer_attention.cc)
inferx_apply_project_options(inferx_cuda_ops)
target_include_directories(inferx_cuda_ops
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/kernels/cuda/ops)
target_link_libraries(inferx_cuda_ops
  PUBLIC inferx::ops inferx_cuda
  PRIVATE inferx_cuda_ops_kernels CUDA::cudart CUDA::cublasLt
          absl::status absl::statusor absl::strings)
set_target_properties(inferx_cuda_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

# M2 keeps inferx_cuda build-only; no install/export rule is intentional.
