# CUDA backend for the unified kernels dispatcher (ADR 0031).
#
# Host-side provider/backend code plus conditional external-provider wiring.
# The hpc-ops probes always compile (they carry the audit answer and link no
# hpc-ops code); flashinfer/cutlass providers compile only when their
# gitlinks are initialized, and the matching category libraries must exist
# (built under kernels/cuda/ops/<category>).
set(inferx_kernels_cuda_srcs
  cuda_kernel_backend.cc
  cuda_tensor_checks.cc
  custom_providers.cc
  hpc_ops_providers.cc
  op_launchers.cc
  op_providers.cc
  op_validation.cc)

set(inferx_kernels_cuda_private_includes ${CMAKE_SOURCE_DIR}/platform/cuda/include)
set(inferx_kernels_cuda_private_links "")

set(inferx_flashinfer_root "${CMAKE_SOURCE_DIR}/third_party/flashinfer")
if(EXISTS "${inferx_flashinfer_root}/include/flashinfer/norm.cuh")
  list(APPEND inferx_kernels_cuda_srcs flashinfer_providers.cc)
  list(APPEND inferx_kernels_cuda_private_links
       inferx_kernels_cuda_layernorm inferx_kernels_cuda_transform)
endif()

set(inferx_cutlass_root "${CMAKE_SOURCE_DIR}/third_party/cutlass")
if(EXISTS "${inferx_cutlass_root}/include/cutlass/cutlass.h")
  list(APPEND inferx_kernels_cuda_srcs cutlass_gemm_provider.cc)
  list(APPEND inferx_kernels_cuda_private_links inferx_kernels_cuda_gemm)
endif()

add_library(inferx_kernels_cuda STATIC ${inferx_kernels_cuda_srcs})
add_library(inferx::kernels_cuda ALIAS inferx_kernels_cuda)
target_include_directories(inferx_kernels_cuda
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${inferx_kernels_cuda_private_includes})
target_link_libraries(inferx_kernels_cuda
  PUBLIC inferx::kernels CUDA::cudart
  PRIVATE ${inferx_kernels_cuda_private_links}
          inferx_kernels_cuda_activation
          inferx_kernels_cuda_attention
          inferx_kernels_cuda_attn_res
          inferx_kernels_cuda_conv
          inferx_kernels_cuda_embedding
          inferx_kernels_cuda_hyperconnection
          inferx_kernels_cuda_mhc
          inferx_kernels_cuda_moe
          inferx_kernels_cuda_quantization
          inferx_kernels_cuda_sampling
          inferx_kernels_cuda_transform
          $<BUILD_INTERFACE:absl::status>
          $<BUILD_INTERFACE:absl::statusor>
          $<BUILD_INTERFACE:absl::strings>)
inferx_apply_project_options(inferx_kernels_cuda)

if(EXISTS "${inferx_flashinfer_root}/include/flashinfer/norm.cuh")
  target_compile_definitions(inferx_kernels_cuda PRIVATE INFERX_KERNELS_HAVE_FLASHINFER)
endif()
if(EXISTS "${inferx_cutlass_root}/include/cutlass/cutlass.h")
  target_compile_definitions(inferx_kernels_cuda PRIVATE INFERX_KERNELS_HAVE_CUTLASS)
endif()
# Build-only during the transition; see ADR 0031.
