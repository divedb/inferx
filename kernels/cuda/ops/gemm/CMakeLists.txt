# GEMM category kernels: custom CUTLASS implementation (ADR 0032).
#
# The CUTLASS provider builds only when the dependency is enabled and its
# gitlink is initialized; without it the gemm/logits operators report
# Unimplemented from the new architecture (the M4 cuBLASLt path in
# platform/cuda remains available and untouched).
if(INFERX_ENABLE_CUTLASS)
  add_library(inferx_kernels_cuda_gemm STATIC cutlass_gemm.cu)
  target_include_directories(inferx_kernels_cuda_gemm
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cutlass/include)
  target_link_libraries(inferx_kernels_cuda_gemm PRIVATE CUDA::cudart)
  set_target_properties(inferx_kernels_cuda_gemm PROPERTIES
    CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON)
  if(INFERX_CUDA_ENABLE_LINEINFO)
    target_compile_options(inferx_kernels_cuda_gemm PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()
endif()
