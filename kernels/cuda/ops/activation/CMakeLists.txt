# activation category kernels (ADR 0031 taxonomy; owned correctness kernels, ADR 0030 style).
add_library(inferx_kernels_cuda_activation STATIC swiglu.cu residual.cu)
target_include_directories(inferx_kernels_cuda_activation
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(inferx_kernels_cuda_activation PRIVATE CUDA::cudart)
set_target_properties(inferx_kernels_cuda_activation PROPERTIES
  CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  POSITION_INDEPENDENT_CODE ON)
if(INFERX_CUDA_ENABLE_LINEINFO)
  target_compile_options(inferx_kernels_cuda_activation PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
