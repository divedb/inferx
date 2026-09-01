add_library(inferx_cuda_ops_kernels STATIC ops_kernels.cu)
target_include_directories(inferx_cuda_ops_kernels
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(inferx_cuda_ops_kernels PRIVATE CUDA::cudart)
set_target_properties(inferx_cuda_ops_kernels PROPERTIES
  CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  POSITION_INDEPENDENT_CODE ON)
if(INFERX_CUDA_ENABLE_LINEINFO)
  target_compile_options(inferx_cuda_ops_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
