add_library(inferx_kernels_cuda_quantization STATIC fp8_quant.cu)
target_include_directories(inferx_kernels_cuda_quantization
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(inferx_kernels_cuda_quantization PRIVATE CUDA::cudart)
set_target_properties(inferx_kernels_cuda_quantization PROPERTIES
  CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  POSITION_INDEPENDENT_CODE ON)
if(INFERX_CUDA_ENABLE_LINEINFO)
  target_compile_options(inferx_kernels_cuda_quantization PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
