add_library(inferx_cuda_test_kernels STATIC
  strided_copy.cu
  synchronization_probe.cu
  bounded_delay.cu
  fault_injection.cu)
target_include_directories(inferx_cuda_test_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(inferx_cuda_test_kernels PUBLIC CUDA::cudart)
set_target_properties(inferx_cuda_test_kernels PROPERTIES
  CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(inferx_cuda_test_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${INFERX_CUDA_ENABLE_LINEINFO}>>:-lineinfo>)
