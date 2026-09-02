set(inferx_cutlass_root "${CMAKE_SOURCE_DIR}/third_party/cutlass")
if(NOT EXISTS "${inferx_cutlass_root}/include/cutlass/cutlass.h")
  if(INFERX_ENABLE_CUTLASS)
    message(FATAL_ERROR "INFERX_ENABLE_CUTLASS=ON but the cutlass gitlink is missing")
  endif()
  return()
endif()

if(NOT TARGET inferx_kernels_cuda_gemm)
  return()
endif()

add_library(inferx_kernels_cuda_hyperconnection STATIC hc_mix.cu)
target_include_directories(inferx_kernels_cuda_hyperconnection
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include/.. ${CMAKE_SOURCE_DIR}/third_party/cutlass/include)
target_link_libraries(inferx_kernels_cuda_hyperconnection PRIVATE CUDA::cudart)
set_target_properties(inferx_kernels_cuda_hyperconnection PROPERTIES
  CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  POSITION_INDEPENDENT_CODE ON)
if(INFERX_CUDA_ENABLE_LINEINFO)
  target_compile_options(inferx_kernels_cuda_hyperconnection PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
