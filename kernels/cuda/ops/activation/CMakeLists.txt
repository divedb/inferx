set(inferx_flashinfer_root "${CMAKE_SOURCE_DIR}/third_party/flashinfer")
if(NOT EXISTS "${inferx_flashinfer_root}/include/flashinfer/norm.cuh")
  set(inferx_flashinfer_present OFF)
else()
  set(inferx_flashinfer_present ON)
endif()
set(inferx_kernels_cuda_cccl
    "${inferx_flashinfer_root}/3rdparty/cccl/libcudacxx/include"
    "${inferx_flashinfer_root}/3rdparty/cccl/cub"
    "${inferx_flashinfer_root}/3rdparty/cccl/thrust")

set(inferx_kernels_cuda_activation_srcs swiglu.cu residual.cu add3.cu)
set(inferx_kernels_cuda_activation_fl "")
if(inferx_flashinfer_present)
  list(APPEND inferx_kernels_cuda_activation_srcs act_and_mul.cu)
  set(inferx_kernels_cuda_activation_fl
      "${inferx_flashinfer_root}/include" ${inferx_kernels_cuda_cccl})
endif()

add_library(inferx_kernels_cuda_activation STATIC ${inferx_kernels_cuda_activation_srcs})
target_include_directories(inferx_kernels_cuda_activation
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE ${inferx_kernels_cuda_activation_fl})
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
