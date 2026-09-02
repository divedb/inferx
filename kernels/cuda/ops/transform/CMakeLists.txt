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

set(inferx_kernels_cuda_transform_srcs hadamard.cu)
set(inferx_kernels_cuda_transform_fl "")
if(EXISTS "${inferx_flashinfer_root}/include/flashinfer/pos_enc.cuh")
  list(APPEND inferx_kernels_cuda_transform_srcs flashinfer_rope.cu)
  set(inferx_kernels_cuda_transform_fl
      "${inferx_flashinfer_root}/include" ${inferx_kernels_cuda_cccl})
elseif(INFERX_ENABLE_FLASHINFER)
  message(FATAL_ERROR "INFERX_ENABLE_FLASHINFER=ON but the flashinfer gitlink is missing; "
          "run: python3 tools/deps/bootstrap.py --dependency flashinfer")
endif()

add_library(inferx_kernels_cuda_transform STATIC ${inferx_kernels_cuda_transform_srcs})
target_include_directories(inferx_kernels_cuda_transform
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE ${inferx_kernels_cuda_transform_fl})
target_link_libraries(inferx_kernels_cuda_transform PRIVATE CUDA::cudart)
set_target_properties(inferx_kernels_cuda_transform PROPERTIES
  CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  POSITION_INDEPENDENT_CODE ON)
if(INFERX_CUDA_ENABLE_LINEINFO)
  target_compile_options(inferx_kernels_cuda_transform PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
