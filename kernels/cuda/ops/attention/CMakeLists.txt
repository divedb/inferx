# Attention category: FlashInfer paged decode/prefill mapped from the
# InferX contiguous BSHD cache (ADR 0032). FlashInfer-gated.
set(inferx_flashinfer_root "${CMAKE_SOURCE_DIR}/third_party/flashinfer")
if(NOT EXISTS "${inferx_flashinfer_root}/include/flashinfer/attention/decode.cuh")
  if(INFERX_ENABLE_FLASHINFER)
    message(FATAL_ERROR "INFERX_ENABLE_FLASHINFER=ON but the flashinfer gitlink is missing; "
            "run: python3 tools/deps/bootstrap.py --dependency flashinfer")
  endif()
  return()
endif()

add_library(inferx_kernels_cuda_attention STATIC flashinfer_attention.cu)
target_include_directories(inferx_kernels_cuda_attention
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE ${inferx_flashinfer_root}/include
          ${inferx_flashinfer_root}/3rdparty/cccl/libcudacxx/include
          ${inferx_flashinfer_root}/3rdparty/cccl/cub
          ${inferx_flashinfer_root}/3rdparty/cccl/thrust)
target_link_libraries(inferx_kernels_cuda_attention PRIVATE CUDA::cudart)
set_target_properties(inferx_kernels_cuda_attention PROPERTIES
  CUDA_STANDARD ${INFERX_CUDA_CXX_STANDARD}
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  POSITION_INDEPENDENT_CODE ON)
if(INFERX_CUDA_ENABLE_LINEINFO)
  target_compile_options(inferx_kernels_cuda_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()
