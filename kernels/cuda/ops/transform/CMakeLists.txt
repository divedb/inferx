# Transform category kernels: FlashInfer RoPE instantiation (ADR 0032).
#
# Builds only when the flashinfer gitlink is initialized; with the feature
# disabled the category has no kernels and the provider chain reports rope
# Unimplemented from the new architecture.
set(inferx_flashinfer_root "${CMAKE_SOURCE_DIR}/third_party/flashinfer")
if(NOT EXISTS "${inferx_flashinfer_root}/include/flashinfer/pos_enc.cuh")
  if(INFERX_ENABLE_FLASHINFER)
    message(FATAL_ERROR
            "INFERX_ENABLE_FLASHINFER=ON but the flashinfer gitlink is not initialized; "
            "run: python3 tools/deps/bootstrap.py --dependency flashinfer")
  endif()
  return()
endif()

add_library(inferx_kernels_cuda_transform STATIC flashinfer_rope.cu)
target_include_directories(inferx_kernels_cuda_transform
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE ${inferx_flashinfer_root}/include
          ${inferx_flashinfer_root}/3rdparty/cccl/libcudacxx/include)
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
