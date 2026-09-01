// Explicit CUDA Runtime API seam used by M2 production code and failure tests.
#ifndef INFERX_PLATFORM_CUDA_CUDA_API_H_
#define INFERX_PLATFORM_CUDA_CUDA_API_H_

#include <cuda_runtime_api.h>

#include <cstddef>

namespace inferx::cuda {

struct CudaApi {
  const char* (*get_error_name)(cudaError_t);
  const char* (*get_error_string)(cudaError_t);
  cudaError_t (*get_device_count)(int*);
  cudaError_t (*get_device)(int*);
  cudaError_t (*set_device)(int);
  cudaError_t (*get_device_properties)(cudaDeviceProp*, int);
  cudaError_t (*driver_get_version)(int*);
  cudaError_t (*runtime_get_version)(int*);
  cudaError_t (*mem_get_info)(size_t*, size_t*);
  cudaError_t (*device_get_stream_priority_range)(int*, int*);

  cudaError_t (*stream_create_with_priority)(cudaStream_t*, unsigned int, int);
  cudaError_t (*stream_destroy)(cudaStream_t);
  cudaError_t (*stream_wait_event)(cudaStream_t, cudaEvent_t, unsigned int);

  cudaError_t (*event_create_with_flags)(cudaEvent_t*, unsigned int);
  cudaError_t (*event_destroy)(cudaEvent_t);
  cudaError_t (*event_record)(cudaEvent_t, cudaStream_t);
  cudaError_t (*event_query)(cudaEvent_t);

  cudaError_t (*malloc_device)(void**, size_t);
  cudaError_t (*free_device)(void*);
  cudaError_t (*host_alloc)(void**, size_t, unsigned int);
  cudaError_t (*free_host)(void*);
  cudaError_t (*memcpy_async)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
  cudaError_t (*memset_async)(void*, int, size_t, cudaStream_t);
  cudaError_t (*pointer_get_attributes)(cudaPointerAttributes*, const void*);
  cudaError_t (*peek_at_last_error)();
  cudaError_t (*device_synchronize)();

  [[nodiscard]] static const CudaApi& Production() noexcept;

  [[nodiscard]] cudaError_t MemcpyAsync(void* destination, const void* source, size_t bytes,
                                        cudaMemcpyKind direction,
                                        cudaStream_t stream) const noexcept {
    if (memcpy_async == &cudaMemcpyAsync) [[likely]] {
      return cudaMemcpyAsync(destination, source, bytes, direction, stream);
    }
    return memcpy_async(destination, source, bytes, direction, stream);
  }

  [[nodiscard]] cudaError_t PointerGetAttributes(cudaPointerAttributes* attributes,
                                                 const void* address) const noexcept {
    if (pointer_get_attributes == &cudaPointerGetAttributes) [[likely]] {
      return cudaPointerGetAttributes(attributes, address);
    }
    return pointer_get_attributes(attributes, address);
  }
};

}  // namespace inferx::cuda

#endif  // INFERX_PLATFORM_CUDA_CUDA_API_H_
