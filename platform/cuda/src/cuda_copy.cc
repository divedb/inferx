#include "inferx/platform/cuda/cuda_copy.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "inferx/base/checked_math.h"
#include "inferx/platform/cuda/cuda_buffer_access.h"

namespace inferx::cuda {
namespace {

bool RangesOverlap(ByteRange left, ByteRange right) {
  if (left.size.value() == 0 || right.size.value() == 0) return false;
  return left.offset.value() < right.offset.value() + right.size.value() &&
         right.offset.value() < left.offset.value() + left.size.value();
}

absl::Status ValidatePointer(const CudaApi& api, const BufferView& view, DeviceId device,
                             CudaHealth* health) {
#if !defined(NDEBUG) || defined(INFERX_CUDA_DEBUG_POINTERS)
  cudaPointerAttributes attributes{};
  const cudaError_t error = api.PointerGetAttributes(&attributes, BufferAccess::Address(view));
  if (error != cudaSuccess) {
    return CudaErrorStatus(error, "pointer-attributes", device, health);
  }
  const cudaMemoryType expected =
      view.memory_kind() == MemoryKind::kDevice ? cudaMemoryTypeDevice : cudaMemoryTypeHost;
  if (attributes.type != expected) {
    return absl::InvalidArgumentError(
        "cuda.copy.pointer: runtime pointer kind differs from buffer metadata");
  }
#else
  static_cast<void>(api);
  static_cast<void>(view);
  static_cast<void>(device);
  static_cast<void>(health);
#endif
  return absl::OkStatus();
}

}  // namespace

absl::Status CopyAsync(const CopyRequest& request, CudaStream& stream, const CudaApi& api,
                       CudaHealth* health) {
  if (health != nullptr) {
    absl::Status status = health->CheckAcceptingWork();
    if (!status.ok()) return status;
  }
  if (request.bytes.value() > request.source.size().value() ||
      request.bytes.value() > request.destination.size().value()) {
    return absl::OutOfRangeError("cuda.copy.bytes: copy exceeds source or destination view");
  }
  if (request.bytes.value() == 0) return absl::OkStatus();
  const MemoryKind source_kind = request.source.memory_kind();
  const MemoryKind destination_kind = request.destination.memory_kind();
  if (source_kind == MemoryKind::kManaged || destination_kind == MemoryKind::kManaged) {
    return absl::UnimplementedError("cuda.copy: managed memory is not supported in M2");
  }
  if (source_kind == MemoryKind::kHost || destination_kind == MemoryKind::kHost) {
    return absl::FailedPreconditionError(
        "cuda.copy: normal asynchronous host copies require pinned memory");
  }
  cudaMemcpyKind direction = cudaMemcpyDefault;
  if (source_kind == MemoryKind::kPinnedHost && destination_kind == MemoryKind::kDevice) {
    direction = cudaMemcpyHostToDevice;
    if (request.destination.device() != Device::Cuda(stream.device())) {
      return absl::InvalidArgumentError("cuda.copy: destination and stream devices differ");
    }
  } else if (source_kind == MemoryKind::kDevice && destination_kind == MemoryKind::kPinnedHost) {
    direction = cudaMemcpyDeviceToHost;
    if (request.source.device() != Device::Cuda(stream.device())) {
      return absl::InvalidArgumentError("cuda.copy: source and stream devices differ");
    }
  } else if (source_kind == MemoryKind::kDevice && destination_kind == MemoryKind::kDevice) {
    direction = cudaMemcpyDeviceToDevice;
    if (request.source.device() != request.destination.device() ||
        request.source.device() != Device::Cuda(stream.device())) {
      return absl::InvalidArgumentError(
          "cuda.copy: D2D source, destination, and stream devices must match");
    }
  } else {
    return absl::InvalidArgumentError(
        "cuda.copy.direction: unsupported asynchronous copy direction");
  }
  if (request.source.allocation_id() == request.destination.allocation_id() &&
      RangesOverlap(ByteRange{request.source.range().offset, request.bytes},
                    ByteRange{request.destination.range().offset, request.bytes})) {
    return absl::InvalidArgumentError("cuda.copy.alias: overlapping copy is unsupported");
  }
  absl::Status source_pointer = ValidatePointer(api, request.source, stream.device(), health);
  if (!source_pointer.ok()) return source_pointer;
  absl::Status destination_pointer =
      ValidatePointer(api, request.destination.AsConst(), stream.device(), health);
  if (!destination_pointer.ok()) return destination_pointer;
  absl::StatusOr<size_t> bytes = CheckedNarrow<size_t>(request.bytes.value(), "cuda.copy.bytes");
  if (!bytes.ok()) return bytes.status();
  return CudaErrorStatus(
      api.MemcpyAsync(BufferAccess::Address(request.destination),
                      BufferAccess::Address(request.source), *bytes, direction, stream.handle()),
      "memcpy-async", stream.device(), health);
}

absl::Status MemsetAsync(MutableBufferView destination, uint8_t value, CudaStream& stream,
                         const CudaApi& api, CudaHealth* health) {
  if (health != nullptr) {
    absl::Status status = health->CheckAcceptingWork();
    if (!status.ok()) return status;
  }
  if (destination.memory_kind() != MemoryKind::kDevice ||
      destination.device() != Device::Cuda(stream.device())) {
    return absl::InvalidArgumentError(
        "cuda.memset: destination must be device memory on the stream device");
  }
  if (destination.size().value() == 0) return absl::OkStatus();
  absl::Status pointer = ValidatePointer(api, destination.AsConst(), stream.device(), health);
  if (!pointer.ok()) return pointer;
  absl::StatusOr<size_t> bytes =
      CheckedNarrow<size_t>(destination.size().value(), "cuda.memset.bytes");
  if (!bytes.ok()) return bytes.status();
  return CudaErrorStatus(
      api.memset_async(BufferAccess::Address(destination), value, *bytes, stream.handle()),
      "memset-async", stream.device(), health);
}

}  // namespace inferx::cuda
