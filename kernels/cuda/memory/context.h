#ifndef INFERX_KERNELS_CONTEXT_H_
#define INFERX_KERNELS_CONTEXT_H_

#include <cstdint>

namespace inferx::kernels {

/// \brief Device backends addressable through `ExecutionContext`.
enum class Backend : uint8_t { kCuda = 0, kRocm, kAscend, kCPU };

/// \brief A non-owning handle to a backend runtime stream.
///
/// Streams are opaque to the kernels API and may be null, which is a valid
/// no-op argument for operations that enqueue on a stream. The backend runtime
/// may have its own stream semantics, which the kernels API does not attempt to
/// enforce. The kernels API does not create or destroy streams; the caller is
/// responsible for their lifetime.
class Stream {
 public:
  /// \brief Constructs a null stream.
  constexpr Stream() noexcept = default;

  /// \brief Constructs a stream from a native handle.
  ///
  /// \param handle The native stream handle for the backend runtime.
  constexpr explicit Stream(void* handle) noexcept : handle_(handle) {}

  /// \brief Returns the native stream handle for the backend runtime.
  ///
  /// \return The native stream handle for the backend runtime.
  [[nodiscard]] constexpr void* Native() const noexcept { return handle_; }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return handle_ != nullptr;
  }

 private:
  void* handle_ = nullptr;
};

/// \brief The backend and stream one operation is enqueued on.
struct ExecutionContext {
  Backend backend = Backend::kCPU;
  Stream native_stream;
};

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_CONTEXT_H_
