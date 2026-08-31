// Closed M2 storage dtype vocabulary. FP16/BF16 are storage-only here.
#ifndef INFERX_TENSOR_DTYPE_H_
#define INFERX_TENSOR_DTYPE_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "inferx/base/token.h"

namespace inferx {

enum class DType : uint8_t {
  kBool,
  kUInt8,
  kInt8,
  kUInt16,
  kInt16,
  kUInt32,
  kInt32,
  kUInt64,
  kInt64,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64,
};

[[nodiscard]] absl::StatusOr<ByteCount> DTypeSize(DType dtype);
[[nodiscard]] absl::StatusOr<absl::string_view> DTypeName(DType dtype);
[[nodiscard]] absl::StatusOr<bool> IsFloatingPoint(DType dtype);
[[nodiscard]] absl::StatusOr<bool> IsIntegral(DType dtype);

}  // namespace inferx

#endif  // INFERX_TENSOR_DTYPE_H_
