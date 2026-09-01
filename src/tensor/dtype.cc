#include "inferx/tensor/dtype.h"

#include "absl/status/status.h"

namespace inferx {

absl::StatusOr<ByteCount> DTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kBool:
    case DType::kUInt8:
    case DType::kInt8:
      return ByteCount(1);
    case DType::kUInt16:
    case DType::kInt16:
    case DType::kFloat16:
    case DType::kBFloat16:
      return ByteCount(2);
    case DType::kUInt32:
    case DType::kInt32:
    case DType::kFloat32:
      return ByteCount(4);
    case DType::kUInt64:
    case DType::kInt64:
    case DType::kFloat64:
      return ByteCount(8);
  }
  return absl::InvalidArgumentError("dtype: invalid dtype value");
}

absl::StatusOr<absl::string_view> DTypeName(DType dtype) {
  switch (dtype) {
    case DType::kBool:
      return "bool";
    case DType::kUInt8:
      return "uint8";
    case DType::kInt8:
      return "int8";
    case DType::kUInt16:
      return "uint16";
    case DType::kInt16:
      return "int16";
    case DType::kUInt32:
      return "uint32";
    case DType::kInt32:
      return "int32";
    case DType::kUInt64:
      return "uint64";
    case DType::kInt64:
      return "int64";
    case DType::kFloat16:
      return "float16";
    case DType::kBFloat16:
      return "bfloat16";
    case DType::kFloat32:
      return "float32";
    case DType::kFloat64:
      return "float64";
  }
  return absl::InvalidArgumentError("dtype: invalid dtype value");
}

absl::StatusOr<bool> IsFloatingPoint(DType dtype) {
  absl::StatusOr<ByteCount> size = DTypeSize(dtype);
  if (!size.ok()) {
    return size.status();
  }
  return dtype == DType::kFloat16 || dtype == DType::kBFloat16 || dtype == DType::kFloat32 ||
         dtype == DType::kFloat64;
}

absl::StatusOr<bool> IsIntegral(DType dtype) {
  absl::StatusOr<ByteCount> size = DTypeSize(dtype);
  if (!size.ok()) {
    return size.status();
  }
  return dtype != DType::kFloat16 && dtype != DType::kBFloat16 && dtype != DType::kFloat32 &&
         dtype != DType::kFloat64;
}

}  // namespace inferx
