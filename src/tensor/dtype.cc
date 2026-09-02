#include "inferx/tensor/dtype.h"

#include "absl/status/status.h"

namespace inferx {

absl::StatusOr<ByteCount> DtypeSize(Dtype dtype) {
  switch (dtype) {
    case Dtype::kBool:
    case Dtype::kUInt8:
    case Dtype::kInt8:
      return ByteCount(1);
    case Dtype::kUInt16:
    case Dtype::kInt16:
    case Dtype::kFloat16:
    case Dtype::kBFloat16:
      return ByteCount(2);
    case Dtype::kUInt32:
    case Dtype::kInt32:
    case Dtype::kFloat32:
      return ByteCount(4);
    case Dtype::kUInt64:
    case Dtype::kInt64:
    case Dtype::kFloat64:
      return ByteCount(8);
  }
  return absl::InvalidArgumentError("dtype: invalid dtype value");
}

absl::StatusOr<absl::string_view> DtypeName(Dtype dtype) {
  switch (dtype) {
    case Dtype::kBool:
      return "bool";
    case Dtype::kUInt8:
      return "uint8";
    case Dtype::kInt8:
      return "int8";
    case Dtype::kUInt16:
      return "uint16";
    case Dtype::kInt16:
      return "int16";
    case Dtype::kUInt32:
      return "uint32";
    case Dtype::kInt32:
      return "int32";
    case Dtype::kUInt64:
      return "uint64";
    case Dtype::kInt64:
      return "int64";
    case Dtype::kFloat16:
      return "float16";
    case Dtype::kBFloat16:
      return "bfloat16";
    case Dtype::kFloat32:
      return "float32";
    case Dtype::kFloat64:
      return "float64";
  }
  return absl::InvalidArgumentError("dtype: invalid dtype value");
}

absl::StatusOr<bool> IsFloatingPoint(Dtype dtype) {
  absl::StatusOr<ByteCount> size = DtypeSize(dtype);
  if (!size.ok()) {
    return size.status();
  }
  return dtype == Dtype::kFloat16 || dtype == Dtype::kBFloat16 || dtype == Dtype::kFloat32 ||
         dtype == Dtype::kFloat64;
}

absl::StatusOr<bool> IsIntegral(Dtype dtype) {
  absl::StatusOr<ByteCount> size = DtypeSize(dtype);
  if (!size.ok()) {
    return size.status();
  }
  return dtype != Dtype::kFloat16 && dtype != Dtype::kBFloat16 && dtype != Dtype::kFloat32 &&
         dtype != Dtype::kFloat64;
}

}  // namespace inferx
