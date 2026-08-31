#include "inferx/artifacts/safetensors_reader.h"

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/artifacts/json_internal.h"

namespace inferx::artifacts {
namespace {

uint64_t DecodeLittleEndianU64(const std::array<std::byte, 8>& bytes) {
  uint64_t value = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    value |= static_cast<uint64_t>(std::to_integer<uint8_t>(bytes[i])) << (8 * i);
  }
  return value;
}

absl::StatusOr<ArtifactTensor> ParseTensor(std::string name, simdjson::dom::element value,
                                           uint64_t data_start, uint64_t file_size,
                                           const ArtifactLimits& limits) {
  const std::string path = absl::StrCat("/", name);
  auto object = internal::Object(value, path);
  if (!object.ok()) return object.status();
  if (auto status = internal::CheckUniqueKeys(*object, path); !status.ok()) {
    return status;
  }
  std::set<std::string> seen;
  for (const auto field : *object) {
    seen.emplace(field.key);
  }
  const std::set<std::string> expected = {"data_offsets", "dtype", "shape"};
  if (seen != expected) {
    return internal::JsonError(path,
                               "tensor object must contain exactly dtype, "
                               "shape, and data_offsets");
  }

  auto dtype_value = internal::Required(*object, "dtype", path);
  if (!dtype_value.ok()) return dtype_value.status();
  auto dtype_name = internal::String(*dtype_value, path + "/dtype");
  if (!dtype_name.ok()) return dtype_name.status();
  auto dtype = ParseArtifactDType(*dtype_name);
  if (!dtype.ok()) {
    return internal::JsonError(path + "/dtype", dtype.status().message());
  }

  auto shape_value = internal::Required(*object, "shape", path);
  if (!shape_value.ok()) return shape_value.status();
  auto shape_array = internal::Array(*shape_value, path + "/shape");
  if (!shape_array.ok()) return shape_array.status();
  if (shape_array->size() > limits.max_tensor_rank) {
    return absl::ResourceExhaustedError(
        absl::StrCat(path, "/shape: tensor rank exceeds configured limit"));
  }
  ArtifactShape shape;
  shape.reserve(shape_array->size());
  size_t dimension_index = 0;
  for (simdjson::dom::element dimension : *shape_array) {
    auto parsed = internal::Uint64(dimension, absl::StrCat(path, "/shape/", dimension_index));
    if (!parsed.ok()) return parsed.status();
    shape.push_back(*parsed);
    ++dimension_index;
  }

  auto offsets_value = internal::Required(*object, "data_offsets", path);
  if (!offsets_value.ok()) return offsets_value.status();
  auto offsets_array = internal::Array(*offsets_value, path + "/data_offsets");
  if (!offsets_array.ok()) return offsets_array.status();
  if (offsets_array->size() != 2) {
    return internal::JsonError(path + "/data_offsets", "expected exactly [begin, end]");
  }
  std::array<uint64_t, 2> offsets{};
  size_t offset_index = 0;
  for (simdjson::dom::element offset : *offsets_array) {
    auto parsed = internal::Uint64(offset, absl::StrCat(path, "/data_offsets/", offset_index));
    if (!parsed.ok()) return parsed.status();
    offsets[offset_index++] = *parsed;
  }
  if (offsets[0] > offsets[1]) {
    return absl::DataLossError(absl::StrCat(path, ": tensor data offsets are reversed"));
  }
  auto packed_size = PackedTensorBytes(*dtype, shape);
  if (!packed_size.ok()) return packed_size.status();
  if (offsets[1] - offsets[0] != *packed_size) {
    return absl::DataLossError(
        absl::StrCat(path, ": dtype/shape byte size does not match offsets"));
  }
  if (offsets[1] > file_size - data_start) {
    return absl::DataLossError(absl::StrCat(path, ": tensor range exceeds safetensors payload"));
  }
  if (data_start > std::numeric_limits<uint64_t>::max() - offsets[0]) {
    return absl::OutOfRangeError("absolute tensor file offset overflows");
  }
  return ArtifactTensor{std::move(name),         *dtype,
                        std::move(shape),        ArtifactByteRange{offsets[0], *packed_size},
                        data_start + offsets[0], *packed_size};
}

}  // namespace

const ArtifactTensor* SafeTensorsMetadata::Find(std::string_view name) const {
  const auto iterator = std::lower_bound(
      tensors.begin(), tensors.end(), name,
      [](const ArtifactTensor& tensor, std::string_view target) { return tensor.name < target; });
  return iterator != tensors.end() && iterator->name == name ? &*iterator : nullptr;
}

absl::StatusOr<SafeTensorsMetadata> SafeTensorReader::ReadHeader(
    const ArtifactFile& file, const ArtifactLimits& limits) const {
  if (auto status = limits.Validate(); !status.ok()) return status;
  if (file.identity().size < 8) {
    return absl::DataLossError("safetensors file is shorter than 8 bytes");
  }
  std::array<std::byte, 8> length_bytes{};
  if (auto status = file.ReadExact(0, length_bytes); !status.ok()) return status;
  const uint64_t header_size = DecodeLittleEndianU64(length_bytes);
  if (header_size == 0 || header_size > limits.max_safetensors_header_bytes) {
    return absl::ResourceExhaustedError(
        "safetensors header length is outside the configured range");
  }
  if (header_size > file.identity().size - 8 || header_size > std::numeric_limits<size_t>::max()) {
    return absl::DataLossError("safetensors header length exceeds the opened file");
  }
  const uint64_t data_start = 8 + header_size;
  std::vector<std::byte> header(static_cast<size_t>(header_size));
  if (auto status = file.ReadExact(8, header); !status.ok()) return status;
  if (header.empty() || header.front() != std::byte{'{'}) {
    return absl::DataLossError("safetensors header must begin with '{'");
  }
  size_t last = header.size();
  while (last > 0 && header[last - 1] == std::byte{' '}) --last;
  if (last == 0 || header[last - 1] != std::byte{'}'}) {
    return absl::DataLossError("safetensors header permits only ASCII-space trailing padding");
  }

  simdjson::dom::parser parser;
  if (const auto error = parser.allocate(header.size(), limits.max_json_depth); error) {
    return internal::ParseError(file.relative_path(), error);
  }
  simdjson::dom::element document;
  const auto parse_error =
      parser.parse(reinterpret_cast<const uint8_t*>(header.data()), header.size(), true)
          .get(document);
  if (parse_error) return internal::ParseError(file.relative_path(), parse_error);
  auto root = internal::Object(document, "");
  if (!root.ok()) return root.status();
  if (auto status = internal::CheckUniqueKeys(*root, ""); !status.ok()) {
    return status;
  }
  if (root->size() > limits.max_tensors + 1) {
    return absl::ResourceExhaustedError("safetensors tensor count exceeds configured limit");
  }

  SafeTensorsMetadata result;
  result.header_size = header_size;
  result.data_start = data_start;
  result.file_size = file.identity().size;
  result.tensors.reserve(root->size());
  for (const auto field : *root) {
    std::string name(field.key);
    if (name == "__metadata__") {
      auto metadata = internal::Object(field.value, "/__metadata__");
      if (!metadata.ok()) return metadata.status();
      if (auto status = internal::CheckUniqueKeys(*metadata, "/__metadata__"); !status.ok()) {
        return status;
      }
      for (const auto item : *metadata) {
        auto value = internal::String(item.value, absl::StrCat("/__metadata__/", item.key));
        if (!value.ok()) return value.status();
        result.user_metadata.emplace(std::string(item.key), std::move(*value));
      }
      continue;
    }
    if (name.empty()) {
      return internal::JsonError("/", "tensor name must not be empty");
    }
    auto tensor =
        ParseTensor(std::move(name), field.value, data_start, file.identity().size, limits);
    if (!tensor.ok()) return tensor.status();
    result.tensors.push_back(std::move(*tensor));
  }
  if (result.tensors.size() > limits.max_tensors) {
    return absl::ResourceExhaustedError("safetensors tensor count exceeds configured limit");
  }

  std::vector<const ArtifactTensor*> by_offset;
  by_offset.reserve(result.tensors.size());
  for (const auto& tensor : result.tensors) by_offset.push_back(&tensor);
  std::sort(by_offset.begin(), by_offset.end(),
            [](const ArtifactTensor* lhs, const ArtifactTensor* rhs) {
              return std::tie(lhs->data.offset, lhs->data.size, lhs->name) <
                     std::tie(rhs->data.offset, rhs->data.size, rhs->name);
            });
  uint64_t cursor = 0;
  for (const ArtifactTensor* tensor : by_offset) {
    if (tensor->data.size == 0) {
      if (tensor->data.offset != cursor) {
        return absl::DataLossError("zero-byte tensor is not positioned at a payload boundary");
      }
      continue;
    }
    if (tensor->data.offset != cursor) {
      return absl::DataLossError(tensor->data.offset < cursor
                                     ? "safetensors tensor payload ranges overlap"
                                     : "safetensors tensor payload contains a hole");
    }
    cursor += tensor->data.size;
  }
  if (cursor != file.identity().size - data_start) {
    return absl::DataLossError("safetensors tensor ranges do not cover the complete payload");
  }
  std::sort(
      result.tensors.begin(), result.tensors.end(),
      [](const ArtifactTensor& lhs, const ArtifactTensor& rhs) { return lhs.name < rhs.name; });
  if (auto status = file.CheckUnchanged(); !status.ok()) return status;
  return result;
}

}  // namespace inferx::artifacts
