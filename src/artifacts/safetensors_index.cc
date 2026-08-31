#include "inferx/artifacts/safetensors_index.h"

#include <simdjson.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "json_internal.h"

namespace inferx::artifacts {

absl::StatusOr<SafetensorsIndex> SafetensorsIndexReader::Read(const ArtifactFile& file,
                                                              const ArtifactLimits& limits) const {
  if (auto status = limits.Validate(); !status.ok()) return status;
  auto bytes = file.ReadAll(limits.max_json_bytes);
  if (!bytes.ok()) return bytes.status();

  simdjson::dom::parser parser;
  if (const auto error = parser.allocate(bytes->size(), limits.max_json_depth); error) {
    return internal::ParseError(file.relative_path(), error);
  }
  simdjson::dom::element document;
  const auto parse_error =
      parser.parse(reinterpret_cast<const uint8_t*>(bytes->data()), bytes->size(), true)
          .get(document);
  if (parse_error) return internal::ParseError(file.relative_path(), parse_error);
  auto root = internal::Object(document, "");
  if (!root.ok()) return root.status();
  if (auto status = internal::CheckUniqueKeys(*root, ""); !status.ok()) {
    return status;
  }
  for (const auto field : *root) {
    if (field.key != "weight_map" && field.key != "metadata") {
      return internal::JsonError(absl::StrCat("/", field.key), "unknown shard-index field");
    }
  }

  auto map_value = internal::Required(*root, "weight_map", "");
  if (!map_value.ok()) return map_value.status();
  auto weight_map = internal::Object(*map_value, "/weight_map");
  if (!weight_map.ok()) return weight_map.status();
  if (auto status = internal::CheckUniqueKeys(*weight_map, "/weight_map"); !status.ok()) {
    return status;
  }
  if (weight_map->size() == 0) {
    return internal::JsonError("/weight_map", "map must not be empty");
  }
  if (weight_map->size() > limits.max_tensors) {
    return absl::ResourceExhaustedError("shard index tensor count exceeds configured limit");
  }

  SafetensorsIndex result;
  result.entries.reserve(weight_map->size());
  std::set<SafeRelativePath> unique_shards;
  for (const auto field : *weight_map) {
    const std::string name(field.key);
    if (name.empty()) {
      return internal::JsonError("/weight_map", "tensor name is empty");
    }
    auto shard_name = internal::String(field.value, absl::StrCat("/weight_map/", name));
    if (!shard_name.ok()) return shard_name.status();
    if (!std::string_view(*shard_name).ends_with(".safetensors")) {
      return internal::JsonError(absl::StrCat("/weight_map/", name),
                                 "shard path must end in .safetensors");
    }
    auto shard = SafeRelativePath::Parse(*shard_name);
    if (!shard.ok()) {
      return internal::JsonError(absl::StrCat("/weight_map/", name), shard.status().message());
    }
    unique_shards.insert(*shard);
    result.entries.push_back({name, std::move(*shard)});
  }
  if (unique_shards.size() > limits.max_shards) {
    return absl::ResourceExhaustedError("shard index shard count exceeds configured limit");
  }
  result.shards.assign(unique_shards.begin(), unique_shards.end());
  std::sort(result.entries.begin(), result.entries.end(),
            [](const SafetensorsIndexEntry& lhs, const SafetensorsIndexEntry& rhs) {
              return lhs.tensor_name < rhs.tensor_name;
            });

  simdjson::dom::element metadata_value;
  const auto metadata_error = root->at_key("metadata").get(metadata_value);
  if (!metadata_error) {
    auto metadata = internal::Object(metadata_value, "/metadata");
    if (!metadata.ok()) return metadata.status();
    if (auto status = internal::CheckUniqueKeys(*metadata, "/metadata"); !status.ok()) {
      return status;
    }
    simdjson::dom::element total_value;
    const auto total_error = metadata->at_key("total_size").get(total_value);
    if (!total_error) {
      auto total = internal::Uint64(total_value, "/metadata/total_size");
      if (!total.ok()) return total.status();
      result.declared_total_size = *total;
    } else if (total_error != simdjson::NO_SUCH_FIELD) {
      return internal::JsonError("/metadata", simdjson::error_message(total_error));
    }
  } else if (metadata_error != simdjson::NO_SUCH_FIELD) {
    return internal::JsonError("/metadata", simdjson::error_message(metadata_error));
  }
  if (auto status = file.CheckUnchanged(); !status.ok()) return status;
  return result;
}

}  // namespace inferx::artifacts
