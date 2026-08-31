#include "inferx/artifacts/artifact_manifest.h"

#include <simdjson.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "json_internal.h"

namespace inferx::artifacts {

const ArtifactManifestEntry* ArtifactManifest::Find(const SafeRelativePath& path) const {
  const auto iterator =
      std::lower_bound(files.begin(), files.end(), path,
                       [](const ArtifactManifestEntry& entry, const SafeRelativePath& target) {
                         return entry.path < target;
                       });
  return iterator != files.end() && iterator->path == path ? &*iterator : nullptr;
}

absl::StatusOr<ArtifactManifest> ArtifactManifestReader::Read(const ArtifactFile& file,
                                                              const ArtifactLimits& limits) const {
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
  const std::set<std::string> expected = {"files", "model_revision", "schema_version",
                                          "weights_entry"};
  std::set<std::string> actual;
  for (const auto field : *root) actual.emplace(field.key);
  if (actual != expected) {
    return internal::JsonError("",
                               "manifest must contain exactly schema_version, model_revision, "
                               "weights_entry, and files");
  }

  auto schema_value = internal::Required(*root, "schema_version", "");
  if (!schema_value.ok()) return schema_value.status();
  auto schema = internal::Uint64(*schema_value, "/schema_version");
  if (!schema.ok()) return schema.status();
  if (*schema != 1) {
    return absl::UnimplementedError("unsupported InferX manifest schema");
  }
  auto revision_value = internal::Required(*root, "model_revision", "");
  if (!revision_value.ok()) return revision_value.status();
  auto revision = internal::String(*revision_value, "/model_revision");
  if (!revision.ok()) return revision.status();
  if (revision->empty() || revision->size() > 256) {
    return internal::JsonError("/model_revision", "must contain between 1 and 256 UTF-8 bytes");
  }
  auto weights_value = internal::Required(*root, "weights_entry", "");
  if (!weights_value.ok()) return weights_value.status();
  auto weights_text = internal::String(*weights_value, "/weights_entry");
  if (!weights_text.ok()) return weights_text.status();
  auto weights_entry = SafeRelativePath::Parse(*weights_text);
  if (!weights_entry.ok()) return weights_entry.status();

  auto files_value = internal::Required(*root, "files", "");
  if (!files_value.ok()) return files_value.status();
  auto files = internal::Array(*files_value, "/files");
  if (!files.ok()) return files.status();
  if (files->size() == 0 || files->size() > limits.max_tensors) {
    return absl::ResourceExhaustedError("manifest file count is empty or exceeds configured limit");
  }

  std::vector<ArtifactManifestEntry> entries;
  entries.reserve(files->size());
  size_t index = 0;
  for (simdjson::dom::element element : *files) {
    const std::string path = absl::StrCat("/files/", index);
    auto object = internal::Object(element, path);
    if (!object.ok()) return object.status();
    if (auto status = internal::CheckUniqueKeys(*object, path); !status.ok()) {
      return status;
    }
    const std::set<std::string> required = {"blake3", "path", "size"};
    std::set<std::string> fields;
    for (const auto field : *object) fields.emplace(field.key);
    if (fields != required) {
      return internal::JsonError(path, "file entry must contain exactly path, size, and blake3");
    }
    auto path_value = internal::Required(*object, "path", path);
    if (!path_value.ok()) return path_value.status();
    auto path_text = internal::String(*path_value, path + "/path");
    if (!path_text.ok()) return path_text.status();
    auto safe_path = SafeRelativePath::Parse(*path_text);
    if (!safe_path.ok()) return safe_path.status();
    auto size_value = internal::Required(*object, "size", path);
    if (!size_value.ok()) return size_value.status();
    auto size = internal::Uint64(*size_value, path + "/size");
    if (!size.ok()) return size.status();
    auto digest_value = internal::Required(*object, "blake3", path);
    if (!digest_value.ok()) return digest_value.status();
    auto digest_text = internal::String(*digest_value, path + "/blake3");
    if (!digest_text.ok()) return digest_text.status();
    auto digest = Digest256::ParseHex(*digest_text);
    if (!digest.ok()) return digest.status();
    entries.push_back({std::move(*safe_path), *size, *digest});
    ++index;
  }
  std::sort(entries.begin(), entries.end(),
            [](const ArtifactManifestEntry& lhs, const ArtifactManifestEntry& rhs) {
              return lhs.path < rhs.path;
            });
  for (size_t i = 1; i < entries.size(); ++i) {
    if (entries[i - 1].path == entries[i].path) {
      return internal::JsonError("/files", "duplicate manifest file path");
    }
  }
  if (auto status = file.CheckUnchanged(); !status.ok()) return status;
  return ArtifactManifest{1, std::move(*revision), std::move(*weights_entry), std::move(entries)};
}

absl::Status VerifyManifestEntry(const ArtifactManifestEntry& expected, const ArtifactFile& file,
                                 const Digest256& digest) {
  if (expected.path.string() != file.relative_path()) {
    return absl::InvalidArgumentError("manifest entry does not identify the opened artifact");
  }
  if (expected.size != file.identity().size) {
    return absl::DataLossError(absl::StrCat("manifest size mismatch for ", file.relative_path()));
  }
  if (!(expected.digest == digest)) {
    return absl::DataLossError(absl::StrCat("manifest digest mismatch for ", file.relative_path()));
  }
  return absl::OkStatus();
}

}  // namespace inferx::artifacts
