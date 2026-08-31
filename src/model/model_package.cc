#include "inferx/model/model_package.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/artifacts/artifact_manifest.h"
#include "inferx/artifacts/digest.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/artifacts/safetensors_index.h"
#include "inferx/artifacts/safetensors_reader.h"
#include "inferx/model/llama_model_factory.h"
#include "inferx/model/weight_planner.h"

namespace inferx::model {
namespace {

absl::StatusOr<artifacts::SafeRelativePath> Path(std::string_view text) {
  return artifacts::SafeRelativePath::Parse(text);
}

struct ConsumedFile {
  artifacts::SafeRelativePath path;
  artifacts::FileIdentity identity;
  artifacts::Digest256 digest;
};

absl::StatusOr<ConsumedFile> ConsumeFile(
    const artifacts::ArtifactSession& session, const artifacts::SafeRelativePath& path,
    const std::optional<artifacts::ArtifactManifest>& manifest) {
  auto file = session.OpenRegular(path);
  if (!file.ok()) return file.status();
  auto digest = artifacts::HashFile(*file);
  if (!digest.ok()) return digest.status();
  if (manifest.has_value()) {
    const auto* entry = manifest->Find(path);
    if (entry == nullptr) {
      return absl::DataLossError(
          absl::StrCat("consumed artifact missing from manifest: ", path.string()));
    }
    if (auto status = artifacts::VerifyManifestEntry(*entry, *file, *digest); !status.ok()) {
      return status;
    }
  }
  return ConsumedFile{path, file->identity(), *digest};
}

absl::StatusOr<std::vector<ExternalTensor>> ReadStandalone(
    const artifacts::ArtifactSession& session, const artifacts::SafeRelativePath& path,
    const ConsumedFile& consumed, const artifacts::ArtifactLimits& limits) {
  auto file = session.OpenRegular(path);
  if (!file.ok()) return file.status();
  if (!consumed.identity.SameFileAndVersion(file->identity())) {
    return absl::AbortedError("weight artifact changed between hash and parse");
  }
  artifacts::SafeTensorReader reader;
  auto metadata = reader.ReadHeader(*file, limits);
  if (!metadata.ok()) return metadata.status();
  std::vector<ExternalTensor> tensors;
  tensors.reserve(metadata->tensors.size());
  for (const auto& tensor : metadata->tensors) {
    std::optional<artifacts::Digest256> tensor_digest;
    if (tensor.name == "model.embed_tokens.weight" || tensor.name == "lm_head.weight") {
      auto digest =
          artifacts::HashFileRange(*file, tensor.absolute_file_offset, tensor.packed_size_bytes);
      if (!digest.ok()) return digest.status();
      tensor_digest = *digest;
    }
    tensors.push_back(
        {tensor.name, path, file->identity(), tensor, consumed.digest, tensor_digest});
  }
  return tensors;
}

absl::StatusOr<std::vector<ExternalTensor>> ReadSharded(
    const artifacts::ArtifactSession& session, const artifacts::SafeRelativePath& index_path,
    const ConsumedFile& consumed_index, const std::optional<artifacts::ArtifactManifest>& manifest,
    const artifacts::ArtifactLimits& limits, std::vector<ConsumedFile>* consumed) {
  auto index_file = session.OpenRegular(index_path);
  if (!index_file.ok()) return index_file.status();
  if (!consumed_index.identity.SameFileAndVersion(index_file->identity())) {
    return absl::AbortedError("shard index changed between hash and parse");
  }
  artifacts::SafetensorsIndexReader index_reader;
  auto index = index_reader.Read(*index_file, limits);
  if (!index.ok()) return index.status();

  std::map<std::string, artifacts::SafeRelativePath, std::less<>> weight_map;
  for (const auto& entry : index->entries) {
    weight_map.emplace(entry.tensor_name, entry.shard);
  }
  uint64_t total_payload = 0;
  std::vector<ExternalTensor> tensors;
  for (const auto& shard : index->shards) {
    auto consumed_shard = ConsumeFile(session, shard, manifest);
    if (!consumed_shard.ok()) return consumed_shard.status();
    auto file = session.OpenRegular(shard);
    if (!file.ok()) return file.status();
    if (!consumed_shard->identity.SameFileAndVersion(file->identity())) {
      return absl::AbortedError("shard changed between hash and parse");
    }
    artifacts::SafeTensorReader reader;
    auto metadata = reader.ReadHeader(*file, limits);
    if (!metadata.ok()) return metadata.status();
    bool referenced = false;
    for (const auto& tensor : metadata->tensors) {
      const auto mapping = weight_map.find(tensor.name);
      if (mapping == weight_map.end() || mapping->second != shard) {
        return absl::DataLossError(absl::StrCat(
            "shard tensor is absent or points elsewhere in weight_map: ", tensor.name));
      }
      referenced = true;
      if (tensor.packed_size_bytes > std::numeric_limits<uint64_t>::max() - total_payload) {
        return absl::OutOfRangeError("shard payload total overflows uint64");
      }
      total_payload += tensor.packed_size_bytes;
      std::optional<artifacts::Digest256> tensor_digest;
      if (tensor.name == "model.embed_tokens.weight" || tensor.name == "lm_head.weight") {
        auto digest =
            artifacts::HashFileRange(*file, tensor.absolute_file_offset, tensor.packed_size_bytes);
        if (!digest.ok()) return digest.status();
        tensor_digest = *digest;
      }
      tensors.push_back(
          {tensor.name, shard, file->identity(), tensor, consumed_shard->digest, tensor_digest});
    }
    if (!referenced) {
      return absl::DataLossError("shard contains no referenced tensor");
    }
    consumed->push_back(std::move(*consumed_shard));
  }
  for (const auto& entry : index->entries) {
    const bool found =
        std::any_of(tensors.begin(), tensors.end(), [&](const ExternalTensor& tensor) {
          return tensor.name == entry.tensor_name && tensor.shard == entry.shard;
        });
    if (!found) {
      return absl::DataLossError(
          absl::StrCat("weight_map tensor does not exist in named shard: ", entry.tensor_name));
    }
  }
  if (index->declared_total_size.has_value() && *index->declared_total_size != total_payload) {
    return absl::DataLossError(
        "shard-index metadata.total_size does not equal tensor payload bytes");
  }
  return tensors;
}

}  // namespace

absl::StatusOr<InspectedModelArtifacts> ModelArtifactLoader::Inspect(
    const std::filesystem::path& root, const artifacts::ArtifactLimits& limits) const {
  auto session = artifacts::ModelLocator::OpenLocal(root, limits);
  if (!session.ok()) return session.status();
  auto config_path = Path("config.json");
  auto tokenizer_path = Path("tokenizer.json");
  auto index_path = Path("model.safetensors.index.json");
  auto standalone_path = Path("model.safetensors");
  auto manifest_path = Path("inferx.manifest.json");
  if (!config_path.ok() || !tokenizer_path.ok() || !index_path.ok() || !standalone_path.ok() ||
      !manifest_path.ok()) {
    return absl::InternalError("fixed artifact path validation failed");
  }

  auto has_manifest = session->ExistsRegular(*manifest_path);
  if (!has_manifest.ok()) return has_manifest.status();
  std::optional<artifacts::ArtifactManifest> manifest;
  if (*has_manifest) {
    auto file = session->OpenRegular(*manifest_path);
    if (!file.ok()) return file.status();
    artifacts::ArtifactManifestReader reader;
    auto parsed = reader.Read(*file, limits);
    if (!parsed.ok()) return parsed.status();
    manifest = std::move(*parsed);
  }

  auto has_index = session->ExistsRegular(*index_path);
  if (!has_index.ok()) return has_index.status();
  auto has_standalone = session->ExistsRegular(*standalone_path);
  if (!has_standalone.ok()) return has_standalone.status();
  if (!*has_index && !*has_standalone) {
    return absl::NotFoundError("model root has neither safetensors index nor standalone weights");
  }
  artifacts::SafeRelativePath selected = *has_index ? *index_path : *standalone_path;
  if (*has_index && *has_standalone) {
    if (!manifest.has_value() || manifest->weights_entry != *index_path) {
      return absl::FailedPreconditionError(
          "both indexed and standalone weights exist; a manifest must "
          "explicitly select model.safetensors.index.json");
    }
    selected = *index_path;
  }
  if (manifest.has_value() && manifest->weights_entry != selected) {
    return absl::FailedPreconditionError(
        "manifest weights_entry disagrees with deterministic discovery");
  }

  std::vector<ConsumedFile> consumed;
  for (const auto& required : {*config_path, *tokenizer_path, selected}) {
    auto file = ConsumeFile(*session, required, manifest);
    if (!file.ok()) return file.status();
    if ((required == *config_path || required == *tokenizer_path) &&
        file->identity.size > limits.max_json_bytes) {
      return absl::ResourceExhaustedError(
          absl::StrCat(required.string(), " exceeds artifact.max_json_bytes"));
    }
    consumed.push_back(std::move(*file));
  }
  for (const std::string_view optional_name : {"tokenizer_config.json", "special_tokens_map.json",
                                               "generation_config.json", "chat_template.jinja"}) {
    auto optional_path = Path(optional_name);
    if (!optional_path.ok()) return optional_path.status();
    auto exists = session->ExistsRegular(*optional_path);
    if (!exists.ok()) return exists.status();
    if (*exists) {
      auto file = ConsumeFile(*session, *optional_path, manifest);
      if (!file.ok()) return file.status();
      consumed.push_back(std::move(*file));
    }
  }

  auto config_file = session->OpenRegular(*config_path);
  if (!config_file.ok()) return config_file.status();
  const auto consumed_config =
      std::find_if(consumed.begin(), consumed.end(),
                   [&](const ConsumedFile& file) { return file.path == *config_path; });
  if (consumed_config == consumed.end() ||
      !consumed_config->identity.SameFileAndVersion(config_file->identity())) {
    return absl::AbortedError("config changed between hash and parse");
  }
  LlamaModelFactory factory;
  auto model_spec = factory.ParseConfig(*config_file, limits);
  if (!model_spec.ok()) return model_spec.status();
  auto parameters = factory.BuildParameters(*model_spec);
  if (!parameters.ok()) return parameters.status();

  std::vector<ExternalTensor> external_tensors;
  if (selected == *index_path) {
    const auto consumed_index =
        std::find_if(consumed.begin(), consumed.end(),
                     [&](const ConsumedFile& file) { return file.path == selected; });
    if (consumed_index == consumed.end()) {
      return absl::InternalError("selected shard index was not consumed");
    }
    auto tensors = ReadSharded(*session, selected, *consumed_index, manifest, limits, &consumed);
    if (!tensors.ok()) return tensors.status();
    external_tensors = std::move(*tensors);
  } else {
    const auto selected_file =
        std::find_if(consumed.begin(), consumed.end(),
                     [&](const ConsumedFile& file) { return file.path == selected; });
    if (selected_file == consumed.end()) {
      return absl::InternalError("selected weight artifact was not consumed");
    }
    auto tensors = ReadStandalone(*session, selected, *selected_file, limits);
    if (!tensors.ok()) return tensors.status();
    external_tensors = std::move(*tensors);
  }
  auto catalog = ExternalTensorCatalog::Build(std::move(external_tensors));
  if (!catalog.ok()) return catalog.status();
  WeightPlanner planner;
  auto plan = planner.Build(*model_spec, *parameters, *catalog);
  if (!plan.ok()) return plan.status();

  std::sort(consumed.begin(), consumed.end(),
            [](const ConsumedFile& lhs, const ConsumedFile& rhs) { return lhs.path < rhs.path; });
  for (size_t i = 1; i < consumed.size(); ++i) {
    if (consumed[i - 1].path == consumed[i].path) {
      return absl::InternalError("artifact was consumed more than once");
    }
  }
  if (manifest.has_value() && manifest->files.size() != consumed.size()) {
    return absl::DataLossError(
        "manifest declares files that the selected package does not consume");
  }
  std::vector<artifacts::FingerprintedArtifact> fingerprinted;
  fingerprinted.reserve(consumed.size());
  for (const auto& file : consumed) {
    fingerprinted.push_back({file.path, file.identity.size, file.digest});
  }
  return InspectedModelArtifacts{std::move(*model_spec),
                                 std::move(*parameters),
                                 std::move(*plan),
                                 std::move(fingerprinted),
                                 selected,
                                 manifest.has_value(),
                                 manifest.has_value() ? manifest->model_revision : std::string()};
}

}  // namespace inferx::model
