// Transactional validated-package load.

#include "inferx/input/model_package.h"

#include <algorithm>
#include <bit>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/artifacts/artifact_file.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "inferx/model/model_spec.h"
#include "inferx/tokenization/tokenizer_metadata.h"

namespace inferx::input {
namespace {

absl::StatusOr<std::string> ReadArtifact(const artifacts::ArtifactSession& session,
                                         const artifacts::SafeRelativePath& path, uint64_t limit) {
  auto file = session.OpenRegular(path);
  if (!file.ok()) return file.status();
  auto bytes = file->ReadAll(limit);
  if (!bytes.ok()) return bytes.status();
  auto unchanged = file->CheckUnchanged();
  if (!unchanged.ok()) return unchanged;
  return std::string(reinterpret_cast<const char*>(bytes->data()), bytes->size());
}

absl::StatusOr<artifacts::SafeRelativePath> ParsePath(std::string_view name) {
  return artifacts::SafeRelativePath::Parse(name);
}

}  // namespace

absl::Status CrossCheckTokenizerModel(const tokenization::TokenizerMetadata& metadata,
                                      const model::ModelSpec& spec) {
  const auto& llama = spec.llama();

  // Every id the tokenizer can emit must fit the model vocabulary.
  if (metadata.max_token_id >= 0 &&
      static_cast<uint64_t>(metadata.max_token_id) >= llama.vocab_size) {
    return absl::FailedPreconditionError(absl::StrCat("tokenizer emits id ", metadata.max_token_id,
                                                      " but the model vocabulary has ",
                                                      llama.vocab_size, " rows"));
  }

  // Config/tokenizer special-token agreement: disagreement is
  // FailedPrecondition, never a warning.
  if (llama.bos_token_id.has_value() && metadata.bos_id.has_value() &&
      static_cast<int32_t>(*llama.bos_token_id) != metadata.bos_id->value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("config.json declares bos_token_id ", *llama.bos_token_id,
                     " but the tokenizer resolves its BOS token to ", metadata.bos_id->value()));
  }
  if (llama.pad_token_id.has_value() && metadata.pad_id.has_value() &&
      static_cast<int32_t>(*llama.pad_token_id) != metadata.pad_id->value()) {
    return absl::FailedPreconditionError(
        absl::StrCat("config.json declares pad_token_id ", *llama.pad_token_id,
                     " but the tokenizer resolves its PAD token to ", metadata.pad_id->value()));
  }
  if (metadata.eos_id.has_value()) {
    for (int32_t config_eos : llama.eos_token_ids) {
      if (config_eos == metadata.eos_id->value()) {
        return absl::OkStatus();  // at least one stop id agrees
      }
    }
    // The config EOS list may carry additional valid stop ids;
    // a tokenizer EOS entirely absent from it is a mismatch.
    if (!llama.eos_token_ids.empty()) {
      return absl::FailedPreconditionError(
          absl::StrCat("the tokenizer resolves its EOS token to ", metadata.eos_id->value(),
                       " which is not among the model's declared stop ids"));
    }
  }
  for (int32_t special : metadata.special_ids) {
    if (special < 0 || static_cast<uint64_t>(special) >= llama.vocab_size) {
      return absl::FailedPreconditionError(
          absl::StrCat("special token id ", special, " is outside the model vocabulary"));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ValidatedModelPackage> ModelPackageLoader::Load(
    const std::string& root, const PackageTokenizerPolicy& policy,
    const artifacts::ArtifactLimits& limits) {
  // Step 1: the artifact/model half, fully validated and hashed.
  model::ModelArtifactLoader loader;
  auto artifacts = loader.Inspect(std::filesystem::path(root), limits);
  if (!artifacts.ok()) return artifacts.status();

  // Step 2: tokenizer artifacts through the same rooted session contract.
  auto session = artifacts::ModelLocator::OpenLocal(std::filesystem::path(root), limits);
  if (!session.ok()) return session.status();

  tokenization::TokenizerArtifacts tokenizer_artifacts;
  auto tokenizer_json_path = ParsePath("tokenizer.json");
  if (!tokenizer_json_path.ok()) return tokenizer_json_path.status();
  auto tokenizer_json = ReadArtifact(*session, *tokenizer_json_path, limits.max_json_bytes);
  if (!tokenizer_json.ok()) return tokenizer_json.status();
  tokenizer_artifacts.tokenizer_json = std::move(*tokenizer_json);

  auto attach = [&](std::string_view name, std::optional<std::string>& slot) {
    auto path = ParsePath(name);
    if (!path.ok()) return path.status();
    auto exists = session->ExistsRegular(*path);
    if (!exists.ok()) return exists.status();
    if (!*exists) return absl::OkStatus();
    auto bytes = ReadArtifact(*session, *path, limits.max_json_bytes);
    if (!bytes.ok()) return bytes.status();
    slot = std::move(*bytes);
    return absl::OkStatus();
  };
  if (auto status = attach("tokenizer_config.json", tokenizer_artifacts.tokenizer_config);
      !status.ok()) {
    return status;
  }
  if (auto status = attach("special_tokens_map.json", tokenizer_artifacts.special_tokens_map);
      !status.ok()) {
    return status;
  }
  if (auto status = attach("chat_template.jinja", tokenizer_artifacts.chat_template_jinja);
      !status.ok()) {
    return status;
  }
  if (auto status = attach("config.json", tokenizer_artifacts.model_config); !status.ok()) {
    return status;
  }

  // Step 3: load and qualify the tokenizer.
  tokenization::TokenizerInstancePolicy instance_policy;
  instance_policy.instance_count = policy.instance_count;
  instance_policy.strict = policy.strict;
  auto tokenizer = tokenization::Tokenizer::Load(std::move(tokenizer_artifacts), instance_policy);
  if (!tokenizer.ok()) return tokenizer.status();
  auto shared = std::shared_ptr<const tokenization::Tokenizer>(std::move(*tokenizer));

  // Step 4: cross-check tokenizer against the model spec.
  if (auto status = CrossCheckTokenizerModel(shared->metadata(), artifacts->model_spec);
      !status.ok()) {
    return status;
  }

  // Step 5: fingerprint the complete package, including the tokenizer
  // capability record.
  artifacts::ModelFingerprintInput input;
  input.architecture = "llama";
  input.semantic_config = std::vector<std::byte>(artifacts->model_spec.canonical_bytes().begin(),
                                                 artifacts->model_spec.canonical_bytes().end());
  const std::string capability = tokenization::CanonicalMetadataRecord(shared->metadata());
  input.tokenizer_capability.resize(capability.size());
  std::transform(capability.begin(), capability.end(), input.tokenizer_capability.begin(),
                 [](char c) { return static_cast<std::byte>(c); });
  input.artifacts = std::move(artifacts->artifacts);
  input.model_revision = artifacts->model_revision;
  // RoPE policy record: versioned type/theta; only the default
  // (non-scaled) rotary only (ADR 0022: rope_scaling objects are
  // Unimplemented), so the record is type + IEEE-754 theta bits.
  const uint64_t theta_bits = std::bit_cast<uint64_t>(artifacts->model_spec.llama().rope_theta);
  const std::string rope_record =
      absl::StrCat("rope.v1:default:theta=0x", absl::Hex(theta_bits, absl::kZeroPad16));
  input.rope.resize(rope_record.size());
  std::transform(rope_record.begin(), rope_record.end(), input.rope.begin(),
                 [](char c) { return static_cast<std::byte>(c); });
  auto fingerprint = artifacts::ModelFingerprint::Build(std::move(input));
  if (!fingerprint.ok()) return fingerprint.status();

  return ValidatedModelPackage(
      std::move(*artifacts), std::make_unique<artifacts::ModelFingerprint>(std::move(*fingerprint)),
      std::move(shared));
}

}  // namespace inferx::input
