// The validated model package: the artifact/model
// half plus the qualified tokenizer, cross-checked and fingerprinted. Lives
// in the input layer because it ties model + tokenization together; the
// tokenization target itself never depends on model weights.

#ifndef INFERX_INPUT_MODEL_PACKAGE_H_
#define INFERX_INPUT_MODEL_PACKAGE_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "inferx/artifacts/artifact_limits.h"
#include "inferx/artifacts/model_fingerprint.h"
#include "inferx/artifacts/model_locator.h"
#include "inferx/model/model_package.h"
#include "inferx/model/model_spec.h"
#include "inferx/tokenization/tokenizer.h"
#include "inferx/tokenization/tokenizer_metadata.h"

namespace inferx::input {

struct PackageTokenizerPolicy {
  // Exclusive engine instances the package's facade keeps (also the pool's
  // maximum worker count).
  uint32_t instance_count = 1;
  // Fail load when the tokenizer checkpoint produced cross-check warnings.
  bool strict = false;
};

class ValidatedModelPackage {
 public:
  const model::ModelSpec& model_spec() const { return artifacts_.model_spec; }
  const model::WeightPlan& weight_plan() const { return artifacts_.weight_plan; }
  const model::ParameterCatalog& parameters() const { return artifacts_.parameters; }
  const artifacts::ModelFingerprint& fingerprint() const { return *fingerprint_; }
  const tokenization::Tokenizer& tokenizer() const { return *tokenizer_; }
  const tokenization::TokenizerMetadata& tokenizer_metadata() const {
    return tokenizer_->metadata();
  }
  const std::vector<artifacts::FingerprintedArtifact>& artifacts() const {
    return artifacts_.artifacts;
  }
  const artifacts::SafeRelativePath& weights_entry() const { return artifacts_.weights_entry; }
  bool integrity_manifest() const { return artifacts_.integrity_manifest; }
  const std::string& model_revision() const { return artifacts_.model_revision; }

 private:
  friend class ModelPackageLoader;
  ValidatedModelPackage(model::InspectedModelArtifacts artifacts,
                        std::unique_ptr<artifacts::ModelFingerprint> fingerprint,
                        std::shared_ptr<const tokenization::Tokenizer> tokenizer)
      : artifacts_(std::move(artifacts)),
        fingerprint_(std::move(fingerprint)),
        tokenizer_(std::move(tokenizer)) {}

  model::InspectedModelArtifacts artifacts_;
  std::unique_ptr<artifacts::ModelFingerprint> fingerprint_;
  std::shared_ptr<const tokenization::Tokenizer> tokenizer_;
};

class ModelPackageLoader {
 public:
  // Transactional load: inspect artifacts, read the
  // tokenizer artifacts through the rooted session, load and qualify the
  // tokenizer, cross-check it against the model spec, and fingerprint the
  // complete package. Nothing is published on failure.
  static absl::StatusOr<ValidatedModelPackage> Load(const std::string& root,
                                                    const PackageTokenizerPolicy& policy = {},
                                                    const artifacts::ArtifactLimits& limits = {});
};

// Model/tokenizer cross-check. Disagreement is
// FailedPrecondition, never a warning.
absl::Status CrossCheckTokenizerModel(const tokenization::TokenizerMetadata& metadata,
                                      const model::ModelSpec& spec);

}  // namespace inferx::input

#endif  // INFERX_INPUT_MODEL_PACKAGE_H_
