// Portions adapted from divedb/tokenizer at
// f109b7aef148dd4866a3dae7a8e5a6d221f95c75. See
// third_party/notices/divedb-tokenizer-MIT.txt.

#include "src/artifacts/hf_cache.h"

#include <simdjson.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <system_error>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/strip.h"
#include "inferx/artifacts/safe_relative_path.h"

namespace inferx::artifacts::internal {
namespace {

std::optional<std::string> GetEnvironment(std::string_view name) {
  const char* value = std::getenv(std::string(name).c_str());
  if (value == nullptr || *value == '\0') return std::nullopt;
  return std::string(value);
}

absl::Status IoError(std::string_view action, const std::filesystem::path& path,
                     const std::error_code& error) {
  return absl::InternalError(absl::StrCat(action, " \"", path.string(), "\": ", error.message()));
}

bool IsSafeBlobKey(std::string_view key) {
  if (key.empty() || key == "." || key == "..") return false;
  for (const char value : key) {
    const unsigned char character = static_cast<unsigned char>(value);
    if (!std::isalnum(character) && character != '-' && character != '_' && character != '.') {
      return false;
    }
  }
  return true;
}

bool IsWithin(const std::filesystem::path& child, const std::filesystem::path& parent) {
  const std::filesystem::path relative = child.lexically_relative(parent);
  if (relative.empty() || relative.is_absolute()) return false;
  const auto first = relative.begin();
  return first == relative.end() || *first != "..";
}

bool IndexedWeightsAreComplete(const std::filesystem::path& snapshot,
                               const std::filesystem::path& index) {
  constexpr uintmax_t kMaxIndexBytes = 16U * 1024U * 1024U;
  std::error_code error;
  const uintmax_t bytes = std::filesystem::file_size(index, error);
  if (error || bytes > kMaxIndexBytes) return false;

  simdjson::dom::parser parser;
  simdjson::dom::element document;
  if (parser.load(index.string()).get(document)) return false;
  simdjson::dom::object root;
  if (document.get_object().get(root)) return false;
  simdjson::dom::element weight_map_value;
  if (root["weight_map"].get(weight_map_value)) return false;
  simdjson::dom::object weight_map;
  if (weight_map_value.get_object().get(weight_map)) return false;

  bool found = false;
  for (const auto field : weight_map) {
    std::string_view filename;
    if (field.value.get_string().get(filename)) return false;
    auto path = SafeRelativePath::Parse(filename);
    if (!path.ok()) return false;
    error.clear();
    if (!std::filesystem::is_regular_file(snapshot / path->string(), error)) return false;
    found = true;
  }
  return found;
}

std::filesystem::path UniqueStagingPath(const std::filesystem::path& parent,
                                        std::string_view stem) {
  static std::atomic<uint64_t> sequence{0};
  return parent / absl::StrCat(".staging-", stem, "-", static_cast<uint64_t>(::getpid()), "-",
                               sequence.fetch_add(1));
}

class StagingDirectory {
 public:
  explicit StagingDirectory(std::filesystem::path path) : path_(std::move(path)) {}
  ~StagingDirectory() {
    if (keep_) return;
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  void Keep() noexcept { keep_ = true; }

 private:
  std::filesystem::path path_;
  bool keep_ = false;
};

}  // namespace

std::string HuggingFaceRepoDirName(std::string_view repo_id) {
  return absl::StrCat("models--", absl::StrReplaceAll(repo_id, {{"/", "--"}}));
}

bool IsFullCommitSha(std::string_view revision) {
  if (revision.size() != 40) return false;
  for (const char value : revision) {
    const unsigned char character = static_cast<unsigned char>(value);
    if (!absl::ascii_isxdigit(character)) return false;
  }
  return true;
}

bool SnapshotIsUsable(const std::filesystem::path& snapshot) {
  std::error_code error;
  if (!std::filesystem::is_directory(snapshot, error)) return false;
  const bool has_config = std::filesystem::is_regular_file(snapshot / "config.json", error);
  error.clear();
  const bool has_tokenizer = std::filesystem::is_regular_file(snapshot / "tokenizer.json", error);
  error.clear();
  if (std::filesystem::is_regular_file(snapshot / "model.safetensors", error)) {
    return has_config && has_tokenizer;
  }
  error.clear();
  const std::filesystem::path index = snapshot / "model.safetensors.index.json";
  const bool has_weights =
      std::filesystem::is_regular_file(index, error) && IndexedWeightsAreComplete(snapshot, index);
  return has_config && has_tokenizer && has_weights;
}

std::filesystem::path HuggingFaceCache::ResolveRoot(
    const std::optional<std::filesystem::path>& override_root) {
  if (override_root.has_value()) return *override_root;
  if (auto value = GetEnvironment("HF_HUB_CACHE")) return std::filesystem::path(*value);
  if (auto value = GetEnvironment("HF_HOME")) return std::filesystem::path(*value) / "hub";
  if (auto value = GetEnvironment("XDG_CACHE_HOME")) {
    return std::filesystem::path(*value) / "huggingface" / "hub";
  }
  if (auto value = GetEnvironment("HOME")) {
    return std::filesystem::path(*value) / ".cache" / "huggingface" / "hub";
  }
  return std::filesystem::path(".cache") / "huggingface" / "hub";
}

HuggingFaceCache::HuggingFaceCache(std::filesystem::path root, std::string repo_id)
    : root_(std::move(root)),
      repo_id_(std::move(repo_id)),
      repo_dir_(root_ / HuggingFaceRepoDirName(repo_id_)) {}

std::filesystem::path HuggingFaceCache::SnapshotDir(std::string_view revision) const {
  return repo_dir_ / "snapshots" / std::string(revision);
}

std::filesystem::path HuggingFaceCache::BlobPath(std::string_view etag) const {
  return repo_dir_ / "blobs" / std::string(etag);
}

std::filesystem::path HuggingFaceCache::RefPath(std::string_view revision) const {
  return repo_dir_ / "refs" / std::string(revision);
}

std::filesystem::path HuggingFaceCache::InferxSnapshotDir(std::string_view revision) const {
  return repo_dir_ / "inferx" / "snapshots" / std::string(revision);
}

std::optional<std::string> HuggingFaceCache::ReadRef(std::string_view revision) const {
  std::ifstream input(RefPath(revision), std::ios::binary);
  if (!input) return std::nullopt;
  std::string sha;
  std::getline(input, sha);
  sha = std::string(absl::StripAsciiWhitespace(sha));
  if (!IsFullCommitSha(sha)) return std::nullopt;
  return sha;
}

absl::Status HuggingFaceCache::WriteRef(std::string_view revision, std::string_view sha) const {
  if (!IsFullCommitSha(sha)) {
    return absl::InvalidArgumentError("Hugging Face cache ref requires a full commit SHA");
  }
  const std::filesystem::path target = RefPath(revision);
  std::error_code error;
  std::filesystem::create_directories(target.parent_path(), error);
  if (error) return IoError("cannot create cache refs directory", target.parent_path(), error);

  const std::filesystem::path staged = UniqueStagingPath(target.parent_path(), "ref");
  {
    std::ofstream output(staged, std::ios::binary | std::ios::trunc);
    if (!output) return absl::InternalError("cannot stage Hugging Face cache ref");
    output.write(sha.data(), static_cast<std::streamsize>(sha.size()));
    if (!output) return absl::InternalError("cannot write Hugging Face cache ref");
  }
  std::filesystem::rename(staged, target, error);
  if (error) {
    std::filesystem::remove(staged);
    return IoError("cannot publish Hugging Face cache ref", target, error);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::filesystem::path> HuggingFaceCache::EnsureBlobsDir() const {
  const std::filesystem::path blobs = repo_dir_ / "blobs";
  std::error_code error;
  std::filesystem::create_directories(blobs, error);
  if (error) return IoError("cannot create Hugging Face blobs directory", blobs, error);
  return blobs;
}

absl::Status HuggingFaceCache::StoreFile(std::string_view revision, std::string_view filename,
                                         std::string_view etag,
                                         const std::filesystem::path& staged_file) const {
  auto safe_path = SafeRelativePath::Parse(filename);
  if (!safe_path.ok()) return safe_path.status();
  if (!IsSafeBlobKey(etag)) {
    return absl::DataLossError("Hugging Face returned an unsafe artifact ETag");
  }

  const std::filesystem::path blob = BlobPath(etag);
  std::error_code error;
  std::filesystem::create_directories(blob.parent_path(), error);
  if (error)
    return IoError("cannot create Hugging Face blobs directory", blob.parent_path(), error);

  if (std::filesystem::exists(blob, error)) {
    std::filesystem::remove(staged_file, error);
  } else {
    error.clear();
    std::filesystem::rename(staged_file, blob, error);
    if (error) {
      error.clear();
      std::filesystem::copy_file(staged_file, blob,
                                 std::filesystem::copy_options::overwrite_existing, error);
      std::filesystem::remove(staged_file);
      if (error) return IoError("cannot store downloaded Hugging Face blob", blob, error);
    }
  }

  const std::filesystem::path target = SnapshotDir(revision) / safe_path->string();
  std::filesystem::create_directories(target.parent_path(), error);
  if (error)
    return IoError("cannot create Hugging Face snapshot directory", target.parent_path(), error);
  std::filesystem::remove(target, error);
  error.clear();

  const std::filesystem::path relative_blob = blob.lexically_relative(target.parent_path());
  std::filesystem::create_symlink(relative_blob, target, error);
  if (!error) return absl::OkStatus();

  error.clear();
  std::filesystem::create_hard_link(blob, target, error);
  if (!error) return absl::OkStatus();

  error.clear();
  std::filesystem::copy_file(blob, target, std::filesystem::copy_options::overwrite_existing,
                             error);
  if (error) return IoError("cannot place artifact in Hugging Face snapshot", target, error);
  return absl::OkStatus();
}

absl::StatusOr<std::filesystem::path> HuggingFaceCache::Materialize(
    std::string_view revision) const {
  const std::filesystem::path source = SnapshotDir(revision);
  const std::filesystem::path destination = InferxSnapshotDir(revision);
  if (SnapshotIsUsable(destination)) return destination;
  if (!SnapshotIsUsable(source)) {
    return absl::FailedPreconditionError(
        absl::StrCat("cached snapshot is incomplete: \"", source.string(), "\""));
  }

  std::error_code error;
  if (std::filesystem::exists(destination, error)) {
    return absl::DataLossError(
        absl::StrCat("cached InferX snapshot is incomplete: \"", destination.string(), "\""));
  }
  const std::filesystem::path parent = destination.parent_path();
  std::filesystem::create_directories(parent, error);
  if (error) return IoError("cannot create InferX snapshot directory", parent, error);

  const std::filesystem::path staged = UniqueStagingPath(parent, "snapshot");
  std::filesystem::create_directories(staged, error);
  if (error) return IoError("cannot stage InferX snapshot", staged, error);
  StagingDirectory cleanup(staged);

  const std::filesystem::path canonical_repo = std::filesystem::weakly_canonical(repo_dir_, error);
  if (error) return IoError("cannot canonicalize Hugging Face repository cache", repo_dir_, error);

  std::filesystem::recursive_directory_iterator iterator(source, error);
  const std::filesystem::recursive_directory_iterator end;
  while (!error && iterator != end) {
    const std::filesystem::directory_entry entry = *iterator;
    const std::filesystem::path relative = entry.path().lexically_relative(source);
    auto safe_path = SafeRelativePath::Parse(relative.generic_string());
    if (!safe_path.ok()) return safe_path.status();
    const std::filesystem::path target = staged / safe_path->string();

    const std::filesystem::file_status status = entry.symlink_status(error);
    if (error) break;
    if (std::filesystem::is_directory(status)) {
      std::filesystem::create_directories(target, error);
      if (error) break;
      ++iterator;
      continue;
    }
    if (!std::filesystem::is_regular_file(status) && !std::filesystem::is_symlink(status)) {
      return absl::DataLossError(absl::StrCat("unsupported entry in Hugging Face snapshot: \"",
                                              entry.path().string(), "\""));
    }

    const std::filesystem::path canonical_source = std::filesystem::canonical(entry.path(), error);
    if (error) break;
    if (!IsWithin(canonical_source, canonical_repo)) {
      return absl::PermissionDeniedError(
          absl::StrCat("Hugging Face snapshot entry escapes its repository cache: \"",
                       entry.path().string(), "\""));
    }
    std::filesystem::create_directories(target.parent_path(), error);
    if (error) break;
    std::filesystem::create_hard_link(canonical_source, target, error);
    if (error) {
      error.clear();
      std::filesystem::copy_file(canonical_source, target, error);
      if (error) break;
    }
    ++iterator;
  }
  if (error) return IoError("cannot materialize Hugging Face snapshot", source, error);
  if (!SnapshotIsUsable(staged)) {
    return absl::DataLossError("materialized Hugging Face snapshot is incomplete");
  }

  std::filesystem::rename(staged, destination, error);
  if (error) {
    if (SnapshotIsUsable(destination)) return destination;
    return IoError("cannot publish InferX snapshot", destination, error);
  }
  cleanup.Keep();
  return destination;
}

}  // namespace inferx::artifacts::internal
