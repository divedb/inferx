// Hub resolution, cache handling, retries, token lookup, and atomic staging
// retain the implementation structure of divedb/tokenizer at
// f109b7aef148dd4866a3dae7a8e5a6d221f95c75. The repository-metadata pass and
// model-weight selection are InferX-specific. See the retained MIT notice in
// third_party/notices/divedb-tokenizer-MIT.txt.

#include "inferx/artifacts/model_resolver.h"

#include <fcntl.h>
#include <simdjson.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/strip.h"
#include "inferx/artifacts/safe_relative_path.h"
#include "src/artifacts/hf_cache.h"
#include "src/artifacts/hf_http_client.h"
#include "src/artifacts/json_internal.h"

namespace inferx::artifacts {
namespace {

constexpr std::string_view kDefaultEndpoint = "https://huggingface.co";
constexpr std::string_view kResolutionReason = "type.inferx.dev/model-resolution-reason";

std::optional<std::string> GetEnvironment(std::string_view name) {
  const char* value = std::getenv(std::string(name).c_str());
  if (value == nullptr || *value == '\0') return std::nullopt;
  return std::string(value);
}

bool EnvironmentFlag(std::string_view name) {
  auto value = GetEnvironment(name);
  if (!value.has_value()) return false;
  absl::AsciiStrToUpper(&*value);
  return *value == "1" || *value == "ON" || *value == "YES" || *value == "TRUE";
}

bool LooksLikePath(std::string_view value) {
  if (value.empty()) return false;
  if (value.front() == '.' || value.front() == '/' || value.front() == '~') return true;
  if (value.size() >= 2 && value[1] == ':') return true;
  return value.find('\\') != std::string_view::npos;
}

bool IsValidRepoId(std::string_view value) {
  if (value.empty() || value.size() > 256 || value.front() == '/' || value.back() == '/') {
    return false;
  }
  size_t slash_count = 0;
  size_t component_size = 0;
  for (const char character : value) {
    if (character == '/') {
      if (component_size == 0 || ++slash_count > 1) return false;
      component_size = 0;
      continue;
    }
    const unsigned char byte = static_cast<unsigned char>(character);
    if (!absl::ascii_isalnum(byte) && character != '_' && character != '-' && character != '.') {
      return false;
    }
    ++component_size;
  }
  return component_size != 0;
}

absl::Status ValidateRevision(std::string_view revision) {
  if (revision.empty() || revision.size() > 256) {
    return absl::InvalidArgumentError("model revision must contain between 1 and 256 bytes");
  }
  auto path = SafeRelativePath::Parse(revision);
  if (!path.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("invalid model revision: ", path.status().message()));
  }
  return absl::OkStatus();
}

std::vector<std::filesystem::path> LocalModelDirectories(const ModelResolverOptions& options) {
  std::vector<std::filesystem::path> result = options.local_model_dirs;
  if (auto value = GetEnvironment("INFERX_MODEL_DIR")) {
    size_t begin = 0;
    while (begin <= value->size()) {
      const size_t end = value->find(':', begin);
      const std::string_view component(value->data() + begin,
                                       (end == std::string::npos ? value->size() : end) - begin);
      if (!component.empty()) result.emplace_back(component);
      if (end == std::string::npos) break;
      begin = end + 1;
    }
  }
  return result;
}

std::optional<std::filesystem::path> FindLocalModel(std::string_view model,
                                                    const ModelResolverOptions& options) {
  std::error_code error;
  const std::filesystem::path direct{std::string(model)};
  if (std::filesystem::is_directory(direct, error)) return direct;
  for (const auto& root : LocalModelDirectories(options)) {
    const std::filesystem::path candidate = root / direct;
    error.clear();
    if (std::filesystem::is_directory(candidate, error)) return candidate;
  }
  return std::nullopt;
}

std::string Endpoint(const ModelResolverOptions& options) {
  std::string endpoint = options.endpoint.value_or(
      GetEnvironment("HF_ENDPOINT").value_or(std::string(kDefaultEndpoint)));
  while (endpoint.size() > 1 && endpoint.back() == '/') endpoint.pop_back();
  return endpoint;
}

std::filesystem::path HuggingFaceHome() {
  if (auto value = GetEnvironment("HF_HOME")) return std::filesystem::path(*value);
  if (auto value = GetEnvironment("XDG_CACHE_HOME")) {
    return std::filesystem::path(*value) / "huggingface";
  }
  if (auto value = GetEnvironment("HOME")) {
    return std::filesystem::path(*value) / ".cache" / "huggingface";
  }
  return std::filesystem::path(".cache") / "huggingface";
}

std::optional<std::string> Token(const ModelResolverOptions& options) {
  if (options.token.has_value() && !options.token->empty()) return options.token;
  if (auto value = GetEnvironment("HF_TOKEN")) return value;
  if (auto value = GetEnvironment("HUGGING_FACE_HUB_TOKEN")) return value;

  const std::filesystem::path token_path =
      GetEnvironment("HF_TOKEN_PATH")
          .transform([](const std::string& value) { return std::filesystem::path(value); })
          .value_or(HuggingFaceHome() / "token");
  std::ifstream input(token_path, std::ios::binary);
  if (!input) return std::nullopt;
  std::string token;
  std::getline(input, token);
  token = std::string(absl::StripAsciiWhitespace(token));
  return token.empty() ? std::nullopt : std::optional<std::string>(std::move(token));
}

std::string PercentEncode(std::string_view value, bool preserve_slash) {
  constexpr char hex[] = "0123456789ABCDEF";
  std::string result;
  result.reserve(value.size());
  for (const char character : value) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (absl::ascii_isalnum(byte) || character == '-' || character == '_' || character == '.' ||
        character == '~' || (preserve_slash && character == '/')) {
      result.push_back(character);
    } else {
      result.push_back('%');
      result.push_back(hex[byte >> 4U]);
      result.push_back(hex[byte & 0x0FU]);
    }
  }
  return result;
}

internal::HttpRequest Request(std::string url, const ModelResolverOptions& options,
                              const std::optional<std::string>& token) {
  internal::HttpRequest request;
  request.url = std::move(url);
  request.timeout = options.timeout;
  request.proxy = options.proxy;
  if (token.has_value()) {
    request.headers.emplace_back("Authorization", absl::StrCat("Bearer ", *token));
  }
  return request;
}

bool IsRetryable(int status_code) { return status_code == 429 || status_code >= 500; }

bool IsRetryable(const absl::Status& status) {
  return absl::IsUnavailable(status) || absl::IsDeadlineExceeded(status);
}

void Backoff(int attempt, std::optional<int> retry_after_seconds) {
  if (retry_after_seconds.has_value()) {
    std::this_thread::sleep_for(std::chrono::seconds(*retry_after_seconds));
    return;
  }
  static thread_local std::mt19937 random(std::random_device{}());
  const int base_ms = 200 << std::min(attempt, 5);
  std::uniform_int_distribution<int> jitter(0, base_ms);
  std::this_thread::sleep_for(std::chrono::milliseconds(jitter(random)));
}

std::string ServerExplanation(const internal::HttpResponse& response) {
  if (auto message = response.Header("X-Error-Message"); message.has_value() && !message->empty()) {
    return absl::StrCat(": ", *message);
  }
  if (auto code = response.Header("X-Error-Code"); code.has_value() && !code->empty()) {
    return absl::StrCat(" (", *code, ")");
  }
  return {};
}

absl::Status ResponseStatus(const internal::HttpResponse& response, std::string_view repo_id,
                            std::string_view endpoint, bool token_sent) {
  const std::string explanation = ServerExplanation(response);
  if (response.status_code == 401 || response.status_code == 403) {
    return absl::PermissionDeniedError(absl::StrCat(
        "cannot read ", repo_id, " (HTTP ", response.status_code, ")", explanation,
        token_sent ? "; the token sent may not have access"
                   : "; set HF_TOKEN for a private or gated model, or check the model name"));
  }
  if (response.status_code == 404) {
    if (response.Header("X-Error-Code") == "RevisionNotFound") {
      return absl::NotFoundError(absl::StrCat("model revision was not found in ", repo_id));
    }
    return absl::NotFoundError(
        absl::StrCat("model ", repo_id, " was not found on ", endpoint, explanation));
  }
  if (response.status_code == 429) {
    return absl::ResourceExhaustedError(
        absl::StrCat("rate limited by ", endpoint, " while resolving ", repo_id, explanation));
  }
  if (response.status_code >= 500) {
    return absl::UnavailableError(absl::StrCat(endpoint, " returned HTTP ", response.status_code,
                                               " while resolving ", repo_id, explanation));
  }
  return absl::UnavailableError(absl::StrCat("unexpected HTTP ", response.status_code,
                                             " while resolving ", repo_id, explanation));
}

absl::StatusOr<internal::HttpResponse> GetWithRetries(internal::HttpClient* client,
                                                      const internal::HttpRequest& request,
                                                      const ModelResolverOptions& options,
                                                      std::string_view repo_id,
                                                      std::string_view endpoint, bool token_sent) {
  absl::Status last = absl::UnavailableError("no HTTP attempt was made");
  for (int attempt = 0; attempt <= options.max_retries; ++attempt) {
    auto response = client->Get(request);
    if (!response.ok()) {
      last = response.status();
      if (!IsRetryable(last)) break;
      if (attempt < options.max_retries) Backoff(attempt, std::nullopt);
      continue;
    }
    if (response->status_code == 200) {
      return std::move(*response);
    }
    last = ResponseStatus(*response, repo_id, endpoint, token_sent);
    if (!IsRetryable(response->status_code) || attempt == options.max_retries) break;
    std::optional<int> retry_after;
    if (auto value = response->Header("Retry-After")) {
      int seconds = 0;
      if (absl::SimpleAtoi(*value, &seconds) && seconds >= 0) {
        retry_after = std::min(seconds, 60);
      }
    }
    Backoff(attempt, retry_after);
  }
  return last;
}

struct HubModelMetadata {
  std::string sha;
  std::vector<std::string> files;
};

bool IsModelArtifact(std::string_view filename) {
  if (absl::StartsWith(filename, "original/") || absl::StartsWith(filename, ".")) return false;
  return absl::EndsWith(filename, ".json") || absl::EndsWith(filename, ".safetensors") ||
         absl::EndsWith(filename, ".model") || absl::EndsWith(filename, ".txt") ||
         absl::EndsWith(filename, ".tiktoken") || absl::EndsWith(filename, ".jinja");
}

absl::StatusOr<HubModelMetadata> ParseMetadata(std::string_view body) {
  simdjson::dom::parser parser;
  simdjson::dom::element document;
  const auto parse_error = parser.parse(body.data(), body.size()).get(document);
  if (parse_error) {
    return absl::DataLossError(absl::StrCat("invalid Hugging Face model metadata: ",
                                            simdjson::error_message(parse_error)));
  }
  auto root = internal::Object(document, "");
  if (!root.ok()) return root.status();
  auto sha_value = internal::Required(*root, "sha", "");
  if (!sha_value.ok()) return sha_value.status();
  auto sha = internal::String(*sha_value, "/sha");
  if (!sha.ok()) return sha.status();
  if (!internal::IsFullCommitSha(*sha)) {
    return absl::DataLossError("Hugging Face metadata did not contain a full commit SHA");
  }
  auto siblings_value = internal::Required(*root, "siblings", "");
  if (!siblings_value.ok()) return siblings_value.status();
  auto siblings = internal::Array(*siblings_value, "/siblings");
  if (!siblings.ok()) return siblings.status();
  if (siblings->size() > 100'000) {
    return absl::ResourceExhaustedError("Hugging Face model metadata lists too many files");
  }

  std::set<std::string> files;
  for (simdjson::dom::element element : *siblings) {
    auto sibling = internal::Object(element, "/siblings");
    if (!sibling.ok()) return sibling.status();
    auto filename_value = internal::Required(*sibling, "rfilename", "/siblings");
    if (!filename_value.ok()) return filename_value.status();
    auto filename = internal::String(*filename_value, "/siblings/rfilename");
    if (!filename.ok()) return filename.status();
    if (!IsModelArtifact(*filename)) continue;
    auto safe_path = SafeRelativePath::Parse(*filename);
    if (!safe_path.ok()) {
      return absl::DataLossError(
          absl::StrCat("unsafe filename in Hugging Face metadata: ", safe_path.status().message()));
    }
    files.emplace(*filename);
  }
  if (!files.contains("config.json")) {
    return absl::FailedPreconditionError("Hugging Face model has no config.json");
  }
  if (!files.contains("tokenizer.json")) {
    return absl::FailedPreconditionError(
        "Hugging Face model has no tokenizer.json; InferX does not load legacy-only tokenizers");
  }
  const bool has_weights =
      files.contains("model.safetensors") || files.contains("model.safetensors.index.json");
  if (!has_weights) {
    return absl::FailedPreconditionError(
        "Hugging Face model has no supported safetensors weights at repository root");
  }
  return HubModelMetadata{std::string(*sha), {files.begin(), files.end()}};
}

std::string Etag(const internal::HttpResponse& response) {
  auto value = response.Header("X-Linked-Etag");
  if (!value.has_value()) value = response.Header("ETag");
  if (!value.has_value()) return {};
  std::string_view stripped = absl::StripPrefix(*value, "W/");
  stripped = absl::StripPrefix(stripped, "\"");
  stripped = absl::StripSuffix(stripped, "\"");
  return std::string(stripped);
}

std::filesystem::path StagingFile(const std::filesystem::path& blobs, std::string_view filename) {
  static std::atomic<uint64_t> sequence{0};
  return blobs / absl::StrCat(".staging-", absl::StrReplaceAll(filename, {{"/", "_"}, {"\\", "_"}}),
                              "-", static_cast<uint64_t>(::getpid()), "-", sequence.fetch_add(1));
}

class DownloadLock {
 public:
  static absl::StatusOr<DownloadLock> Acquire(const std::filesystem::path& root,
                                              std::string_view repo_id) {
    const std::filesystem::path directory = root / ".locks";
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
      return absl::InternalError(
          absl::StrCat("cannot create Hugging Face lock directory: ", error.message()));
    }
    const std::filesystem::path path =
        directory / absl::StrCat(internal::HuggingFaceRepoDirName(repo_id), ".lock");
    const int descriptor = ::open(path.c_str(), O_CREAT | O_CLOEXEC | O_RDWR, 0600);
    if (descriptor < 0) return absl::InternalError("cannot open Hugging Face download lock");
    if (::flock(descriptor, LOCK_EX) != 0) {
      ::close(descriptor);
      return absl::InternalError("cannot acquire Hugging Face download lock");
    }
    return DownloadLock(descriptor);
  }

  DownloadLock(DownloadLock&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}
  DownloadLock& operator=(DownloadLock&&) = delete;
  DownloadLock(const DownloadLock&) = delete;
  DownloadLock& operator=(const DownloadLock&) = delete;
  ~DownloadLock() {
    if (descriptor_ >= 0) {
      ::flock(descriptor_, LOCK_UN);
      ::close(descriptor_);
    }
  }

 private:
  explicit DownloadLock(int descriptor) : descriptor_(descriptor) {}
  int descriptor_ = -1;
};

absl::StatusOr<ResolvedModel> ResolveCached(internal::HuggingFaceCache* cache,
                                            std::string_view model, std::string_view revision) {
  std::optional<std::string> sha;
  if (internal::IsFullCommitSha(revision)) {
    sha = std::string(revision);
  } else {
    sha = cache->ReadRef(revision);
  }
  if (!sha.has_value() || !internal::SnapshotIsUsable(cache->SnapshotDir(*sha))) {
    return absl::NotFoundError("model is not in the Hugging Face cache");
  }
  auto path = cache->Materialize(*sha);
  if (!path.ok()) return path.status();
  return ResolvedModel{std::move(*path), ModelSource::kHuggingFaceCache, std::string(model), *sha};
}

}  // namespace

std::string_view ModelSourceName(ModelSource source) noexcept {
  switch (source) {
    case ModelSource::kLocalDirectory:
      return "local";
    case ModelSource::kHuggingFaceCache:
      return "huggingface-cache";
    case ModelSource::kHuggingFaceDownload:
      return "huggingface-download";
  }
  return "unknown";
}

absl::StatusOr<ResolvedModel> internal::ResolveModelWithClient(std::string_view model,
                                                               const ModelResolverOptions& options,
                                                               HttpClient* client) {
  if (model.empty()) {
    return absl::InvalidArgumentError(
        "no model was named; pass a local directory or Hugging Face model ID");
  }
  if (options.max_retries < 0 || options.max_retries > 10) {
    return absl::InvalidArgumentError("model resolver max_retries must be between 0 and 10");
  }
  if (options.timeout <= std::chrono::milliseconds::zero()) {
    return absl::InvalidArgumentError("model resolver timeout must be positive");
  }
  if (auto local = FindLocalModel(model, options)) {
    return ResolvedModel{std::move(*local), ModelSource::kLocalDirectory, std::string(model), {}};
  }
  if (LooksLikePath(model)) {
    return absl::NotFoundError(
        absl::StrCat("\"", model, "\" looks like a local path but is not a directory"));
  }
  if (!IsValidRepoId(model)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "\"", model,
        "\" is neither a local directory nor a valid Hugging Face model ID (name or org/name)"));
  }
  if (auto status = ValidateRevision(options.revision); !status.ok()) return status;

  internal::HuggingFaceCache cache(internal::HuggingFaceCache::ResolveRoot(options.download_dir),
                                   std::string(model));
  auto cached = ResolveCached(&cache, model, options.revision);
  if (cached.ok()) return cached;
  if (!absl::IsNotFound(cached.status())) return cached.status();
  if (options.local_files_only || EnvironmentFlag("HF_HUB_OFFLINE")) {
    absl::Status status = absl::FailedPreconditionError(
        absl::StrCat("\"", model, "\" is not complete in the local Hugging Face cache \"",
                     cache.repo_dir().string(), "\" and offline operation was requested"));
    status.SetPayload(kResolutionReason, absl::Cord("OFFLINE_MISS"));
    return status;
  }

  auto lock = DownloadLock::Acquire(cache.root(), model);
  if (!lock.ok()) return lock.status();
  cached = ResolveCached(&cache, model, options.revision);
  if (cached.ok()) return cached;
  if (!absl::IsNotFound(cached.status())) return cached.status();

  HttpClient* transport = client != nullptr ? client : &DefaultHttpClient();
  const std::string endpoint = Endpoint(options);
  const std::optional<std::string> token = Token(options);
  const std::string metadata_url =
      absl::StrCat(endpoint, "/api/models/", PercentEncode(model, true), "/revision/",
                   PercentEncode(options.revision, false));
  auto metadata_response = GetWithRetries(transport, Request(metadata_url, options, token), options,
                                          model, endpoint, token.has_value());
  if (!metadata_response.ok()) return metadata_response.status();
  auto metadata = ParseMetadata(metadata_response->body);
  if (!metadata.ok()) return metadata.status();

  auto blobs = cache.EnsureBlobsDir();
  if (!blobs.ok()) return blobs.status();
  for (const std::string& filename : metadata->files) {
    const std::filesystem::path cached_file = cache.SnapshotDir(metadata->sha) / filename;
    std::error_code error;
    if (std::filesystem::is_regular_file(cached_file, error)) continue;

    const std::filesystem::path staged = StagingFile(*blobs, filename);
    internal::HttpRequest request =
        Request(absl::StrCat(endpoint, "/", PercentEncode(model, true), "/resolve/", metadata->sha,
                             "/", PercentEncode(filename, true)),
                options, token);
    request.sink = staged;
    auto response = GetWithRetries(transport, request, options, model, endpoint, token.has_value());
    if (!response.ok()) {
      std::filesystem::remove(staged, error);
      return response.status();
    }
    if (auto commit = response->Header("X-Repo-Commit");
        commit.has_value() && *commit != metadata->sha) {
      std::filesystem::remove(staged, error);
      return absl::DataLossError("Hugging Face artifact commit does not match model metadata");
    }
    const std::string etag = Etag(*response);
    if (etag.empty()) {
      std::filesystem::remove(staged, error);
      return absl::DataLossError("Hugging Face artifact response has no ETag");
    }
    if (auto status = cache.StoreFile(metadata->sha, filename, etag, staged); !status.ok()) {
      std::filesystem::remove(staged, error);
      return status;
    }
  }
  if (auto status = cache.WriteRef(options.revision, metadata->sha); !status.ok()) return status;
  auto path = cache.Materialize(metadata->sha);
  if (!path.ok()) return path.status();
  return ResolvedModel{std::move(*path), ModelSource::kHuggingFaceDownload, std::string(model),
                       metadata->sha};
}

absl::StatusOr<ResolvedModel> ModelResolver::Resolve(std::string_view model,
                                                     const ModelResolverOptions& options) const {
  return internal::ResolveModelWithClient(model, options, nullptr);
}

}  // namespace inferx::artifacts
