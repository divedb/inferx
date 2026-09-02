#include "inferx/artifacts/model_resolver.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "src/artifacts/hf_http_client.h"

namespace inferx::artifacts {
namespace {

constexpr std::string_view kSha = "0123456789abcdef0123456789abcdef01234567";

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    constexpr std::string_view prefix = "/tmp/inferx-model-resolver-XXXXXX";
    std::array<char, prefix.size() + 1> pattern{};
    std::copy(prefix.begin(), prefix.end(), pattern.begin());
    path_ = ::mkdtemp(pattern.data());
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void Write(const std::filesystem::path& path, std::string_view content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output.good());
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  ASSERT_TRUE(output.good());
}

struct ScriptedResponse {
  int status_code = 200;
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
};

class FakeHttpClient final : public internal::HttpClient {
 public:
  void OnSuffix(std::string suffix, ScriptedResponse response) {
    responses_.emplace(std::move(suffix), std::move(response));
  }

  absl::StatusOr<internal::HttpResponse> Get(const internal::HttpRequest& request) override {
    urls_.push_back(request.url);
    const auto iterator = std::find_if(
        responses_.begin(), responses_.end(),
        [&](const auto& entry) { return std::string_view(request.url).ends_with(entry.first); });
    if (iterator == responses_.end()) {
      return absl::NotFoundError("fake transport has no response for URL");
    }
    const ScriptedResponse& scripted = iterator->second;
    if (request.sink.has_value() && scripted.status_code == 200) {
      Write(*request.sink, scripted.body);
    }
    return internal::HttpResponse{scripted.status_code, scripted.headers,
                                  request.sink.has_value() ? std::string() : scripted.body};
  }

  size_t request_count() const { return urls_.size(); }
  const std::vector<std::string>& urls() const { return urls_; }

 private:
  std::map<std::string, ScriptedResponse> responses_;
  std::vector<std::string> urls_;
};

std::string Metadata(std::string_view files) {
  return std::string("{\"sha\":\"") + std::string(kSha) + "\",\"siblings\":[" + std::string(files) +
         "]}";
}

ScriptedResponse Artifact(std::string body, std::string etag) {
  return {200,
          std::move(body),
          {{"ETag", absl::StrCat("\"", etag, "\"")}, {"X-Repo-Commit", std::string(kSha)}}};
}

TEST(ModelResolverTest, ExistingDirectoryWinsWithoutHttp) {
  TemporaryDirectory temporary;
  const auto model = temporary.path() / "local-model";
  std::filesystem::create_directories(model);
  FakeHttpClient client;

  auto resolved = internal::ResolveModelWithClient(model.string(), {}, &client);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_EQ(resolved->path, model);
  EXPECT_EQ(resolved->source, ModelSource::kLocalDirectory);
  EXPECT_EQ(client.request_count(), 0U);
}

TEST(ModelResolverTest, NamedLocalDirectoryWinsBeforeHubCache) {
  TemporaryDirectory temporary;
  const auto model = temporary.path() / "models" / "Qwen" / "Qwen2-test";
  std::filesystem::create_directories(model);
  ModelResolverOptions options;
  options.local_model_dirs.push_back(temporary.path() / "models");
  FakeHttpClient client;

  auto resolved = internal::ResolveModelWithClient("Qwen/Qwen2-test", options, &client);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_EQ(resolved->path, model);
  EXPECT_EQ(resolved->source, ModelSource::kLocalDirectory);
  EXPECT_EQ(client.request_count(), 0U);
}

TEST(ModelResolverTest, MissingExplicitPathDoesNotBecomeAHubRequest) {
  FakeHttpClient client;
  auto resolved = internal::ResolveModelWithClient("./missing-model", {}, &client);
  ASSERT_FALSE(resolved.ok());
  EXPECT_TRUE(absl::IsNotFound(resolved.status()));
  EXPECT_EQ(client.request_count(), 0U);
}

TEST(ModelResolverTest, ReadsReferenceCacheAndMaterializesSymlinkFreeView) {
  TemporaryDirectory temporary;
  const auto repo = temporary.path() / "models--Qwen--Qwen2-test";
  const auto blobs = repo / "blobs";
  const auto snapshot = repo / "snapshots" / std::string(kSha);
  Write(blobs / "config", "{}");
  Write(blobs / "tokenizer", "{}");
  Write(blobs / "weights", "weights");
  std::filesystem::create_directories(snapshot);
  std::filesystem::create_symlink("../../blobs/config", snapshot / "config.json");
  std::filesystem::create_symlink("../../blobs/tokenizer", snapshot / "tokenizer.json");
  std::filesystem::create_symlink("../../blobs/weights", snapshot / "model.safetensors");
  Write(repo / "refs" / "main", kSha);

  ModelResolverOptions options;
  options.download_dir = temporary.path();
  FakeHttpClient client;
  auto resolved = internal::ResolveModelWithClient("Qwen/Qwen2-test", options, &client);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_EQ(resolved->source, ModelSource::kHuggingFaceCache);
  EXPECT_EQ(resolved->revision, kSha);
  EXPECT_EQ(client.request_count(), 0U);
  EXPECT_TRUE(std::filesystem::is_regular_file(resolved->path / "config.json"));
  EXPECT_FALSE(std::filesystem::is_symlink(resolved->path / "config.json"));
}

TEST(ModelResolverTest, OfflineMissNeverTouchesTransport) {
  TemporaryDirectory temporary;
  ModelResolverOptions options;
  options.download_dir = temporary.path();
  options.local_files_only = true;
  FakeHttpClient client;

  auto resolved = internal::ResolveModelWithClient("Qwen/Qwen2-test", options, &client);
  ASSERT_FALSE(resolved.ok());
  EXPECT_TRUE(absl::IsFailedPrecondition(resolved.status()));
  EXPECT_EQ(client.request_count(), 0U);
  auto reason = resolved.status().GetPayload("type.inferx.dev/model-resolution-reason");
  ASSERT_TRUE(reason.has_value());
  EXPECT_EQ(std::string(*reason), "OFFLINE_MISS");
}

TEST(ModelResolverTest, CorruptCachedSnapshotIsNotHiddenAsAnOfflineMiss) {
  TemporaryDirectory temporary;
  const auto repo = temporary.path() / "models--Qwen--Qwen2-test";
  const auto snapshot = repo / "snapshots" / std::string(kSha);
  const auto outside = temporary.path() / "outside-config.json";
  Write(outside, "{}");
  Write(snapshot / "tokenizer.json", "{}");
  Write(snapshot / "model.safetensors", "weights");
  std::filesystem::create_symlink(outside, snapshot / "config.json");
  Write(repo / "refs" / "main", kSha);

  ModelResolverOptions options;
  options.download_dir = temporary.path();
  options.local_files_only = true;
  FakeHttpClient client;
  auto resolved = internal::ResolveModelWithClient("Qwen/Qwen2-test", options, &client);

  ASSERT_FALSE(resolved.ok());
  EXPECT_TRUE(absl::IsPermissionDenied(resolved.status())) << resolved.status();
  EXPECT_EQ(client.request_count(), 0U);
  EXPECT_FALSE(resolved.status().GetPayload("type.inferx.dev/model-resolution-reason").has_value());
}

TEST(ModelResolverTest, DownloadsSupportedModelFilesAndThenHitsCache) {
  TemporaryDirectory temporary;
  ModelResolverOptions options;
  options.download_dir = temporary.path();
  FakeHttpClient client;
  client.OnSuffix("/api/models/Qwen/Qwen2-test/revision/main",
                  {200,
                   Metadata("{\"rfilename\":\"config.json\"},"
                            "{\"rfilename\":\"tokenizer.json\"},"
                            "{\"rfilename\":\"model.safetensors\"},"
                            "{\"rfilename\":\"README.md\"},"
                            "{\"rfilename\":\"pytorch_model.bin\"},"
                            "{\"rfilename\":\"original/model.safetensors\"}"),
                   {}});
  client.OnSuffix("/resolve/0123456789abcdef0123456789abcdef01234567/config.json",
                  Artifact("{}", "config-etag"));
  client.OnSuffix("/resolve/0123456789abcdef0123456789abcdef01234567/tokenizer.json",
                  Artifact("{}", "tokenizer-etag"));
  client.OnSuffix("/resolve/0123456789abcdef0123456789abcdef01234567/model.safetensors",
                  Artifact("weights", "weights-etag"));

  auto resolved = internal::ResolveModelWithClient("Qwen/Qwen2-test", options, &client);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_EQ(resolved->source, ModelSource::kHuggingFaceDownload);
  EXPECT_EQ(resolved->revision, kSha);
  EXPECT_EQ(client.request_count(), 4U);
  EXPECT_TRUE(std::filesystem::is_regular_file(resolved->path / "model.safetensors"));
  EXPECT_FALSE(std::filesystem::exists(resolved->path / "pytorch_model.bin"));
  for (const auto& url : client.urls()) {
    EXPECT_EQ(url.find("pytorch_model.bin"), std::string::npos);
    EXPECT_EQ(url.find("original/model.safetensors"), std::string::npos);
  }

  FakeHttpClient second_client;
  auto cached = internal::ResolveModelWithClient("Qwen/Qwen2-test", options, &second_client);
  ASSERT_TRUE(cached.ok()) << cached.status();
  EXPECT_EQ(cached->source, ModelSource::kHuggingFaceCache);
  EXPECT_EQ(cached->path, resolved->path);
  EXPECT_EQ(second_client.request_count(), 0U);
}

TEST(ModelResolverTest, IncompleteIndexedCacheDownloadsMissingShard) {
  TemporaryDirectory temporary;
  const auto repo = temporary.path() / "models--Qwen--Qwen2-test";
  const auto snapshot = repo / "snapshots" / std::string(kSha);
  Write(snapshot / "config.json", "{}");
  Write(snapshot / "tokenizer.json", "{}");
  Write(
      snapshot / "model.safetensors.index.json",
      R"({"metadata":{"total_size":7},"weight_map":{"model.weight":"model-00001-of-00001.safetensors"}})");
  Write(repo / "refs" / "main", kSha);

  ModelResolverOptions options;
  options.download_dir = temporary.path();
  FakeHttpClient client;
  client.OnSuffix("/api/models/Qwen/Qwen2-test/revision/main",
                  {200,
                   Metadata("{\"rfilename\":\"config.json\"},"
                            "{\"rfilename\":\"tokenizer.json\"},"
                            "{\"rfilename\":\"model.safetensors.index.json\"},"
                            "{\"rfilename\":\"model-00001-of-00001.safetensors\"}"),
                   {}});
  client.OnSuffix(
      "/resolve/0123456789abcdef0123456789abcdef01234567/model-00001-of-00001.safetensors",
      Artifact("weights", "shard-etag"));

  auto resolved = internal::ResolveModelWithClient("Qwen/Qwen2-test", options, &client);
  ASSERT_TRUE(resolved.ok()) << resolved.status();
  EXPECT_EQ(resolved->source, ModelSource::kHuggingFaceDownload);
  EXPECT_EQ(client.request_count(), 2U);
  EXPECT_TRUE(
      std::filesystem::is_regular_file(resolved->path / "model-00001-of-00001.safetensors"));
}

TEST(ModelResolverTest, RejectsUnsafeRepositoryFilenamesBeforeDownload) {
  TemporaryDirectory temporary;
  ModelResolverOptions options;
  options.download_dir = temporary.path();
  FakeHttpClient client;
  client.OnSuffix("/api/models/org/model/revision/main",
                  {200,
                   Metadata("{\"rfilename\":\"config.json\"},"
                            "{\"rfilename\":\"tokenizer.json\"},"
                            "{\"rfilename\":\"model.safetensors\"},"
                            "{\"rfilename\":\"nested/../escape.json\"}"),
                   {}});

  auto resolved = internal::ResolveModelWithClient("org/model", options, &client);
  ASSERT_FALSE(resolved.ok());
  EXPECT_TRUE(absl::IsDataLoss(resolved.status()));
  EXPECT_EQ(client.request_count(), 1U);
}

TEST(ModelResolverTest, PreservesHubCasingInRequestsAndCacheNames) {
  TemporaryDirectory temporary;
  ModelResolverOptions options;
  options.download_dir = temporary.path();
  options.local_files_only = true;
  FakeHttpClient client;
  auto resolved = internal::ResolveModelWithClient("Qwen/Qwen2-Test", options, &client);
  ASSERT_FALSE(resolved.ok());
  EXPECT_NE(resolved.status().message().find("models--Qwen--Qwen2-Test"), std::string::npos);
}

}  // namespace
}  // namespace inferx::artifacts
