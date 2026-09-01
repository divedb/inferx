#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "inferx/artifacts/model_resolver.h"

namespace inferx::artifacts::internal {

struct HttpRequest {
  std::string url;
  std::vector<std::pair<std::string, std::string>> headers;
  std::chrono::milliseconds timeout{30'000};
  std::optional<std::string> proxy;
  std::optional<std::filesystem::path> sink;
  size_t max_body_bytes = 16U * 1024U * 1024U;
};

struct HttpResponse {
  int status_code = 0;
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;

  [[nodiscard]] std::optional<std::string> Header(std::string_view name) const;
};

class HttpClient {
 public:
  virtual ~HttpClient() = default;
  virtual absl::StatusOr<HttpResponse> Get(const HttpRequest& request) = 0;
};

HttpClient& DefaultHttpClient();

absl::StatusOr<ResolvedModel> ResolveModelWithClient(std::string_view model,
                                                     const ModelResolverOptions& options,
                                                     HttpClient* client);

}  // namespace inferx::artifacts::internal
