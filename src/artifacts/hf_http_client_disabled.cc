#include "absl/status/status.h"
#include "src/artifacts/hf_http_client.h"

namespace inferx::artifacts::internal {
namespace {

class DisabledHttpClient final : public HttpClient {
 public:
  absl::StatusOr<HttpResponse> Get(const HttpRequest&) override {
    return absl::UnimplementedError(
        "this InferX build has Hugging Face downloads disabled; rebuild with "
        "INFERX_ENABLE_HF_HUB=ON or use a local/cached model");
  }
};

}  // namespace

HttpClient& DefaultHttpClient() {
  static DisabledHttpClient client;
  return client;
}

}  // namespace inferx::artifacts::internal
