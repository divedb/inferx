// Adapted from divedb/tokenizer at
// f109b7aef148dd4866a3dae7a8e5a6d221f95c75. See
// third_party/notices/divedb-tokenizer-MIT.txt.

#include <curl/curl.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <mutex>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "src/artifacts/hf_http_client.h"

namespace inferx::artifacts::internal {
namespace {

std::once_flag curl_init_once;
CURLcode curl_init_result = CURLE_OK;

CURLcode EnsureCurlInitialized() {
  std::call_once(curl_init_once, [] { curl_init_result = curl_global_init(CURL_GLOBAL_DEFAULT); });
  return curl_init_result;
}

struct StringSink {
  std::string* output = nullptr;
  size_t limit = 0;
  bool exceeded = false;
};

size_t WriteToString(char* data, size_t size, size_t count, void* user_data) {
  auto* sink = static_cast<StringSink*>(user_data);
  if (size != 0 && count > std::numeric_limits<size_t>::max() / size) {
    sink->exceeded = true;
    return 0;
  }
  const size_t bytes = size * count;
  if (bytes > sink->limit - std::min(sink->output->size(), sink->limit)) {
    sink->exceeded = true;
    return 0;
  }
  sink->output->append(data, bytes);
  return bytes;
}

size_t WriteToFile(char* data, size_t size, size_t count, void* user_data) {
  auto* file = static_cast<std::FILE*>(user_data);
  return std::fwrite(data, size, count, file) * size;
}

size_t CollectHeader(char* data, size_t size, size_t count, void* user_data) {
  auto* headers = static_cast<std::vector<std::pair<std::string, std::string>>*>(user_data);
  const std::string_view line(data, size * count);
  const size_t colon = line.find(':');
  if (colon != std::string_view::npos) {
    std::string name(absl::StripAsciiWhitespace(line.substr(0, colon)));
    std::string value(absl::StripAsciiWhitespace(line.substr(colon + 1)));
    headers->emplace_back(std::move(name), std::move(value));
  }
  return size * count;
}

class CurlHttpClient final : public HttpClient {
 public:
  absl::StatusOr<HttpResponse> Get(const HttpRequest& request) override {
    if (EnsureCurlInitialized() != CURLE_OK) {
      return absl::InternalError("libcurl global initialization failed");
    }
    CURL* handle = ThreadHandle();
    if (handle == nullptr) return absl::InternalError("cannot create a libcurl handle");
    curl_easy_reset(handle);

    HttpResponse response;
    StringSink string_sink{&response.body, request.max_body_bytes, false};
    std::FILE* file = nullptr;
    curl_slist* headers = nullptr;
    struct Cleanup {
      CURL* handle;
      std::FILE** file;
      curl_slist** headers;
      ~Cleanup() {
        if (*file != nullptr) std::fclose(*file);
        if (*headers != nullptr) {
          curl_easy_setopt(handle, CURLOPT_HTTPHEADER, nullptr);
          curl_slist_free_all(*headers);
        }
      }
    } cleanup{handle, &file, &headers};

    curl_easy_setopt(handle, CURLOPT_URL, request.url.c_str());
    curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(handle, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(handle, CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(handle, CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(handle, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(handle, CURLOPT_USERAGENT, "inferx/0.0 model-resolver");
    const auto timeout =
        std::clamp<int64_t>(request.timeout.count(), 1, std::numeric_limits<long>::max());
    curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout));
    if (request.proxy.has_value() && !request.proxy->empty()) {
      curl_easy_setopt(handle, CURLOPT_PROXY, request.proxy->c_str());
    }
    for (const auto& [name, value] : request.headers) {
      curl_slist* appended = curl_slist_append(headers, absl::StrCat(name, ": ", value).c_str());
      if (appended == nullptr) {
        return absl::ResourceExhaustedError("cannot allocate HTTP headers");
      }
      headers = appended;
    }
    if (headers != nullptr) curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(handle, CURLOPT_HEADERFUNCTION, CollectHeader);
    curl_easy_setopt(handle, CURLOPT_HEADERDATA, &response.headers);
    if (request.sink.has_value()) {
      file = std::fopen(request.sink->string().c_str(), "wb");
      if (file == nullptr) {
        return absl::InternalError(
            absl::StrCat("cannot open download staging file \"", request.sink->string(), "\""));
      }
      curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, WriteToFile);
      curl_easy_setopt(handle, CURLOPT_WRITEDATA, file);
    } else {
      curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, WriteToString);
      curl_easy_setopt(handle, CURLOPT_WRITEDATA, &string_sink);
    }

    const CURLcode result = curl_easy_perform(handle);
    if (result != CURLE_OK) {
      if (string_sink.exceeded) {
        return absl::ResourceExhaustedError("Hugging Face metadata response exceeds 16 MiB");
      }
      if (result == CURLE_OPERATION_TIMEDOUT) {
        return absl::DeadlineExceededError(absl::StrCat("timed out fetching ", request.url));
      }
      return absl::UnavailableError(
          absl::StrCat("cannot fetch ", request.url, ": ", curl_easy_strerror(result)));
    }
    if (file != nullptr) {
      if (std::fclose(file) != 0) {
        file = nullptr;
        return absl::DataLossError(
            absl::StrCat("cannot flush download staging file for ", request.url));
      }
      file = nullptr;
    }
    long status_code = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status_code);
    response.status_code = static_cast<int>(status_code);
    return response;
  }

 private:
  static CURL* ThreadHandle() {
    struct Handle {
      CURL* value = curl_easy_init();
      ~Handle() {
        if (value != nullptr) curl_easy_cleanup(value);
      }
    };
    static thread_local Handle handle;
    return handle.value;
  }
};

}  // namespace

HttpClient& DefaultHttpClient() {
  static CurlHttpClient* client = new CurlHttpClient();
  return *client;
}

}  // namespace inferx::artifacts::internal
