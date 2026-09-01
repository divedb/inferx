#include "src/artifacts/hf_http_client.h"

#include <cctype>

namespace inferx::artifacts::internal {

std::optional<std::string> HttpResponse::Header(std::string_view name) const {
  for (auto iterator = headers.rbegin(); iterator != headers.rend(); ++iterator) {
    const auto& [key, value] = *iterator;
    if (key.size() != name.size()) continue;
    bool equal = true;
    for (size_t index = 0; index < key.size(); ++index) {
      if (std::tolower(static_cast<unsigned char>(key[index])) !=
          std::tolower(static_cast<unsigned char>(name[index]))) {
        equal = false;
        break;
      }
    }
    if (equal) return value;
  }
  return std::nullopt;
}

}  // namespace inferx::artifacts::internal
