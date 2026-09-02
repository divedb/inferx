#include "tokenizer/internal/detail/json_bridge.h"

#include <cmath>
#include <limits>

namespace tokenizer {
namespace {

const Json* FindMember(const Json& obj, std::string_view key) {
  if (!obj.is_object()) return nullptr;
  auto it = obj.find(std::string(key));
  if (it == obj.end() || it->is_null()) return nullptr;
  return &*it;
}

}  // namespace

StatusOr<Json> ParseJson(std::string_view text, std::string_view what) {
  Json parsed = Json::parse(text, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (parsed.is_discarded()) {
    return InvalidArgumentError(what, " is not valid JSON");
  }
  return parsed;
}

std::optional<std::string> GetString(const Json& obj, std::string_view key) {
  const Json* member = FindMember(obj, key);
  if (member == nullptr || !member->is_string()) return std::nullopt;
  return member->get<std::string>();
}

std::optional<bool> GetBool(const Json& obj, std::string_view key) {
  const Json* member = FindMember(obj, key);
  if (member == nullptr || !member->is_boolean()) return std::nullopt;
  return member->get<bool>();
}

std::optional<int64_t> GetInt(const Json& obj, std::string_view key) {
  const Json* member = FindMember(obj, key);
  if (member == nullptr || !member->is_number_integer()) return std::nullopt;
  return member->get<int64_t>();
}

std::vector<std::string> GetStringArray(const Json& obj, std::string_view key) {
  std::vector<std::string> out;
  const Json* member = FindMember(obj, key);
  if (member == nullptr || !member->is_array()) return out;
  for (const Json& entry : *member) {
    if (entry.is_string()) out.push_back(entry.get<std::string>());
  }
  return out;
}

std::optional<int64_t> GetClampedInt(const Json& obj, std::string_view key,
                                     int64_t fallback_for_huge) {
  const Json* member = FindMember(obj, key);
  if (member == nullptr) return std::nullopt;

  if (member->is_number_integer()) {
    // nlohmann parses an integer literal too large for int64_t as a double,
    // so reaching here means it fits.
    return member->get<int64_t>();
  }
  if (member->is_number_float()) {
    const double value = member->get<double>();
    if (!std::isfinite(value)) return fallback_for_huge;
    if (value >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
      return fallback_for_huge;
    }
    if (value <= 0) return std::nullopt;
    return static_cast<int64_t>(value);
  }
  return std::nullopt;
}

}  // namespace tokenizer
