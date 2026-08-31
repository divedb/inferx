#pragma once

#include <simdjson.h>

#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace inferx::artifacts::internal {

inline absl::Status JsonError(std::string_view path, std::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat(path, ": ", message));
}

inline absl::Status ParseError(std::string_view artifact, simdjson::error_code error) {
  return absl::DataLossError(
      absl::StrCat(artifact, ": invalid JSON: ", simdjson::error_message(error)));
}

inline absl::StatusOr<simdjson::dom::object> Object(simdjson::dom::element element,
                                                    std::string_view path) {
  simdjson::dom::object object;
  const auto error = element.get_object().get(object);
  if (error) return JsonError(path, "expected object");
  return object;
}

inline absl::StatusOr<simdjson::dom::array> Array(simdjson::dom::element element,
                                                  std::string_view path) {
  simdjson::dom::array array;
  const auto error = element.get_array().get(array);
  if (error) return JsonError(path, "expected array");
  return array;
}

inline absl::StatusOr<simdjson::dom::element> Required(simdjson::dom::object object,
                                                       std::string_view key,
                                                       std::string_view path) {
  simdjson::dom::element value;
  const auto error = object.at_key(key).get(value);
  if (error == simdjson::NO_SUCH_FIELD) {
    return JsonError(absl::StrCat(path, "/", key), "required field missing");
  }
  if (error) return JsonError(path, simdjson::error_message(error));
  return value;
}

inline absl::StatusOr<std::string> String(simdjson::dom::element element, std::string_view path) {
  std::string_view value;
  const auto error = element.get_string().get(value);
  if (error) return JsonError(path, "expected string");
  return std::string(value);
}

inline absl::StatusOr<uint64_t> Uint64(simdjson::dom::element element, std::string_view path) {
  uint64_t value = 0;
  const auto error = element.get_uint64().get(value);
  if (error) return JsonError(path, "expected nonnegative uint64 integer");
  return value;
}

inline absl::StatusOr<int64_t> Int64(simdjson::dom::element element, std::string_view path) {
  int64_t value = 0;
  const auto error = element.get_int64().get(value);
  if (error) return JsonError(path, "expected int64 integer");
  return value;
}

inline absl::StatusOr<double> FiniteDouble(simdjson::dom::element element, std::string_view path) {
  double value = 0;
  const auto error = element.get_double().get(value);
  if (error || !std::isfinite(value)) {
    return JsonError(path, "expected finite number");
  }
  return value;
}

inline absl::StatusOr<bool> Bool(simdjson::dom::element element, std::string_view path) {
  bool value = false;
  const auto error = element.get_bool().get(value);
  if (error) return JsonError(path, "expected boolean");
  return value;
}

inline absl::Status CheckUniqueKeys(simdjson::dom::object object, std::string_view path) {
  std::set<std::string> keys;
  for (const auto field : object) {
    const std::string key(field.key);
    if (!keys.insert(key).second) {
      return JsonError(absl::StrCat(path, "/", key), "duplicate key");
    }
  }
  return absl::OkStatus();
}

}  // namespace inferx::artifacts::internal
