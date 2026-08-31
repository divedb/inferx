#include "simdjson_reader.h"

#include <set>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/config/parser_limits.h"
#include "simdjson.h"

namespace inferx::config::internal {

namespace {

absl::Status Invalid(absl::string_view field, absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat("config.", field, ": ", message));
}

}  // namespace

absl::StatusOr<std::map<std::string, uint64_t>> ParseFlatIntegerObject(absl::string_view text,
                                                                       uint64_t max_bytes) {
  if (text.size() > max_bytes) {
    return Invalid("__file__", absl::StrCat("size ", text.size(), " exceeds limit ", max_bytes));
  }
  simdjson::dom::parser parser;
  simdjson::dom::element root;
  // simdjson's C++ API signals errors through error_code values rather than
  // throwing; the allocate/parse call itself is wrapped defensively so no
  // dependency exception can cross the adapter boundary (ADR 0004).
  try {
    const simdjson::error_code error = parser.parse(text.data(), text.size()).get(root);
    if (error != simdjson::SUCCESS) {
      return Invalid("__json__", simdjson::error_message(error));
    }
  } catch (const std::exception& exception) {
    return Invalid("__json__", absl::StrCat("parser threw: ", exception.what()));
  }

  simdjson::dom::object object;
  if (const simdjson::error_code error = root.get(object); error != simdjson::SUCCESS) {
    return Invalid("__json__", "top-level value must be an object");
  }

  std::map<std::string, uint64_t> values;
  std::set<std::string> seen;
  for (auto [key, value] : object) {
    std::string name(key);
    if (!seen.insert(name).second) {
      return Invalid(name, "duplicate field");
      // name stays usable below; the emplace copies are tiny and rare.
    }
    if (values.size() >= static_cast<size_t>(kMaxObjectMembers)) {
      return Invalid(name, absl::StrCat("more than ", kMaxObjectMembers, " members"));
    }
    uint64_t parsed = 0;
    if (const simdjson::error_code error = value.get(parsed); error != simdjson::SUCCESS) {
      return Invalid(name, "must be an unsigned integer");
    }
    values.emplace(std::move(name), parsed);
  }
  return values;
}

}  // namespace inferx::config::internal
