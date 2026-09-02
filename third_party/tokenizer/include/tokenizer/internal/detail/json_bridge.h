#pragma once

// INTERNAL. This is the only place the vendored JSON library is allowed to
// appear alongside our own types. Nothing under include/ may include it.

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/core/json_value.h"
#include "tokenizer/core/status.h"

namespace tokenizer {

using Json = nlohmann::json;
using OrderedJson = nlohmann::ordered_json;

/// \brief Converts between the vendored JSON type and our public handle.
///
/// A friend of JsonValue so it can reach the pimpl; the conversion is a copy
/// in both directions, which is fine for the small values that cross the
/// public API.
class JsonBridge {
 public:
  static JsonValue ToPublic(const Json& value);
  static const Json& ToInternal(const JsonValue& value);
  static Json& MutableInternal(JsonValue& value);
  static OrderedJson ToOrdered(const JsonValue& value);
};

/// \brief Reads a whole file. NotFound when it does not exist.
/// \brief Parses `text`. InvalidArgument carries the parser's own message.
StatusOr<Json> ParseJson(std::string_view text, std::string_view what);

// --- Safe accessors -------------------------------------------------------
//
// Hugging Face configs are hand-edited and inconsistent, so every getter here
// tolerates a missing key and a wrong type rather than throwing. A field whose
// type we do not understand is treated as absent; the caller's default wins.

std::optional<std::string> GetString(const Json& obj, std::string_view key);
std::optional<bool> GetBool(const Json& obj, std::string_view key);
std::optional<int64_t> GetInt(const Json& obj, std::string_view key);
std::vector<std::string> GetStringArray(const Json& obj, std::string_view key);

/// \brief Reads an integer that may be written as a float too large for
///        int64_t.
///
/// gpt-oss-20b declares `model_max_length: 1000000000000000019884624838656`,
/// which overflows. Values out of range clamp to `fallback_for_huge` rather
/// than wrapping into something small and wrong.
std::optional<int64_t> GetClampedInt(const Json& obj, std::string_view key,
                                     int64_t fallback_for_huge);

}  // namespace tokenizer
