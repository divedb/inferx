#include "inferx/config/config_loader.h"

#include <cstdlib>
#include <fstream>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "inferx/base/status.h"
#include "inferx/config/parser_limits.h"
#include "src/config/internal/simdjson_reader.h"

namespace inferx::config {

namespace {

absl::Status FieldError(absl::string_view source, absl::string_view name,
                        absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat(source, ".", name, ": ", message));
}

// Applies one overlay layer; every key must be a known field.
absl::Status ApplyLayer(const FieldValues& layer, ConfigSource source,
                        absl::string_view source_label, ParsedConfig& parsed) {
  for (const auto& [name, value] : layer) {
    bool applied = false;
#define INFERX_CONFIG_SET(camel, json_name, default_value) \
  if (name == (json_name)) {                               \
    parsed.Set##camel(value, source);                      \
    applied = true;                                        \
  } else
    INFERX_CONFIG_FIELDS(INFERX_CONFIG_SET)
    INFERX_CUDA_CONFIG_FIELDS(INFERX_CONFIG_SET)
    (void)0;  // terminate the generated else-chain
#undef INFERX_CONFIG_SET
    if (!applied) {
      return FieldError(source_label, name, "unknown field");
    }
  }
  return absl::OkStatus();
}

std::string EnvironmentName(absl::string_view json_name) {
  std::string upper(json_name);
  for (char& character : upper) {
    character = character == '_' || character == '.'
                    ? '_'
                    : static_cast<char>(absl::ascii_toupper(static_cast<unsigned char>(character)));
  }
  return "INFERX_" + upper;
}

}  // namespace

absl::StatusOr<uint64_t> ParseConfigInteger(absl::string_view text, absl::string_view source_name) {
  if (text.empty()) {
    return FieldError("environment", source_name, "value is empty");
  }
  uint64_t value = 0;
  for (const char character : text) {
    if (character < '0' || character > '9') {
      return FieldError("environment", source_name,
                        absl::StrCat("value '", text, "' is not an unsigned integer"));
    }
    const uint64_t digit = static_cast<uint64_t>(character - '0');
    if (value > (UINT64_MAX - digit) / 10) {
      return FieldError("environment", source_name,
                        absl::StrCat("value '", text, "' overflows uint64"));
    }
    value = value * 10 + digit;
  }
  return value;
}

absl::StatusOr<FieldValues> ParseConfigJson(absl::string_view json_text) {
  absl::StatusOr<FieldValues> parsed =
      internal::ParseFlatIntegerObject(json_text, kMaxConfigFileBytes);
  if (!parsed.ok()) {
    return parsed;
  }
  if (const auto schema = parsed->find("schema_version"); schema != parsed->end()) {
    if (schema->second != 1 && schema->second != 2) {
      return FieldError("config", "schema_version", "must be 1 or 2");
    }
    parsed->erase(schema);
  }
  // Unknown JSON keys are errors at the config boundary (ADR 0011), not
  // only when a layer is applied.
  for (const auto& [name, value] : *parsed) {
    bool known = false;
#define INFERX_KNOWN(camel, json_name, default_value) \
  if (name == (json_name)) {                          \
    known = true;                                     \
  }
    INFERX_CONFIG_FIELDS(INFERX_KNOWN)
    INFERX_CUDA_CONFIG_FIELDS(INFERX_KNOWN)
#undef INFERX_KNOWN
    (void)value;
    if (!known) {
      return FieldError("config", name, "unknown field");
    }
  }
  return parsed;
}

absl::StatusOr<FieldValues> ReadConfigFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return absl::NotFoundError(absl::StrCat("config.file: cannot open '", path, "'"));
  }
  const std::streampos size = file.tellg();
  if (static_cast<uint64_t>(size) > kMaxConfigFileBytes) {
    return absl::OutOfRangeError(
        absl::StrCat("config.file: size exceeds limit ", kMaxConfigFileBytes));
  }
  file.seekg(0, std::ios::beg);
  std::string text(static_cast<size_t>(size), '\0');
  if (size > 0) {
    file.read(text.data(), size);
  }
  return ParseConfigJson(text);
}

absl::StatusOr<FieldValues> ReadConfigEnvironment() {
  FieldValues values;
#define INFERX_CONFIG_ENV(camel, json_name, default_value)                   \
  if (const char* raw = std::getenv(EnvironmentName((json_name)).c_str())) { \
    absl::StatusOr<uint64_t> parsed = ParseConfigInteger(raw, (json_name));  \
    if (!parsed.ok()) {                                                      \
      return parsed.status();                                                \
    }                                                                        \
    values.emplace((json_name), *parsed);                                    \
  }
  INFERX_CONFIG_FIELDS(INFERX_CONFIG_ENV)
  INFERX_CUDA_CONFIG_FIELDS(INFERX_CONFIG_ENV)
#undef INFERX_CONFIG_ENV
  return values;
}

absl::StatusOr<ParsedConfig> LoadConfig(const std::optional<FieldValues>& file_values,
                                        const std::optional<FieldValues>& environment_values,
                                        const std::optional<FieldValues>& command_line_values) {
  ParsedConfig parsed;
  // Invalid values in ANY layer are errors even when a later layer would
  // override them: broken deployment inputs must surface.
  if (file_values.has_value()) {
    absl::Status status = ApplyLayer(*file_values, ConfigSource::kFile, "config", parsed);
    if (!status.ok()) {
      return status;
    }
  }
  if (environment_values.has_value()) {
    absl::Status status =
        ApplyLayer(*environment_values, ConfigSource::kEnvironment, "environment", parsed);
    if (!status.ok()) {
      return status;
    }
  }
  if (command_line_values.has_value()) {
    absl::Status status =
        ApplyLayer(*command_line_values, ConfigSource::kCommandLine, "flags", parsed);
    if (!status.ok()) {
      return status;
    }
  }
  return parsed;
}

}  // namespace inferx::config
