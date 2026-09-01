// Configuration loading: source precedence and strict integer parsing
// (m1.md sections 8.1-8.2). JSON is read by a private simdjson adapter; no
// simdjson type appears in this public header.

#ifndef INFERX_CONFIG_CONFIG_LOADER_H_
#define INFERX_CONFIG_CONFIG_LOADER_H_

#include <map>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "inferx/config/parsed_config.h"

namespace inferx::config {

// Raw per-source field overrides, already keyed by their documented spelling
// (JSON name / INFERX_<UPPERCASE> / --name). The loader applies them in
// precedence order and rejects unknown names, non-integer values, and range
// overflow without partially mutating the result.
using FieldValues = std::map<std::string, uint64_t>;  // sorted: stable errors

// Parses strict JSON object text into field values. Rejects unknown fields,
// duplicate keys, non-integer values, nesting, oversized input, and invalid
// UTF-8 before any value is used.
absl::StatusOr<FieldValues> ParseConfigJson(absl::string_view json_text);

// Reads a config file after the byte-size limit check.
absl::StatusOr<FieldValues> ReadConfigFile(const std::string& path);

// Collects documented INFERX_* variables from the process environment.
// Only documented names are read; unrelated environment entries are ignored.
absl::StatusOr<FieldValues> ReadConfigEnvironment();

// Strict unsigned decimal parsing used for environment/CLI values.
absl::StatusOr<uint64_t> ParseConfigInteger(absl::string_view text, absl::string_view source_name);

// Applies overlays in precedence order (file < environment < CLI) on top of
// defaults. Every provided value must parse and be a known field; failures
// name the field and source.
absl::StatusOr<ParsedConfig> LoadConfig(const std::optional<FieldValues>& file_values,
                                        const std::optional<FieldValues>& environment_values,
                                        const std::optional<FieldValues>& command_line_values);

}  // namespace inferx::config

#endif  // INFERX_CONFIG_CONFIG_LOADER_H_
